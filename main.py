from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import torch
import numpy as np
import json
import io
import random
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Part Identifier API")

# ── CORS — allows your React app to call this server ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your React port in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
CATALOG_PATH    = Path("catalog.json")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
COLOR_WEIGHT    = 0.45
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
logger.info(f"Loading DINOv2 on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
logger.info("✅ Model ready")

# ── CATALOG HELPERS ──────────────────────────────────────────────────────────
def load_catalog():
    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH) as f:
                data = json.load(f)
            return {
                k: {"dino": np.array(v["dino"]), "color": np.array(v["color"])}
                for k, v in data.items()
            }
        except Exception as e:
            logger.error(f"Catalog load error: {e}")
            return {}
    return {}

def save_catalog(catalog):
    try:
        with open(CATALOG_PATH, "w") as f:
            json.dump(
                {k: {"dino": v["dino"].tolist(), "color": v["color"].tolist()}
                 for k, v in catalog.items()}, f
            )
    except Exception as e:
        logger.error(f"Catalog save error: {e}")
        raise HTTPException(500, "Failed to save catalog")

# ── FEATURE EXTRACTION ───────────────────────────────────────────────────────
@torch.no_grad()
def embed_batch(pil_images: list) -> np.ndarray:
    inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
    out    = model(**inputs)
    embs   = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(embs, dim=-1).cpu().numpy()

def color_histogram(pil_img, bins=32) -> np.ndarray:
    img = np.array(pil_img.resize((224, 224))).astype(np.float32)
    # Grey world color constancy — removes lighting color cast
    mean_r = img[:,:,0].mean()
    mean_g = img[:,:,1].mean()
    mean_b = img[:,:,2].mean()
    grey   = (mean_r + mean_g + mean_b) / 3
    img[:,:,0] = np.clip(img[:,:,0] * (grey / (mean_r + 1e-8)), 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * (grey / (mean_g + 1e-8)), 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * (grey / (mean_b + 1e-8)), 0, 255)
    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        h = h.flatten() / (h.sum() + 1e-8)
        hist.append(h)
    vec = np.concatenate(hist)
    return vec / (np.linalg.norm(vec) + 1e-8)

def augment(img: Image.Image, n: int = 4) -> list:
    variants = [img]
    for _ in range(n - 1):
        aug = img.copy()
        if random.random() > 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        aug = aug.rotate(random.uniform(-15, 15), fillcolor=(128, 128, 128))
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.85, 1.15))
        aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.85, 1.15))
        w, h = aug.size
        s = random.uniform(0.92, 1.0)
        nw, nh = int(w * s), int(h * s)
        l, t = random.randint(0, w - nw), random.randint(0, h - nh)
        aug = aug.crop((l, t, l + nw, t + nh)).resize((w, h))
        variants.append(aug)
    return variants

def extract_features(pil_images: list):
    dino_vec  = embed_batch(pil_images).mean(axis=0)
    dino_vec  = dino_vec / (np.linalg.norm(dino_vec) + 1e-8)
    color_vec = np.mean([color_histogram(im) for im in pil_images], axis=0)
    color_vec = color_vec / (np.linalg.norm(color_vec) + 1e-8)
    return dino_vec, color_vec

def fused_similarity(stored, q_dino, q_color):
    dino_sim  = float(np.dot(stored["dino"],  q_dino))
    color_sim = float(np.dot(stored["color"], q_color))
    combined  = (1.0 - COLOR_WEIGHT) * dino_sim + COLOR_WEIGHT * color_sim
    return combined, dino_sim, color_sim

# ── ROUTES ───────────────────────────────────────────────────────────────────

# GET /health
@app.get("/health")
def health():
    catalog = load_catalog()
    return {
        "status":       "ok",
        "device":       DEVICE,
        "catalog_size": len(catalog),
        "parts":        list(catalog.keys())
    }

# POST /api/register
@app.post("/api/register")
async def register(
    part_id: str              = Form(...),
    images:  list[UploadFile] = File(...)
):
    if len(images) < 3:
        raise HTTPException(400, "Upload at least 3 images")
    if len(images) > 50:
        raise HTTPException(400, "Max 50 images per registration")

    all_imgs = []
    for f in images:
        raw = await f.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Image '{f.filename}' exceeds 10MB")
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Could not read image '{f.filename}'")
        all_imgs.extend(augment(img, n=4))

    try:
        dino_feat, color_feat = await run_in_threadpool(extract_features, all_imgs)
    except Exception as e:
        logger.exception("Feature extraction failed during /api/register")
        raise HTTPException(500, f"Model inference failed: {e}")

    catalog = load_catalog()
    catalog[part_id] = {"dino": dino_feat, "color": color_feat}
    save_catalog(catalog)

    logger.info(f"Registered '{part_id}' with {len(images)} images")
    return {
        "status":      "registered",
        "message":     f"'{part_id}' registered successfully with {len(images)} images",
        "part_id":     part_id,
        "images_used": len(images),
        "total_parts": len(catalog)
    }

# POST /api/identify
@app.post("/api/identify")
async def identify(
    image:     UploadFile = File(...),
    threshold: float      = Form(0.78),
    top_k:     int        = Form(5)
):
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(400, "Catalog is empty — register parts first")

    raw = await image.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds 10MB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")

    def _compute_query_features():
        augs = augment(img, n=5)
        return extract_features(augs)

    try:
        q_dino, q_color = await run_in_threadpool(_compute_query_features)
    except Exception as e:
        logger.exception("Feature extraction failed during /api/identify")
        raise HTTPException(500, f"Model inference failed: {e}")

    results = []
    skipped = []
    for part_id, features in catalog.items():
        try:
            combined, dino_sim, color_sim = fused_similarity(features, q_dino, q_color)
        except ValueError as e:
            logger.warning(f"Skipping '{part_id}' — incompatible stored vectors ({e}). "
                            f"Re-register this part with the current main.py.")
            skipped.append(part_id)
            continue
        results.append({
            "part_id":    part_id,
            "similarity": round(combined, 4),
            "debug":      {"shape": round(dino_sim, 3), "color": round(color_sim, 3)}
        })

    if not results:
        raise HTTPException(
            500,
            f"All {len(skipped)} catalog entries are incompatible with the current model "
            f"({', '.join(skipped)}). Re-register these parts."
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top = results[:top_k]

    return {
        "predicted":  top[0]["part_id"] if top[0]["similarity"] >= threshold else None,
        "confidence": top[0]["similarity"],
        "matched":    top[0]["similarity"] >= threshold,
        "threshold":  threshold,
        "top_k":      top,
        "skipped":    skipped
    }

# GET /api/catalog
@app.get("/api/catalog")
def get_catalog():
    catalog = load_catalog()
    return {
        "parts":  list(catalog.keys()),
        "total":  len(catalog)
    }

# GET /api/catalog/{part_id}
@app.get("/api/catalog/{part_id}")
def get_part(part_id: str):
    catalog = load_catalog()
    if part_id not in catalog:
        raise HTTPException(404, f"'{part_id}' not in catalog")
    return {
        "part_id":      part_id,
        "registered":   True,
    }

# DELETE /api/catalog/{part_id}
@app.delete("/api/catalog/{part_id}")
def delete_part(part_id: str):
    catalog = load_catalog()
    if part_id not in catalog:
        raise HTTPException(404, f"'{part_id}' not found")
    del catalog[part_id]
    save_catalog(catalog)
    return {
        "status":    "removed",
        "part_id":   part_id,
        "remaining": len(catalog)
    }

# DELETE /api/catalog
@app.delete("/api/catalog")
def clear_catalog():
    save_catalog({})
    return {"status": "cleared", "message": "All parts removed from catalog"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
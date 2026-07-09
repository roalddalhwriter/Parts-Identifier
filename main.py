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
from ultralytics import SAM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Part Identifier API")

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
CATALOG_PATH    = Path("catalog.json")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
COLOR_WEIGHT    = 0.45
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB

# ── LOAD MODELS ──────────────────────────────────────────────────────────────
logger.info(f"Loading DINOv2 on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()

logger.info(f"Loading MobileSAM on {DEVICE}...")
sam_model = SAM("mobile_sam.pt") 
logger.info("✅ Models ready")

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

# ── MASK GENERATION ──────────────────────────────────────────────────────────
def get_object_mask(pil_img: Image.Image) -> np.ndarray:
    """Returns a binary boolean mask of the central object using MobileSAM."""
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    center_point = [[w // 2, h // 2]]
    results = sam_model.predict(img_bgr, points=center_point, labels=[1], verbose=False)
    
    if not results or len(results[0].masks.data) == 0:
        return np.ones((h, w), dtype=bool)
        
    mask = results[0].masks.data[0].cpu().numpy().astype(bool)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask

# ── FEATURE EXTRACTION ───────────────────────────────────────────────────────
@torch.no_grad()
def embed_batch(pil_images: list) -> np.ndarray:
    inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
    out    = model(**inputs)
    embs   = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(embs, dim=-1).cpu().numpy()

def color_histogram(pil_img, mask=None, bins=32) -> np.ndarray:
    img = np.array(pil_img.resize((224, 224))).astype(np.float32)
    
    if mask is not None:
        mask_cv = cv2.resize(mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
    else:
        mask_cv = np.ones((224, 224), dtype=np.uint8) * 255

    # Grey world color constancy calculated ONLY on foreground pixels
    if np.any(mask_cv > 0):
        mean_r = img[mask_cv > 0, 0].mean()
        mean_g = img[mask_cv > 0, 1].mean()
        mean_b = img[mask_cv > 0, 2].mean()
        grey = (mean_r + mean_g + mean_b) / 3
        img[:,:,0] = np.clip(img[:,:,0] * (grey / (mean_r + 1e-8)), 0, 255)
        img[:,:,1] = np.clip(img[:,:,1] * (grey / (mean_g + 1e-8)), 0, 255)
        img[:,:,2] = np.clip(img[:,:,2] * (grey / (mean_b + 1e-8)), 0, 255)

    img = img.astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = []
    
    # Pass the mask to calcHist to completely exclude the changing background canvas
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], mask_cv, [bins], [0, 256])
        h = h.flatten() / (h.sum() + 1e-8)
        hist.append(h)
    
    vec = np.concatenate(hist)
    return vec / (np.linalg.norm(vec) + 1e-8)

def augment_pair(img: Image.Image, mask: np.ndarray, n: int = 4) -> tuple[list, list]:
    """Augments both the image and its mask concurrently to keep them aligned."""
    img_variants = [img]
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_variants = [mask_pil]
    
    for _ in range(n - 1):
        aug_img = img.copy()
        aug_mask = mask_pil.copy()
        
        if random.random() > 0.5:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
            aug_mask = aug_mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        rot = random.uniform(-15, 15)
        aug_img = aug_img.rotate(rot, fillcolor=(128, 128, 128))
        aug_mask = aug_mask.rotate(rot, fillcolor=0)
        
        # Color adjustments happen ONLY to the image
        aug_img = ImageEnhance.Brightness(aug_img).enhance(random.uniform(0.85, 1.15))
        aug_img = ImageEnhance.Contrast(aug_img).enhance(random.uniform(0.85, 1.15))
        
        w, h = aug_img.size
        s = random.uniform(0.92, 1.0)
        nw, nh = int(w * s), int(h * s)
        l, t = random.randint(0, w - nw), random.randint(0, h - nh)
        
        aug_img = aug_img.crop((l, t, l + nw, t + nh)).resize((w, h))
        aug_mask = aug_mask.crop((l, t, l + nw, t + nh)).resize((w, h), Image.NEAREST)
        
        img_variants.append(aug_img)
        mask_variants.append(aug_mask)
        
    # Convert mask PIL variants back to boolean numpy arrays
    mask_np_variants = [np.array(m) > 0 for m in mask_variants]
    return img_variants, mask_np_variants

def extract_features(pil_images: list, masks: list = None):
    # DINOv2 evaluates the raw, clean unmasked background scenes
    dino_vec  = embed_batch(pil_images).mean(axis=0)
    dino_vec  = dino_vec / (np.linalg.norm(dino_vec) + 1e-8)
    
    if masks is None:
        masks = [None] * len(pil_images)
        
    color_vec = np.mean([color_histogram(im, m) for im, m in zip(pil_images, masks)], axis=0)
    color_vec = color_vec / (np.linalg.norm(color_vec) + 1e-8)
    return dino_vec, color_vec

def fused_similarity(stored, q_dino, q_color):
    dino_sim  = float(np.dot(stored["dino"],  q_dino))
    color_sim = float(np.dot(stored["color"], q_color))
    combined  = (1.0 - COLOR_WEIGHT) * dino_sim + COLOR_WEIGHT * color_sim
    return combined, dino_sim, color_sim

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    catalog = load_catalog()
    return {
        "status":       "ok",
        "device":       DEVICE,
        "catalog_size": len(catalog),
        "parts":        list(catalog.keys())
    }

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
    all_masks = []
    for f in images:
        raw = await f.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Image '{f.filename}' exceeds 10MB")
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Could not read image '{f.filename}'")
            
        mask = get_object_mask(img)
        aug_imgs, aug_masks = augment_pair(img, mask, n=4)
        all_imgs.extend(aug_imgs)
        all_masks.extend(aug_masks)

    try:
        dino_feat, color_feat = await run_in_threadpool(extract_features, all_imgs, all_masks)
    except Exception as e:
        logger.exception("Feature extraction failed during /api/register")
        raise HTTPException(500, f"Model inference failed: {e}")

    catalog = load_catalog()
    catalog[part_id] = {"dino": dino_feat, "color": color_feat}
    save_catalog(catalog)

    return {
        "status":      "registered",
        "part_id":     part_id,
        "images_used": len(images)
    }

@app.post("/api/identify")
async def identify(
    image:     UploadFile = File(...),
    threshold: float      = Form(0.78),
    top_k:     int        = Form(5)
):
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(400, "Catalog is empty")

    raw = await image.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds 10MB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")

    def _compute_query_features():
        mask = get_object_mask(img)
        aug_imgs, aug_masks = augment_pair(img, mask, n=5)
        return extract_features(aug_imgs, aug_masks)

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
            skipped.append(part_id)
            continue
        results.append({
            "part_id":    part_id,
            "similarity": round(combined, 4),
            "debug":      {"shape": round(dino_sim, 3), "color": round(color_sim, 3)}
        })

    if not results:
        raise HTTPException(500, "All catalog entries are incompatible.")

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

@app.get("/api/catalog")
def get_catalog():
    return {"parts": list(load_catalog().keys())}

@app.delete("/api/catalog/{part_id}")
def delete_part(part_id: str):
    catalog = load_catalog()
    if part_id not in catalog:
        raise HTTPException(404, "Not found")
    del catalog[part_id]
    save_catalog(catalog)
    return {"status": "removed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
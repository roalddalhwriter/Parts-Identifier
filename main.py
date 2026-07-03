from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import json
import io
import random
from PIL import Image, ImageEnhance
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

app = FastAPI(title="Scale Reader - DINO Color API")

# ── CORS MIDDLEWARE (Mirrored exactly from your old code) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Critical for external apps and mobile testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
CATALOG_PATH = Path("catalog.json")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COLOR_IMPORTANCE_WEIGHT = 0.70  

# ── LAZY / STARTUP MODEL CONFIG ──────────────────────────────────────────────
print(f"Loading DINOv2 on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
print("✅ Model ready")

def load_catalog():
    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH) as f:
                data = json.load(f)
            return {k: {"dino": np.array(v["dino"]), "color": np.array(v["color"])} for k, v in data.items()}
        except Exception:
            return {}
    return {}

def save_catalog(catalog):
    with open(CATALOG_PATH, "w") as f:
        serializable = {
            k: {"dino": v["dino"].tolist(), "color": v["color"].tolist()} 
            for k, v in catalog.items()
        }
        json.dump(serializable, f)

# ── FEATURE DESCRIPTORS ──────────────────────────────────────────────────────
@torch.no_grad()
def embed_batch(pil_images: list) -> np.ndarray:
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out    = model(**inputs)
    embs   = out.last_hidden_state[:, 0]
    embs   = torch.nn.functional.normalize(embs, dim=-1)
    return embs.cpu().numpy()

def extract_color_fingerprint(pil_img):
    w, h = pil_img.size
    left, top, right, bottom = int(w * 0.225), int(h * 0.225), int(w * 0.775), int(h * 0.775)
    center_crop = pil_img.crop((left, top, right, bottom))
    
    small_grid = center_crop.resize((8, 8), Image.Resampling.BILINEAR)
    grid_array = np.array(small_grid).astype(np.float32) / 255.0
    color_vector = grid_array.flatten()
    return color_vector / (np.linalg.norm(color_vector) + 1e-8)

def augment(img: Image.Image, n: int = 4) -> list:
    variants = [img]
    for _ in range(n - 1):
        aug = img.copy()
        if random.random() > 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        aug = aug.rotate(random.uniform(-10, 10), fillcolor=(128, 128, 128))
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.95, 1.05))
        w, h = aug.size
        s = random.uniform(0.97, 1.0)
        nw, nh = int(w * s), int(h * s)
        l, t = random.randint(0, w - nw), random.randint(0, h - nh)
        aug = aug.crop((l, t, l + nw, t + nh)).resize((w, h))
        variants.append(aug)
    return variants

def get_separate_features(pil_images: list):
    dino_vec  = embed_batch(pil_images).mean(axis=0)                          
    dino_vec  = dino_vec / np.linalg.norm(dino_vec)
    
    color_vec = np.mean([extract_color_fingerprint(im) for im in pil_images], axis=0)  
    color_vec = color_vec / np.linalg.norm(color_vec)
    
    return dino_vec, color_vec

# ── PURE HEADLESS JSON ENDPOINTS ─────────────────────────────────────────────
@app.post("/api/register")
async def register(
    part_id: str              = Form(...),
    images:  list[UploadFile] = File(...)
):
    if len(images) < 3:
        raise HTTPException(400, "Upload at least 3 images")

    all_imgs = []
    for f in images:
        raw = await f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        all_imgs.extend(augment(img, n=4))

    dino_feat, color_feat = get_separate_features(all_imgs)
    
    catalog = load_catalog()
    catalog[part_id] = {"dino": dino_feat, "color": color_feat}
    save_catalog(catalog)

    return {
        "success": True,
        "part_id": part_id,
        "total_parts": len(catalog)
    }

@app.post("/api/identify")
async def identify(
    image:     UploadFile = File(...),
    threshold: float      = Form(0.78),
    top_k:     int        = Form(5)
):
    catalog = load_catalog()
    if not catalog:
        raise HTTPException(400, "Catalog empty")

    raw  = await image.read()
    img  = Image.open(io.BytesIO(raw)).convert("RGB")
    
    q_dino, q_color = get_separate_features([img])

    results = []
    for part_id, features in catalog.items():
        dino_sim  = float(np.dot(features["dino"], q_dino))
        color_sim = float(np.dot(features["color"], q_color))
        combined_sim = ((1.0 - COLOR_IMPORTANCE_WEIGHT) * dino_sim) + (COLOR_IMPORTANCE_WEIGHT * color_sim)
        
        results.append({
            "part_id": part_id,
            "similarity": round(combined_sim, 4)
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top_idx_results = results[:top_k]
    highest_match = top_idx_results[0]

    return {
        "success": highest_match["similarity"] >= threshold,
        "predicted": highest_match["part_id"] if highest_match["similarity"] >= threshold else None,
        "confidence": highest_match["similarity"],
        "matches": top_idx_results
    }

@app.get("/api/catalog")
def get_catalog():
    catalog = load_catalog()
    return {"parts": list(catalog.keys()), "total": len(catalog)}

@app.delete("/api/catalog/{part_id}")
def delete_part(part_id: str):
    catalog = load_catalog()
    if part_id not in catalog:
        raise HTTPException(404, f"'{part_id}' not found")
    del catalog[part_id]
    save_catalog(catalog)
    return {"success": True, "remaining": len(catalog)}
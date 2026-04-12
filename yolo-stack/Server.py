"""
Spark Sight — YOLO11 Warning Service
OpenAI VLM-style API wrapper for obstacle detection + distance-based warning.

POST /v1/chat/completions
  body: { model, messages: [{role, content: [{type:"image_url", image_url:{url:"data:image/jpeg;base64,..."}}]}] }
  returns: OpenAI-compatible ChatCompletion with warning text in message.content

POST /v1/detect/raw
  body: multipart image file
  returns: raw detection JSON {objects, warning_level, warning_text, latency_ms}
"""

import base64
import io
import time
import re
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MODEL_PATH = "yolo11n.engine"   # switch to "yolo11n.pt" if TRT not yet built
IMG_SIZE   = 640
CONF_THRES = 0.35
DEVICE     = 0                  # GPU 0

# Distance thresholds (metres) — tuned after camera calibration
# Relative depth from YOLO bbox area heuristic until DAv2 is integrated
THRESHOLDS = {
    "CRITICAL": 0.5,    # < 0.5m → immediate stop warning
    "WARNING":  1.5,    # 0.5–1.5m → slow down warning
    "CAUTION":  3.0,    # 1.5–3.0m → heads up
    # > 3.0m → CLEAR, silent
}

WARNING_TEMPLATES = {
    "CRITICAL": "Emergency: {dist:.1f}（{obj}",
    "WARNING":  "Warning:  {dist:.1f} {obj}",
    "CAUTION":  "Notice: {dist:.1f}  {obj}",
    "CLEAR":    "",
}

# YOLO COCO class ids that are navigation-relevant obstacles
# (person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7,
#  fire hydrant=10, stop sign=11, bench=13, dog=16, backpack=24,
#  umbrella=25, suitcase=28, chair=56, potted plant=58, etc.)
OBSTACLE_CLASSES = {
    0, 1, 2, 3, 5, 7, 10, 11, 13, 14, 15, 16,
    24, 25, 26, 28, 32, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67,
}

# ──────────────────────────────────────────────
# Depth heuristic (pre-calibration fallback)
# Replace estimate_distance() with real DAv2 output once integrated.
# ──────────────────────────────────────────────

def estimate_distance_heuristic(bbox_xyxy: list, img_w: int, img_h: int) -> float:
    """
    Rough monocular distance estimate from bbox area ratio.
    Assumes average obstacle ~0.5m wide at 2m distance on a 640px frame.
    Replace with calibrated depth map lookup once DAv2 is running.
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = img_w * img_h
    area_ratio = bbox_area / frame_area  # 0–1

    # Empirical: area_ratio ~0.04 → ~2m, ~0.25 → ~0.5m
    if area_ratio <= 0:
        return 10.0
    # Simple inverse-square approximation: dist ≈ k / sqrt(area_ratio)
    k = 0.4
    dist = k / (area_ratio ** 0.5)
    return round(min(max(dist, 0.1), 20.0), 2)


def classify_level(dist_m: float) -> str:
    if dist_m < THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif dist_m < THRESHOLDS["WARNING"]:
        return "WARNING"
    elif dist_m < THRESHOLDS["CAUTION"]:
        return "CAUTION"
    return "CLEAR"


def build_warning_text(level: str, dist_m: float, obj_name: str) -> str:
    template = WARNING_TEMPLATES.get(level, "")
    if not template:
        return ""
    return template.format(dist=dist_m, obj=obj_name)


# ──────────────────────────────────────────────
# Model lifecycle
# ──────────────────────────────────────────────

model: Optional[YOLO] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"[startup] loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    # warm-up pass — eliminates first-inference latency spike
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=IMG_SIZE, conf=CONF_THRES,
                  device=DEVICE, verbose=False)
    print("[startup] model ready")
    yield
    print("[shutdown] releasing model")
    del model

app = FastAPI(title="Spark Sight Warning API", lifespan=lifespan)


# ──────────────────────────────────────────────
# Core inference
# ──────────────────────────────────────────────

def run_detection(img: Image.Image) -> dict:
    t0 = time.perf_counter()
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]

    results = model.predict(
        img_np,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []
    closest_dist = 999.0
    closest_obj  = None
    closest_level = "CLEAR"

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in OBSTACLE_CLASSES:
            continue

        cls_name = results.names[cls_id]
        conf     = float(box.conf[0])
        xyxy     = box.xyxy[0].tolist()
        dist     = estimate_distance_heuristic(xyxy, w, h)
        level    = classify_level(dist)

        detections.append({
            "class":      cls_name,
            "confidence": round(conf, 3),
            "bbox_xyxy":  [round(v, 1) for v in xyxy],
            "distance_m": dist,
            "level":      level,
        })

        if dist < closest_dist:
            closest_dist  = dist
            closest_obj   = cls_name
            closest_level = level

    warning_text = build_warning_text(closest_level, closest_dist, closest_obj or "障碍物")
    latency_ms   = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "objects":      detections,
        "warning_level": closest_level,
        "warning_text": warning_text,
        "closest_m":    round(closest_dist, 2) if closest_obj else None,
        "latency_ms":   latency_ms,
    }


def decode_image_from_url(url: str) -> Image.Image:
    """Decode base64 data URL or plain base64 string."""
    if url.startswith("data:"):
        # data:image/jpeg;base64,<data>
        match = re.match(r"data:[^;]+;base64,(.+)", url, re.DOTALL)
        if not match:
            raise ValueError("malformed data URL")
        b64 = match.group(1)
    else:
        b64 = url
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ──────────────────────────────────────────────
# OpenAI-compatible endpoint
# ──────────────────────────────────────────────

class ImageURL(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str
    image_url: Optional[ImageURL] = None
    text: Optional[str] = None

class Message(BaseModel):
    role: str
    content: list[ContentPart] | str

class ChatRequest(BaseModel):
    model: str = "yolo11n-warning"
    messages: list[Message]
    max_tokens: int = 256
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Extract image from the last user message
    image: Optional[Image.Image] = None
    for msg in reversed(req.messages):
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str):
            continue
        for part in content:
            if part.type == "image_url" and part.image_url:
                try:
                    image = decode_image_from_url(part.image_url.url)
                except Exception as e:
                    raise HTTPException(400, f"image decode error: {e}")
                break
        if image:
            break

    if image is None:
        raise HTTPException(400, "no image_url found in messages")

    result = run_detection(image)
    content_text = result["warning_text"] or "CLEAR"

    # OpenAI ChatCompletion response schema
    return JSONResponse({
        "id":      f"chatcmpl-yolo-{int(time.time()*1000)}",
        "object":  "chat.completion",
        "model":   req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role":    "assistant",
                "content": content_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # non-standard extension — full detection payload
        "detection": result,
    })


# ──────────────────────────────────────────────
# Raw detection endpoint (for debugging / frontend)
# ──────────────────────────────────────────────

from fastapi import UploadFile, File

@app.post("/v1/detect/raw")
async def detect_raw(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "cannot parse uploaded image")
    result = run_detection(img)
    return JSONResponse(result)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, workers=1)
"""
Spark Sight — YOLO11 Warning Service
Focused on outdoor obstacles a white cane CANNOT detect:
  • Fast-moving objects  (cyclists, cars, motorcycles, skateboards …)
  • Aerial/suspended     (traffic lights, poles, umbrellas, signs …)

Zone-based interrupt policy
  CENTER     (middle 56 % width, above ground band) : WARNING + CRITICAL interrupt
  PERIPHERAL (outer edges, above ground band)        : CRITICAL only
  BOTTOM     (lower 28 % of frame = ground level)   : never interrupts

Throttle: at most one interrupting alert per 4 s window.
Exception: WARNING → CRITICAL escalation always breaks through.

POST /v1/chat/completions  — OpenAI VLM-style (image_url in messages)
POST /v1/detect/raw        — multipart file upload, full JSON payload
GET  /health
"""

from __future__ import annotations

import base64
import io
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "yolo11n.engine"   # swap to "yolo11n.pt" if TRT engine not ready
IMG_SIZE   = 640
CONF_THRES = 0.35
DEVICE     = 0                  # GPU 0

# ─────────────────────────────────────────────────────────────────────────────
# Obstacle classes  (COCO IDs)
#
# Only two categories — things a white cane reliably MISSES:
#   FAST_MOVING  — enter the path faster than cane feedback
#   AERIAL       — suspended above cane reach (pole, sign, umbrella …)
# ─────────────────────────────────────────────────────────────────────────────

FAST_MOVING_CLASSES: frozenset[int] = frozenset({
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    36,  # skateboard
})

AERIAL_CLASSES: frozenset[int] = frozenset({
    9,   # traffic light  (on pole, head height)
    11,  # stop sign      (on pole)
    12,  # parking meter  (waist–chest height)
    25,  # umbrella       (sidewalk café / hand-carried, face height)
    56,  # chair          (outdoor café seating)
    58,  # potted plant   (elevated planter, stoop display)
})

OBSTACLE_CLASSES: frozenset[int] = FAST_MOVING_CLASSES | AERIAL_CLASSES

# ─────────────────────────────────────────────────────────────────────────────
# Distance thresholds  (metres, bbox-area heuristic until DAv2)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "CRITICAL": 0.5,   # < 0.5 m  → stop immediately
    "WARNING":  1.5,   # 0.5–1.5 m → slow / swerve
    "CAUTION":  3.0,   # 1.5–3.0 m → heads up (never interrupts)
}                      # > 3.0 m  → CLEAR

WARNING_TEMPLATES = {
    "CRITICAL": "Emergency: {obj} {dist:.1f} metres ahead",
    "WARNING":  "Warning: {obj} {dist:.1f} metres ahead",
    "CAUTION":  "",    # CAUTION never triggers speech
}

# ─────────────────────────────────────────────────────────────────────────────
# Zone geometry  (normalised 0–1 coordinates)
#
#  ┌──────────────────────────────┐
#  │  PERIPHERAL │  CENTER  │ PERIPHERAL  │  ← y < BOTTOM_Y
#  │─────────────────────────────│
#  │            BOTTOM           │  ← y ≥ BOTTOM_Y  (ground level)
#  └──────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────

ZONE_BOTTOM_Y   = 0.72          # normalised y above which = BOTTOM
ZONE_CENTER_X   = (0.22, 0.78)  # normalised x band = CENTER

# ─────────────────────────────────────────────────────────────────────────────
# Interrupt throttle state  (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

INTERRUPT_COOLDOWN = 4.0   # seconds

@dataclass
class _ThrottleState:
    last_ts:    float = field(default=0.0)
    last_level: str   = field(default="CLEAR")

_throttle = _ThrottleState()


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_distance_heuristic(bbox_xyxy: list[float], img_w: int, img_h: int) -> float:
    """Rough monocular distance from bbox area ratio.
    Replace with DAv2 depth-map lookup once integrated.
    Empirical: area_ratio ~0.04 → ~2 m, ~0.25 → ~0.5 m.
    """
    x1, y1, x2, y2 = bbox_xyxy
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    if area_ratio <= 0:
        return 10.0
    dist = 0.4 / (area_ratio ** 0.5)
    return round(min(max(dist, 0.1), 20.0), 2)


def classify_level(dist_m: float) -> str:
    if dist_m < THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if dist_m < THRESHOLDS["WARNING"]:
        return "WARNING"
    if dist_m < THRESHOLDS["CAUTION"]:
        return "CAUTION"
    return "CLEAR"


def classify_zone(bbox_xyxy: list[float], img_w: int, img_h: int) -> str:
    """Return BOTTOM, CENTER, or PERIPHERAL based on bbox centre point."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    if cy >= ZONE_BOTTOM_Y:
        return "BOTTOM"
    if ZONE_CENTER_X[0] <= cx <= ZONE_CENTER_X[1]:
        return "CENTER"
    return "PERIPHERAL"


def build_warning_text(level: str, dist_m: float, obj_name: str) -> str:
    template = WARNING_TEMPLATES.get(level, "")
    if not template:
        return ""
    return template.format(dist=dist_m, obj=obj_name)


# ─────────────────────────────────────────────────────────────────────────────
# Interrupt gate  (stateful — mutates _throttle)
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL_RANK = {"CLEAR": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}


def _can_interrupt(zone: str, level: str) -> bool:
    """Return True if this detection should trigger an audible alert.

    Rules:
    • BOTTOM  → never
    • CAUTION → never (regardless of zone)
    • PERIPHERAL + WARNING → never
    • Otherwise: allow only if outside the 4 s cooldown,
      OR if the level escalated (WARNING → CRITICAL).
    """
    if level in ("CLEAR", "CAUTION"):
        return False
    if zone == "BOTTOM":
        return False
    if zone == "PERIPHERAL" and level != "CRITICAL":
        return False

    # CENTER + CRITICAL always fires — something very close straight ahead.
    if zone == "CENTER" and level == "CRITICAL":
        _throttle.last_ts    = time.time()
        _throttle.last_level = "CRITICAL"
        return True

    now = time.time()
    elapsed = now - _throttle.last_ts
    escalating = (
        _LEVEL_RANK.get(_throttle.last_level, 0) < _LEVEL_RANK[level]
        and elapsed < INTERRUPT_COOLDOWN
    )

    if elapsed < INTERRUPT_COOLDOWN and not escalating:
        return False

    # Commit — update throttle state.
    _throttle.last_ts    = now
    _throttle.last_level = level
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Model lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(f"[startup] loading model: {MODEL_PATH}")
    _model = YOLO(MODEL_PATH)
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    _model.predict(dummy, imgsz=IMG_SIZE, conf=CONF_THRES, device=DEVICE, verbose=False)
    print("[startup] model ready")
    yield
    print("[shutdown] releasing model")
    del _model


app = FastAPI(title="Spark Sight Warning API", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(img: Image.Image) -> dict:
    t0 = time.perf_counter()
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]

    results = _model.predict(
        img_np, imgsz=IMG_SIZE, conf=CONF_THRES, device=DEVICE, verbose=False,
    )[0]

    detections: list[dict] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in OBSTACLE_CLASSES:
            continue

        cls_name = results.names[cls_id]
        conf     = float(box.conf[0])
        xyxy     = box.xyxy[0].tolist()
        dist     = estimate_distance_heuristic(xyxy, w, h)
        level    = classify_level(dist)
        zone     = classify_zone(xyxy, w, h)
        category = "fast_moving" if cls_id in FAST_MOVING_CLASSES else "aerial"

        detections.append({
            "class":      cls_name,
            "category":   category,
            "confidence": round(conf, 3),
            "bbox_xyxy":  [round(v, 1) for v in xyxy],
            "distance_m": dist,
            "level":      level,
            "zone":       zone,
        })

    # ── Pick the single most threatening interruptable detection ──────────────
    # Priority order: level (CRITICAL > WARNING) then zone (CENTER > PERIPHERAL).
    candidates = [
        d for d in detections
        if d["level"] not in ("CLEAR", "CAUTION")
        and d["zone"] != "BOTTOM"
        and not (d["zone"] == "PERIPHERAL" and d["level"] != "CRITICAL")
    ]

    def _threat_key(d: dict) -> tuple[int, int]:
        return (_LEVEL_RANK[d["level"]], 1 if d["zone"] == "CENTER" else 0)

    best = max(candidates, key=_threat_key) if candidates else None

    # ── Interrupt decision ─────────────────────────────────────────────────────
    should_interrupt = False
    warning_text = ""
    if best:
        should_interrupt = _can_interrupt(best["zone"], best["level"])
        if should_interrupt:
            warning_text = build_warning_text(best["level"], best["distance_m"], best["class"])

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "objects":          detections,
        "warning_level":    best["level"]    if best else "CLEAR",
        "zone":             best["zone"]     if best else "NONE",
        "should_interrupt": should_interrupt,
        "warning_text":     warning_text,
        "closest_m":        best["distance_m"] if best else None,
        "latency_ms":       latency_ms,
    }


def decode_image_from_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        match = re.match(r"data:[^;]+;base64,(.+)", url, re.DOTALL)
        if not match:
            raise ValueError("malformed data URL")
        b64 = match.group(1)
    else:
        b64 = url
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible endpoint
# ─────────────────────────────────────────────────────────────────────────────

class ImageURL(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str
    image_url: Optional[ImageURL] = None
    text:      Optional[str]      = None

class Message(BaseModel):
    role:    str
    content: list[ContentPart] | str

class ChatRequest(BaseModel):
    model:      str  = "yolo11n-warning"
    messages:   list[Message]
    max_tokens: int  = 256
    stream:     bool = False


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    image: Optional[Image.Image] = None
    for msg in reversed(req.messages):
        if msg.role != "user" or isinstance(msg.content, str):
            continue
        for part in msg.content:
            if part.type == "image_url" and part.image_url:
                try:
                    image = decode_image_from_url(part.image_url.url)
                except Exception as exc:
                    raise HTTPException(400, f"image decode error: {exc}")
                break
        if image:
            break

    if image is None:
        raise HTTPException(400, "no image_url found in messages")

    result = run_detection(image)

    # content = warning text if interrupting, "CLEAR" otherwise.
    # WarningAgent reads this field to decide whether to fire a speech event.
    content_text = result["warning_text"] if result["should_interrupt"] else "CLEAR"

    return JSONResponse({
        "id":      f"chatcmpl-yolo-{int(time.time() * 1000)}",
        "object":  "chat.completion",
        "model":   req.model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content_text},
            "finish_reason": "stop",
        }],
        "usage":     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "detection": result,   # non-standard extension — full payload for debugging
    })


# ─────────────────────────────────────────────────────────────────────────────
# Raw detection endpoint  (debugging / direct frontend use)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/detect/raw")
async def detect_raw(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "cannot parse uploaded image")
    return JSONResponse(run_detection(img))


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Server:app", host="0.0.0.0", port=8081, workers=1)

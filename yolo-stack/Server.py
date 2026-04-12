"""
Spark Sight — YOLO11 Warning Service
Focused on outdoor obstacles a white cane CANNOT detect:
  • Fast-moving objects  (cyclists, cars, motorcycles, skateboards …)
  • Aerial/suspended     (traffic lights, poles, umbrellas, signs …)

Zone-based interrupt policy
  CENTER     (middle 56 % width, above ground band) : WARNING + CRITICAL interrupt
  PERIPHERAL (outer edges, above ground band)        : CRITICAL only
  BOTTOM     (lower 28 % of frame = ground level)   : never interrupts

Throttle: at most one interrupting alert per cooldown window.
Exception: CENTER + CRITICAL always fires; WARNING → CRITICAL escalation breaks through.

All tuneable parameters are exposed as CLI arguments (defaults = production values).
Docker usage:
  docker run ... yolo-server \\
      --model yolo11n.engine \\
      --thresh-critical 0.4 \\
      --fast-moving-classes "0,1,2,3,5,7,36"

POST /v1/chat/completions  — OpenAI VLM-style (image_url in messages)
POST /v1/detect/raw        — multipart file upload, full JSON payload
GET  /health
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import time
from collections import deque
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
# Runtime config  (populated from CLI args in main(), defaults = production)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Config:
    # Model
    model_path: str   = "yolo11n.engine"
    img_size:   int   = 640
    conf_thres: float = 0.35
    device:     int   = 0

    # Distance thresholds (metres)
    thresh_critical: float = 0.5   # < this  → CRITICAL (stop immediately)
    thresh_warning:  float = 1.5   # < this  → WARNING  (slow / swerve)
    thresh_caution:  float = 3.0   # < this  → CAUTION  (never interrupts)
                                   # ≥ this  → CLEAR

    # Zone geometry (normalised 0–1)
    zone_bottom_y:     float = 0.72   # cy ≥ this  → BOTTOM zone
    zone_center_x_min: float = 0.22   # CENTER band: [min, max]
    zone_center_x_max: float = 0.78

    # Interrupt throttle
    cooldown: float = 4.0   # seconds between interrupting alerts

    # Obstacle classes (COCO IDs)
    # fast_moving: enter path faster than cane feedback
    fast_moving_classes: frozenset[int] = field(
        default_factory=lambda: frozenset({
            0,   # person
            1,   # bicycle
            2,   # car
            3,   # motorcycle
            5,   # bus
            7,   # truck
            36,  # skateboard
        })
    )
    # aerial: suspended above cane reach (pole / sign / umbrella …)
    aerial_classes: frozenset[int] = field(
        default_factory=lambda: frozenset({
            9,   # traffic light  (on pole, head height)
            11,  # stop sign      (on pole)
            12,  # parking meter  (waist–chest height)
            25,  # umbrella       (sidewalk café / hand-carried, face height)
            56,  # chair          (outdoor café seating)
            58,  # potted plant   (elevated planter, stoop display)
        })
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8081

    @property
    def obstacle_classes(self) -> frozenset[int]:
        return self.fast_moving_classes | self.aerial_classes


# Module-level singleton — overwritten by main() before uvicorn starts.
cfg = _Config()


# ─────────────────────────────────────────────────────────────────────────────
# Warning message templates
# ─────────────────────────────────────────────────────────────────────────────

_WARNING_TEMPLATES = {
    "CRITICAL": "Emergency: {obj} {dist:.1f} metres ahead",
    "WARNING":  "Warning: {obj} {dist:.1f} metres ahead",
    "CAUTION":  "",   # CAUTION never triggers speech
}


# ─────────────────────────────────────────────────────────────────────────────
# Interrupt throttle state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ThrottleState:
    last_ts:    float = field(default=0.0)
    last_level: str   = field(default="CLEAR")

_throttle = _ThrottleState()

_LEVEL_RANK = {"CLEAR": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}

# ─────────────────────────────────────────────────────────────────────────────
# Frame summary cache  (last 10 frames, newest last)
# ─────────────────────────────────────────────────────────────────────────────

_frame_cache: deque[dict] = deque(maxlen=10)


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
    if dist_m < cfg.thresh_critical:
        return "CRITICAL"
    if dist_m < cfg.thresh_warning:
        return "WARNING"
    if dist_m < cfg.thresh_caution:
        return "CAUTION"
    return "CLEAR"


def classify_zone(bbox_xyxy: list[float], img_w: int, img_h: int) -> str:
    """Return BOTTOM, CENTER, or PERIPHERAL based on bbox centre point."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    if cy >= cfg.zone_bottom_y:
        return "BOTTOM"
    if cfg.zone_center_x_min <= cx <= cfg.zone_center_x_max:
        return "CENTER"
    return "PERIPHERAL"


def build_warning_text(level: str, dist_m: float, obj_name: str) -> str:
    template = _WARNING_TEMPLATES.get(level, "")
    if not template:
        return ""
    return template.format(dist=dist_m, obj=obj_name)


# ─────────────────────────────────────────────────────────────────────────────
# Interrupt gate  (stateful — mutates _throttle)
# ─────────────────────────────────────────────────────────────────────────────

def _can_interrupt(zone: str, level: str) -> bool:
    """Return True if this detection should trigger an audible alert.

    Interrupt matrix:
      BOTTOM               → never
      CAUTION (any zone)   → never
      PERIPHERAL + WARNING → never
      CENTER + CRITICAL    → always (bypasses cooldown)
      everything else      → only outside cooldown window, or on escalation
    """
    if level in ("CLEAR", "CAUTION"):
        return False
    if zone == "BOTTOM":
        return False
    if zone == "PERIPHERAL" and level != "CRITICAL":
        return False

    # CENTER + CRITICAL: immediate collision threat — always fire.
    if zone == "CENTER" and level == "CRITICAL":
        _throttle.last_ts    = time.time()
        _throttle.last_level = "CRITICAL"
        return True

    now     = time.time()
    elapsed = now - _throttle.last_ts
    escalating = (
        _LEVEL_RANK.get(_throttle.last_level, 0) < _LEVEL_RANK[level]
        and elapsed < cfg.cooldown
    )

    if elapsed < cfg.cooldown and not escalating:
        return False

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
    print(f"[startup] loading model: {cfg.model_path}")
    _model = YOLO(cfg.model_path)
    dummy = np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
    _model.predict(dummy, imgsz=cfg.img_size, conf=cfg.conf_thres,
                   device=cfg.device, verbose=False)
    print(f"[startup] model ready  "
          f"thresh=({cfg.thresh_critical}/{cfg.thresh_warning}/{cfg.thresh_caution})m  "
          f"cooldown={cfg.cooldown}s  "
          f"fast={sorted(cfg.fast_moving_classes)}  "
          f"aerial={sorted(cfg.aerial_classes)}")
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
        img_np, imgsz=cfg.img_size, conf=cfg.conf_thres,
        device=cfg.device, verbose=False,
    )[0]

    detections: list[dict] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        xyxy   = box.xyxy[0].tolist()
        dist   = estimate_distance_heuristic(xyxy, w, h)
        level  = classify_level(dist)
        zone   = classify_zone(xyxy, w, h)

        # Include if: (a) known obstacle class  OR
        #             (b) CENTER + CRITICAL — anything straight ahead at <thresh_critical
        is_known            = cls_id in cfg.obstacle_classes
        is_central_critical = (zone == "CENTER" and level == "CRITICAL")
        if not is_known and not is_central_critical:
            continue

        cls_name = results.names[cls_id]
        if cls_id in cfg.fast_moving_classes:
            category = "fast_moving"
        elif cls_id in cfg.aerial_classes:
            category = "aerial"
        else:
            category = "proximity"  # caught only by CENTER+CRITICAL proximity rule

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
    # Priority: level first (CRITICAL > WARNING), then zone (CENTER > PERIPHERAL).
    candidates = [
        d for d in detections
        if d["level"] not in ("CLEAR", "CAUTION")
        and d["zone"] != "BOTTOM"
        and not (d["zone"] == "PERIPHERAL" and d["level"] != "CRITICAL")
    ]

    def _threat_key(d: dict) -> tuple[int, int]:
        return (_LEVEL_RANK[d["level"]], 1 if d["zone"] == "CENTER" else 0)

    best = max(candidates, key=_threat_key) if candidates else None

    # ── Interrupt decision ────────────────────────────────────────────────────
    should_interrupt = False
    warning_text = ""
    if best:
        should_interrupt = _can_interrupt(best["zone"], best["level"])
        if should_interrupt:
            # "proximity" objects: skip the COCO class name, use generic phrasing.
            obj_label = (
                "obstacle directly ahead"
                if best["category"] == "proximity"
                else best["class"]
            )
            warning_text = build_warning_text(best["level"], best["distance_m"], obj_label)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Update frame cache ────────────────────────────────────────────────────
    _frame_cache.append({
        "ts":            round(time.time(), 3),
        "classes":       sorted({d["class"] for d in detections}),
        "warning_level": best["level"]      if best else "CLEAR",
        "zone":          best["zone"]       if best else "NONE",
        "closest_m":     best["distance_m"] if best else None,
        "interrupted":   should_interrupt,
    })

    return {
        "objects":          detections,
        "warning_level":    best["level"]      if best else "CLEAR",
        "zone":             best["zone"]       if best else "NONE",
        "should_interrupt": should_interrupt,
        "warning_text":     warning_text,
        "closest_m":        best["distance_m"] if best else None,
        "latency_ms":       latency_ms,
        "recent_frames":    list(_frame_cache),
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
    type:      str
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
        "detection": result,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Raw detection endpoint
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
    return {
        "status": "ok",
        "model":  cfg.model_path,
        "thresholds": {
            "critical_m": cfg.thresh_critical,
            "warning_m":  cfg.thresh_warning,
            "caution_m":  cfg.thresh_caution,
        },
        "cooldown_s":           cfg.cooldown,
        "zone_bottom_y":        cfg.zone_bottom_y,
        "zone_center_x":        [cfg.zone_center_x_min, cfg.zone_center_x_max],
        "fast_moving_classes":  sorted(cfg.fast_moving_classes),
        "aerial_classes":       sorted(cfg.aerial_classes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI + entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spark Sight YOLO Warning Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--model",    default=cfg.model_path, metavar="PATH",
                   help="YOLO model file (.pt or .engine)")
    p.add_argument("--img-size", default=cfg.img_size, type=int)
    p.add_argument("--conf",     default=cfg.conf_thres, type=float,
                   metavar="FLOAT", help="Detection confidence threshold")
    p.add_argument("--device",   default=cfg.device, type=int,
                   metavar="INT", help="CUDA device index")

    # ── Distance thresholds ────────────────────────────────────────────────
    g = p.add_argument_group("distance thresholds (metres)")
    g.add_argument("--thresh-critical", default=cfg.thresh_critical, type=float, metavar="M")
    g.add_argument("--thresh-warning",  default=cfg.thresh_warning,  type=float, metavar="M")
    g.add_argument("--thresh-caution",  default=cfg.thresh_caution,  type=float, metavar="M")

    # ── Zone geometry ──────────────────────────────────────────────────────
    g = p.add_argument_group("zone geometry (normalised 0–1)")
    g.add_argument("--zone-bottom-y",     default=cfg.zone_bottom_y,     type=float,
                   metavar="Y", help="cy ≥ this → BOTTOM zone")
    g.add_argument("--zone-center-x-min", default=cfg.zone_center_x_min, type=float, metavar="X")
    g.add_argument("--zone-center-x-max", default=cfg.zone_center_x_max, type=float, metavar="X")

    # ── Interrupt throttle ─────────────────────────────────────────────────
    p.add_argument("--cooldown", default=cfg.cooldown, type=float,
                   metavar="SEC", help="Interrupt cooldown in seconds")

    # ── Obstacle classes ───────────────────────────────────────────────────
    g = p.add_argument_group("obstacle classes (comma-separated COCO IDs)")
    g.add_argument("--fast-moving-classes",
                   default=",".join(str(i) for i in sorted(cfg.fast_moving_classes)),
                   metavar="IDS",
                   help="person=0,bicycle=1,car=2,motorcycle=3,bus=5,truck=7,skateboard=36")
    g.add_argument("--aerial-classes",
                   default=",".join(str(i) for i in sorted(cfg.aerial_classes)),
                   metavar="IDS",
                   help="traffic_light=9,stop_sign=11,parking_meter=12,"
                        "umbrella=25,chair=56,potted_plant=58")

    # ── Server ─────────────────────────────────────────────────────────────
    p.add_argument("--host", default=cfg.host)
    p.add_argument("--port", default=cfg.port, type=int)

    return p.parse_args()


def _apply_args(args: argparse.Namespace) -> None:
    """Overwrite the global cfg singleton with parsed CLI values."""
    cfg.model_path        = args.model
    cfg.img_size          = args.img_size
    cfg.conf_thres        = args.conf
    cfg.device            = args.device
    cfg.thresh_critical   = args.thresh_critical
    cfg.thresh_warning    = args.thresh_warning
    cfg.thresh_caution    = args.thresh_caution
    cfg.zone_bottom_y     = args.zone_bottom_y
    cfg.zone_center_x_min = args.zone_center_x_min
    cfg.zone_center_x_max = args.zone_center_x_max
    cfg.cooldown          = args.cooldown
    cfg.fast_moving_classes = frozenset(
        int(x.strip()) for x in args.fast_moving_classes.split(",") if x.strip()
    )
    cfg.aerial_classes = frozenset(
        int(x.strip()) for x in args.aerial_classes.split(",") if x.strip()
    )
    cfg.host = args.host
    cfg.port = args.port


if __name__ == "__main__":
    import uvicorn

    args = _parse_args()
    _apply_args(args)
    uvicorn.run("Server:app", host=cfg.host, port=cfg.port, workers=1)

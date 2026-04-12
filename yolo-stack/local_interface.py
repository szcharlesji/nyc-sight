"""
Spark Sight — Gradio UI
webcam → resize 320x180 → POST /v1/detect/raw → display detection + depth overlay
runs on user's machine, calls GB10 server
"""

import base64
import time
import json
from io import BytesIO

import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ── config ──────────────────────────────────────────
API_BASE   = "http://100.90.132.109:8081"   # change to GB10 IP if remote
SEND_W, SEND_H = 320, 180             # resize before sending
FPS_TARGET = 1
FRAME_INTERVAL = 1.0 / FPS_TARGET

LEVEL_COLORS = {
    "CRITICAL": "#FF2D2D",
    "WARNING":  "#FF9500",
    "CAUTION":  "#FFD60A",
    "CLEAR":    "#30D158",
}

LEVEL_BG = {
    "CRITICAL": (255, 45,  45,  200),
    "WARNING":  (255, 149, 0,   200),
    "CAUTION":  (255, 214, 10,  200),
    "CLEAR":    (48,  209, 88,  180),
}

# ── drawing ─────────────────────────────────────────

def draw_detections(frame_pil: Image.Image, detections: list, orig_w: int, orig_h: int) -> Image.Image:
    """Draw bboxes scaled back to display resolution."""
    img = frame_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    dw, dh = img.size
    sx = dw / SEND_W
    sy = dh / SEND_H

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

        level = det.get("level", "CLEAR")
        color = LEVEL_BG.get(level, (100, 100, 100, 180))
        hex_c = LEVEL_COLORS.get(level, "#888")

        # box
        draw.rectangle([x1, y1, x2, y2], outline=hex_c + "FF", width=2)
        # fill top strip for label
        draw.rectangle([x1, y1, x2, y1 + 20], fill=color)

        label = f"{det['class']}  {det['distance_m']}m"
        draw.text((x1 + 4, y1 + 3), label, fill="white")

    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def build_depth_bar(detections: list) -> Image.Image:
    """Simple horizontal depth bar showing closest object distance."""
    W, H = 480, 48
    img = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(img)

    max_dist = 5.0
    dists = [d["distance_m"] for d in detections] if detections else []
    closest = min(dists) if dists else max_dist

    ratio = max(0.0, min(1.0, 1.0 - closest / max_dist))
    bar_w = int(W * ratio)

    # gradient color based on distance
    if closest < 0.5:
        bar_color = (255, 45, 45)
    elif closest < 1.5:
        bar_color = (255, 149, 0)
    elif closest < 3.0:
        bar_color = (255, 214, 10)
    else:
        bar_color = (48, 209, 88)

    draw.rectangle([0, 8, bar_w, H - 8], fill=bar_color)
    # tick marks
    for m in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        tx = int(W * (1.0 - m / max_dist))
        draw.rectangle([tx, 4, tx + 1, H - 4], fill=(80, 80, 80))
        draw.text((tx - 8, H - 14), f"{m}m", fill=(120, 120, 120))

    label = f"closest: {closest:.1f}m" if closest < max_dist else "CLEAR"
    draw.text((8, 14), label, fill="white")
    return img


# ── API call ─────────────────────────────────────────

_last_call = 0.0

def process_frame(frame: np.ndarray):
    global _last_call

    if frame is None:
        return None, None, "──", "──", "no frame"

    now = time.time()
    elapsed = now - _last_call

    # throttle to 4fps — return None placeholders to keep output count correct
    if elapsed < FRAME_INTERVAL:
        return None, None, gr.update(), gr.update(), gr.update(), gr.update()

    _last_call = now
    t0 = time.perf_counter()

    # resize before sending
    pil_orig = Image.fromarray(frame)
    pil_small = pil_orig.resize((SEND_W, SEND_H), Image.BILINEAR)

    buf = BytesIO()
    pil_small.save(buf, format="JPEG", quality=80)
    buf.seek(0)

    try:
        resp = requests.post(
            f"{API_BASE}/v1/detect/raw",
            files={"file": ("frame.jpg", buf, "image/jpeg")},
            timeout=2.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return None, None, "──", "──", f"❌ cannot reach {API_BASE}"
    except Exception as e:
        return None, None, "──", "──", f"❌ {e}"

    round_trip = (time.perf_counter() - t0) * 1000

    detections = data.get("objects", [])
    level      = data.get("warning_level", "CLEAR")
    warn_text  = data.get("warning_text", "") or "CLEAR"
    infer_ms   = data.get("latency_ms", 0)

    # annotated frame (at display resolution)
    annotated = draw_detections(pil_orig, detections, pil_orig.width, pil_orig.height)
    depth_bar = build_depth_bar(detections)

    level_color = LEVEL_COLORS.get(level, "#888")
    level_md = f"<span style='color:{level_color};font-size:1.4em;font-weight:700'>{level}</span>"

    status = f"model {infer_ms:.0f}ms · round-trip {round_trip:.0f}ms · {len(detections)} obj"

    det_json = json.dumps(detections, ensure_ascii=False, indent=2) if detections else "( none )"

    return annotated, depth_bar, level_md, warn_text, status, det_json


# ── UI ───────────────────────────────────────────────

CSS = """
body { background: #0e0e10 !important; }
.gradio-container { background: #0e0e10 !important; font-family: 'SF Mono', monospace; }
#warning-box { 
    border-radius: 12px; padding: 16px 20px;
    background: #1a1a1e; border: 1px solid #2a2a2e;
    font-family: 'SF Mono', monospace;
}
#status-bar { 
    color: #666; font-size: 12px; font-family: 'SF Mono', monospace;
    padding: 4px 0;
}
.dark textarea, .dark input { 
    background: #1a1a1e !important; 
    border-color: #2a2a2e !important;
}
"""

def build_ui(api_base: str):
    global API_BASE
    API_BASE = api_base

with gr.Blocks(css=CSS, title="Spark Sight", theme=gr.themes.Default(
    primary_hue="orange",
    neutral_hue="slate",
)) as demo:

    gr.Markdown(
        "## ◈ Spark Sight — Warning Monitor\n"
        "webcam → resize 320×180 → GB10 YOLO11 → detection + distance"
    )

    with gr.Row():
        api_input = gr.Textbox(
            value=API_BASE,
            label="GB10 API endpoint",
            placeholder="http://192.168.x.x:8080",
            scale=3,
        )
        apply_btn = gr.Button("Apply", scale=1, variant="secondary")

    with gr.Row():
        # left: webcam + annotated
        with gr.Column(scale=3):
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Camera",
                height=360,
            )
            annotated_out = gr.Image(
                label="Detection overlay",
                height=360,
                interactive=False,
            )

        # right: status panel
        with gr.Column(scale=2):
            level_out = gr.HTML(
                value="<span style='color:#666;font-size:1.4em'>──</span>",
                label="Warning level",
                elem_id="warning-box",
            )
            warn_out = gr.Textbox(
                label="Warning text",
                interactive=False,
                lines=2,
            )
            depth_img = gr.Image(
                label="Distance bar",
                height=56,
                interactive=False,
                show_label=True,
            )
            status_out = gr.Markdown("──", elem_id="status-bar")
            det_json_out = gr.Code(
                label="Raw detections",
                language="json",
                lines=10,
                interactive=False,
            )

    # wire webcam stream → process every frame (throttled inside)
    webcam.stream(
        fn=process_frame,
        inputs=[webcam],
        outputs=[annotated_out, depth_img, level_out, warn_out, status_out, det_json_out],
        stream_every=1.0 / FPS_TARGET,
        time_limit=None,
    )

    def update_api(url):
        global API_BASE
        API_BASE = url.strip().rstrip("/")
        return f"endpoint set → {API_BASE}"

    apply_btn.click(update_api, inputs=[api_input], outputs=[status_out])

    gr.Markdown(
        "<small style='color:#444'>resize: 320×180 · target 4fps · "
        "detection via YOLO11n TRT · depth heuristic (pre-DAv2)</small>"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8080", help="GB10 server URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    API_BASE = args.api.rstrip("/")
    print(f"[ui] connecting to {API_BASE}")
    print(f"[ui] open http://localhost:{args.port}")

    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
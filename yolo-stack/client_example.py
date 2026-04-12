

import base64
import json
import sys
from pathlib import Path

import requests
from openai import OpenAI

API_BASE = "http://localhost:8080"

# ──────────────────────────────────────────────
# OpenAI SDK
# ──────────────────────────────────────────────

def call_openai_sdk(image_path: str):
    client = OpenAI(base_url=f"{API_BASE}/v1", api_key="not-used")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lstrip(".") or "jpeg"
    data_url = f"data:image/{ext};base64,{b64}"

    response = client.chat.completions.create(
        model="yolo11n-warning",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": "detect obstacles"},
            ],
        }],
        max_tokens=256,
    )

    print("=== OpenAI SDK ===")
    # Standard content field = warning text
    print("warning text:", response.choices[0].message.content)
    # Extended detection payload (non-standard field)
    raw = response.model_dump()
    if "detection" in raw:
        det = raw["detection"]
        print(f"level: {det['warning_level']}  closest: {det['closest_m']}m  latency: {det['latency_ms']}ms")
        for obj in det["objects"]:
            print(f"  · {obj['class']:12s} {obj['distance_m']}m  conf={obj['confidence']:.2f}")


# ──────────────────────────────────────────────
# raw requests + base64
# ──────────────────────────────────────────────

def call_raw_requests(image_path: str):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lstrip(".") or "jpeg"
    data_url = f"data:image/{ext};base64,{b64}"

    payload = {
        "model": "yolo11n-warning",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}}
            ],
        }],
    }
    r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=5)
    r.raise_for_status()
    data = r.json()

    print("\n=== requests + base64 ===")
    print("warning text:", data["choices"][0]["message"]["content"])
    det = data.get("detection", {})
    print(f"level: {det.get('warning_level')}  latency: {det.get('latency_ms')}ms")


# ──────────────────────────────────────────────
# multipart upload to /v1/detect/raw
# ──────────────────────────────────────────────

def call_detect_raw(image_path: str):
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/v1/detect/raw",
            files={"file": (Path(image_path).name, f, "image/jpeg")},
            timeout=5,
        )
    r.raise_for_status()
    data = r.json()

    print("\n=== /v1/detect/raw ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────
# Quick health check
# ──────────────────────────────────────────────

def check_health():
    r = requests.get(f"{API_BASE}/health", timeout=3)
    print("health:", r.json())


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    check_health()
    call_openai_sdk(image)
    call_raw_requests(image)
    call_detect_raw(image)

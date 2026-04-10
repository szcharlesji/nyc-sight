from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load .env from project root (no external dependency needed)
# ---------------------------------------------------------------------------
def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_env()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
MODEL = "gemma-4-31b-it"

SYSTEM_PROMPT = """\
You are a visual navigation assistant for a blind person walking in New York City.

RULES:
- Be concise. Max 2-3 sentences unless the user asks for detail.
- Always describe: obstacles, distances (approximate feet), directions (left/right/ahead/behind).
- Read any visible text: signs, street names, store names, transit info.
- If you see a crosswalk or intersection, mention it.
- Prioritize safety information first, then wayfinding, then ambient detail.
- Use clock positions for precise directions (e.g., "person approaching at 2 o'clock").

NEARBY CONDITIONS (from NYC Open Data):
{context}

Integrate the nearby conditions naturally into your description when relevant.
Do not recite them as a list."""


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def _build_payload(b64: str, query: str, system: str, stream: bool) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.3,
        "max_tokens": 3000,
        "stream": stream,
    }


def _strip_thinking(text: str) -> str:
    """Remove <thought>...</thought> blocks that Gemma includes in content."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()


def _visible(accumulated: str) -> str:
    """Return the user-visible portion of accumulated streamed text.

    While a <thought> block is still open (no closing tag yet), suppress
    everything inside it so users don't see partial reasoning.
    """
    # Strip complete thought blocks first
    text = re.sub(r"<thought>.*?</thought>", "", accumulated, flags=re.DOTALL)
    # If an opening tag is still open, suppress from that point onward
    text = re.sub(r"<thought>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _auth_headers() -> dict[str, str]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set — check your .env file")
    return {"Authorization": f"Bearer {GEMINI_API_KEY}"}


class VLMClient:
    def __init__(self, base_url: str = GEMINI_BASE_URL) -> None:
        self._base_url = base_url
        self._client = httpx.AsyncClient(timeout=60.0, headers=_auth_headers())

    async def describe(
        self,
        frame: np.ndarray,
        query: str,
        context: str = "No nearby condition data available.",
    ) -> str:
        """Call VLM and return the complete response string (used by orchestrator)."""
        b64 = _encode_frame(frame)
        system = SYSTEM_PROMPT.replace("{context}", context)
        payload = _build_payload(b64, query, system, stream=True)

        full_content = ""
        try:
            async with self._client.stream(
                "POST", self._base_url, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    full_content += chunk["choices"][0]["delta"].get("content") or ""
            return _strip_thinking(full_content) or "No description available."
        except Exception:
            logger.exception("VLM request failed")
            return "I'm sorry, I couldn't process that right now. Obstacle detection is still active."

    async def describe_stream(
        self,
        frame: np.ndarray,
        query: str,
        context: str = "No nearby condition data available.",
    ) -> AsyncGenerator[str, None]:
        """Stream VLM response; yields cumulative content strings for Gradio ChatInterface."""
        b64 = _encode_frame(frame)
        system = SYSTEM_PROMPT.replace("{context}", context)
        payload = _build_payload(b64, query, system, stream=True)

        accumulated = ""
        try:
            async with httpx.AsyncClient(
                timeout=60.0, headers=_auth_headers()
            ) as client:
                async with client.stream("POST", self._base_url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        token = chunk["choices"][0]["delta"].get("content") or ""
                        accumulated += token
                        visible = _visible(accumulated)
                        if visible:
                            yield visible
            if not _strip_thinking(accumulated):
                yield "No description available."
        except Exception:
            logger.exception("VLM stream failed")
            yield "I'm sorry, I couldn't process that right now."

    async def close(self) -> None:
        await self._client.aclose()

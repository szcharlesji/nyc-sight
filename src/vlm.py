from __future__ import annotations

import base64
import logging

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

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

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3.5-9b"


class VLMClient:
    def __init__(self, base_url: str = LM_STUDIO_URL) -> None:
        self._base_url = base_url
        self._client = httpx.AsyncClient(timeout=120.0)

    async def describe(
        self,
        frame: np.ndarray,
        query: str,
        context: str = "No nearby condition data available.",
    ) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode()

        system = SYSTEM_PROMPT.replace("{context}", context)
        payload = {
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
            "max_tokens": 1000,
        }

        try:
            resp = await self._client.post(self._base_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content
        except Exception:
            logger.exception("VLM request failed")
            return "I'm sorry, I couldn't process that right now. Obstacle detection is still active."

    async def close(self) -> None:
        await self._client.aclose()

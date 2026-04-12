"""Warning Agent — fast obstacle detection powered by YOLO11 TRT.

Processes camera frames via the yolo-stack/Server.py service and emits
WARNING or CLEAR signals at ~10 fps, running in parallel with AmbientAgent.

The service at YOLO_NIM_URL is optional: if it is unreachable the agent
silently returns CLEAR so the rest of the system is unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from spark_sight.agents.base import BaseAgent
from spark_sight.bridge.models import AmbientResponse, AmbientSignal
from spark_sight.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
_DEFAULT_NIM_BASE_URL = _settings.yolo.nim_url
_DEFAULT_MODEL = _settings.yolo.model


class WarningAgent(BaseAgent):
    """Sends camera frames to the YOLO11 service and returns WARNING or CLEAR.

    Runs in parallel with the Ambient Agent — it only emits WARNING signals
    (fed into the highest-priority speech queue slot) and is otherwise silent.

    Parameters
    ----------
    nim_base_url:
        Base URL of the YOLO warning service (e.g. ``http://localhost:8080/v1``).
    model:
        Model identifier sent in the API request body.
    """

    def __init__(
        self,
        *,
        nim_base_url: str = _DEFAULT_NIM_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._nim_base_url = nim_base_url
        self._model = model
        self._client: AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        return "WarningAgent"

    async def start(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self._nim_base_url,
            api_key="not-needed",
        )
        logger.info("%s started (url=%s)", self.name, self._nim_base_url)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("%s stopped", self.name)

    async def process(self, input_data: dict[str, Any]) -> AmbientResponse:
        """Detect obstacles in a single camera frame.

        Parameters
        ----------
        input_data:
            ``{"frame_base64": str}`` — a base64-encoded JPEG frame.

        Returns
        -------
        AmbientResponse with signal=WARNING (+ message) or signal=CLEAR.
        """
        frame_b64: str | None = input_data.get("frame_base64")
        if self._client is None or not frame_b64:
            return AmbientResponse(signal=AmbientSignal.CLEAR)

        data_url = f"data:image/jpeg;base64,{frame_b64}"
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=64,
                temperature=0.0,
            )
            content = (response.choices[0].message.content or "").strip()
            if content and content != "CLEAR":
                logger.info("[YOLO] WARNING: %s", content[:100])
                return AmbientResponse(signal=AmbientSignal.WARNING, message=content)
            return AmbientResponse(signal=AmbientSignal.CLEAR)
        except Exception:
            # Service offline or unreachable — degrade silently, don't disrupt system.
            logger.debug("[YOLO] service unreachable — skipping frame")
            return AmbientResponse(signal=AmbientSignal.CLEAR)

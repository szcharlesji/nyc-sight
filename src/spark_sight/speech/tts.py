"""Magpie TTS — text-to-speech via NVIDIA NIM.

Uses the OpenAI-compatible ``/v1/audio/speech`` endpoint exposed by the
Magpie TTS NIM container.  The ``tts_loop`` coroutine drains the
Orchestrator's speech queue, synthesizes WAV audio, and pushes it to the
server's TTS queue for delivery to the iPhone.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from spark_sight.config import get_settings

if TYPE_CHECKING:
    from spark_sight.bridge.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_settings = get_settings()


class TTSClient:
    """Async client for the Magpie TTS NIM endpoint.

    Parameters
    ----------
    nim_base_url:
        Base URL of the Magpie TTS NIM (e.g. ``http://host:9001/v1``).
    model:
        Model identifier passed to the NIM API.
    voice:
        Voice identifier for speech synthesis.
    """

    def __init__(
        self,
        *,
        nim_base_url: str = _settings.magpie.nim_url,
        model: str = _settings.magpie.model,
        voice: str = _settings.magpie.voice,
    ) -> None:
        self._nim_base_url = nim_base_url
        self._model = model
        self._voice = voice
        self._client: AsyncOpenAI | None = None
        self._consecutive_failures = 0
        self._failed = False

    async def start(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self._nim_base_url,
            api_key="not-needed",
        )
        logger.info("TTSClient started (model=%s, voice=%s)", self._model, self._voice)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("TTSClient stopped")

    @property
    def available(self) -> bool:
        """Whether the TTS client is connected and has not failed."""
        return self._client is not None and not self._failed

    async def synthesize(self, text: str) -> bytes | None:
        """Convert *text* to WAV audio bytes via Magpie NIM.

        Returns ``None`` if the client is not started or synthesis fails.
        After 3 consecutive failures, marks itself as unavailable to
        avoid spamming connection errors when the NIM isn't running.
        """
        if self._client is None or self._failed or not text.strip():
            return None

        try:
            response = await self._client.audio.speech.create(
                model=self._model,
                input=text,
                voice=self._voice,
                response_format="wav",
            )
            wav_bytes = response.read()
            self._consecutive_failures = 0
            logger.debug("TTS synthesized %d bytes for: %s", len(wav_bytes), text[:60])
            return wav_bytes
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._failed = True
                logger.error(
                    "TTS unavailable — Magpie NIM at %s not reachable after %d failures. "
                    "TTS synthesis disabled. Speech will be text-only.",
                    self._nim_base_url, self._consecutive_failures,
                )
            else:
                logger.warning("TTS synthesis failed (%d/3): %s", self._consecutive_failures, text[:60])
            return None


async def tts_loop(
    orchestrator: Orchestrator,
    tts_client: TTSClient,
    tts_queue: asyncio.Queue[bytes],
) -> None:
    """Continuously drain the speech queue, synthesize audio, and push to the
    server's TTS output queue.

    Runs until cancelled.

    Parameters
    ----------
    orchestrator:
        Source of speech items via ``next_speech()``.
    tts_client:
        Magpie TTS client for synthesis.
    tts_queue:
        Server-side queue that feeds the WebSocket send loop.
    """
    logger.info("TTS loop started")
    while True:
        priority, text = await orchestrator.next_speech()

        if not tts_client.available:
            # Magpie not running — just drain the queue silently.
            logger.debug("TTS [%s] (skip, unavailable): %s", priority, text[:60])
            continue

        logger.info("TTS [%s]: %s", priority, text[:80])
        wav = await tts_client.synthesize(text)
        if wav:
            await tts_queue.put(wav)
        else:
            logger.debug("TTS produced no audio for: %s", text[:60])

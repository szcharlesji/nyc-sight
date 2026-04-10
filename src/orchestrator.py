from __future__ import annotations

import asyncio
import logging
import time

from src.frame_buffer import FrameBuffer
from src.nyc_data import NYCData
from src.tts import PiperTTS
from src.vlm import VLMClient

logger = logging.getLogger(__name__)

AMBIENT_INTERVAL = 15  # seconds between ambient descriptions


class Orchestrator:
    def __init__(
        self,
        buffer: FrameBuffer,
        alert_queue: asyncio.Queue[str],
        voice_queue: asyncio.Queue[str],
        vlm_trigger_queue: asyncio.Queue[str],
        vlm: VLMClient,
        tts: PiperTTS,
        nyc: NYCData,
        ui_callback=None,
    ) -> None:
        self._buffer = buffer
        self._alert_queue = alert_queue
        self._voice_queue = voice_queue
        self._vlm_trigger_queue = vlm_trigger_queue
        self._vlm = vlm
        self._tts = tts
        self._nyc = nyc
        self._is_vlm_busy = False
        self._ui_callback = ui_callback
        self._latest_response = ""
        self._latest_alert = ""
        self._latest_audio: str | None = None

    async def run(self) -> None:
        logger.info("Orchestrator started")
        await asyncio.gather(
            self._handle_alerts(),
            self._handle_voice(),
        )

    async def _handle_alerts(self) -> None:
        while True:
            alert = await self._alert_queue.get()
            self._latest_alert = alert
            logger.info("Alert: %s", alert)
            wav = self._tts.speak(alert)
            self._latest_audio = wav
            self._notify_ui(alert=alert, audio=wav)

    async def _handle_voice(self) -> None:
        while True:
            text = await self._voice_queue.get()
            if self._is_vlm_busy:
                logger.info("VLM busy, dropping voice request: %s", text)
                continue
            await self._call_vlm(text)

    async def _handle_vlm_triggers(self) -> None:
        while True:
            reason = await self._vlm_trigger_queue.get()
            if self._is_vlm_busy:
                continue
            await self._call_vlm(reason)

    async def _ambient_loop(self) -> None:
        await asyncio.sleep(AMBIENT_INTERVAL)
        while True:
            if not self._is_vlm_busy:
                await self._call_vlm("Briefly describe what you see around me.")
            await asyncio.sleep(AMBIENT_INTERVAL)

    async def _call_vlm(self, query: str) -> None:
        frame = self._buffer.latest()
        if frame is None:
            return

        self._is_vlm_busy = True
        self._notify_ui(status="VLM processing…")
        try:
            context = self._nyc.get_context()
            response = await self._vlm.describe(frame.image, query, context)
            self._latest_response = response
            logger.info("VLM response: %s", response)
            wav = self._tts.speak(response)
            self._latest_audio = wav
            self._notify_ui(response=response, audio=wav)
        except Exception:
            logger.exception("VLM call failed")
        finally:
            self._is_vlm_busy = False
            self._notify_ui(status="Ready")

    def _notify_ui(self, **kwargs) -> None:
        if self._ui_callback:
            self._ui_callback(**kwargs)

    async def handle_text_input(self, text: str) -> None:
        await self._voice_queue.put(text)

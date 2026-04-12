"""Parakeet-EOU ASR — streaming speech-to-text via WebSocket.

Connects to the Parakeet-EOU-120M server over WebSocket and streams raw
audio chunks.  The server performs end-of-utterance (EOU) detection and
returns finalized transcripts as JSON events.

The ``asr_loop`` coroutine drains the server's ``audio_queue`` (raw PCM
chunks from the web client mic), converts from i16le to f32le, and
forwards them to Parakeet-EOU.  When an ``eou`` event arrives, the
transcript is routed to the Orchestrator for planning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import TYPE_CHECKING

import websockets

from spark_sight.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

logger = logging.getLogger(__name__)

_settings = get_settings()

# ---------------------------------------------------------------------------
# Audio constants — must match the web client's mic capture settings.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000   # 16 kHz
SAMPLE_WIDTH = 2       # 16-bit (2 bytes per sample) — incoming i16le
CHANNELS = 1           # mono


def _i16le_to_f32le(pcm_i16: bytes) -> bytes:
    """Convert 16-bit signed PCM to 32-bit float PCM.

    Parakeet-EOU accepts binary WebSocket frames as f32le at 16 kHz mono.
    The web client sends i16le, so we normalize to [-1.0, 1.0] range.
    """
    n_samples = len(pcm_i16) // SAMPLE_WIDTH
    samples_i16 = struct.unpack(f"<{n_samples}h", pcm_i16[: n_samples * SAMPLE_WIDTH])
    samples_f32 = [s / 32768.0 for s in samples_i16]
    return struct.pack(f"<{n_samples}f", *samples_f32)


class ASRClient:
    """Streaming ASR client for the Parakeet-EOU WebSocket endpoint.

    Maintains a persistent WebSocket connection.  Audio is pushed via
    ``send_audio()``, and finalized transcripts arrive through the
    ``on_transcript`` callback provided at ``start()``.
    """

    def __init__(
        self,
        *,
        ws_url: str = _settings.parakeet.ws_url,
    ) -> None:
        self._ws_url = ws_url
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._listener_task: asyncio.Task | None = None
        self._on_transcript: Callable[[str], Awaitable[None]] | None = None
        self._connected = False

    async def start(
        self,
        on_transcript: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Open the WebSocket to Parakeet-EOU and start listening for events."""
        self._on_transcript = on_transcript
        await self._connect()

    async def _connect(self) -> None:
        """Establish WebSocket connection with retry."""
        try:
            self._ws = await websockets.connect(self._ws_url)
            self._connected = True
            self._listener_task = asyncio.create_task(
                self._listen(), name="asr-listener"
            )
            logger.info("ASRClient connected to %s", self._ws_url)
        except Exception:
            self._connected = False
            logger.warning(
                "ASRClient could not connect to %s — ASR disabled", self._ws_url
            )

    async def stop(self) -> None:
        """Send end-of-stream and close the WebSocket."""
        if self._ws and self._connected:
            try:
                await self._ws.send(json.dumps({"type": "eos"}))
                await self._ws.close()
            except Exception:
                pass
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        self._ws = None
        self._connected = False
        logger.info("ASRClient stopped")

    @property
    def available(self) -> bool:
        return self._connected and self._ws is not None

    async def send_audio(self, pcm_i16: bytes) -> None:
        """Send a chunk of i16le PCM audio, converted to f32le for Parakeet-EOU."""
        if not self.available:
            return
        try:
            f32_data = _i16le_to_f32le(pcm_i16)
            await self._ws.send(f32_data)
        except websockets.ConnectionClosed:
            logger.warning("ASR WebSocket closed during send, attempting reconnect")
            self._connected = False
            await self._reconnect()
        except Exception:
            logger.warning("ASR send failed")

    async def _reconnect(self) -> None:
        """Attempt to reconnect after a disconnection."""
        if self._listener_task:
            self._listener_task.cancel()
        await asyncio.sleep(2)
        await self._connect()

    async def _listen(self) -> None:
        """Listen for transcript events from Parakeet-EOU."""
        try:
            async for message in self._ws:
                try:
                    event = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue

                event_type = event.get("type")

                if event_type == "eou":
                    text = event.get("text", "").strip()
                    if text and self._on_transcript:
                        logger.info("ASR transcript (eou): %s", text[:120])
                        await self._on_transcript(text)

                elif event_type == "partial":
                    text = event.get("text", "").strip()
                    if text:
                        logger.debug("ASR partial: %s", text[:80])

                elif event_type == "error":
                    logger.warning(
                        "ASR error from server: %s", event.get("message", "unknown")
                    )
        except websockets.ConnectionClosed:
            logger.warning("ASR WebSocket closed, will reconnect on next audio")
            self._connected = False
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ASR listener error")
            self._connected = False


async def asr_loop(
    audio_queue: asyncio.Queue[bytes],
    asr_client: ASRClient,
    on_transcript: Callable[[str], Awaitable[None]],
) -> None:
    """Continuously forward PCM from the audio queue to Parakeet-EOU.

    Runs until cancelled.  Parakeet-EOU handles voice activity detection
    and end-of-utterance detection internally, so this loop simply
    converts audio format and forwards chunks.

    Parameters
    ----------
    audio_queue:
        Source of raw 16-bit PCM audio chunks from the web client mic.
    asr_client:
        Streaming Parakeet-EOU ASR client.
    on_transcript:
        Passed to ``asr_client.start()`` — invoked when finalized
        transcripts arrive.  Typically ``orchestrator.handle_transcript``.
    """
    logger.info("ASR loop started")

    # Ensure the client has the transcript callback wired.
    if not asr_client._on_transcript:
        asr_client._on_transcript = on_transcript

    while True:
        pcm = await audio_queue.get()
        await asr_client.send_audio(pcm)

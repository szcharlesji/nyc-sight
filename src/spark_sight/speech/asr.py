"""Parakeet ASR — speech-to-text via NVIDIA NIM.

Uses the OpenAI Whisper-compatible ``/v1/audio/transcriptions`` endpoint
exposed by the Parakeet 1.1B RNNT NIM container.

The ``asr_loop`` coroutine drains the server's ``audio_queue`` (raw PCM
chunks from the iPhone), performs simple energy-based voice activity
detection, packages speech segments as WAV, sends them to Parakeet, and
routes the transcript to the Orchestrator for planning.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import struct
import wave
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from spark_sight.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

logger = logging.getLogger(__name__)

_settings = get_settings()

# ---------------------------------------------------------------------------
# Audio constants — must match the iPhone client's mic capture settings.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000   # 16 kHz
SAMPLE_WIDTH = 2       # 16-bit (2 bytes per sample)
CHANNELS = 1           # mono

# ---------------------------------------------------------------------------
# VAD parameters (simple energy-based voice activity detection).
# ---------------------------------------------------------------------------
# RMS energy threshold to consider a chunk as "speech".  Tuned for
# iPhone mic → WebSocket PCM.  Increase if you get false positives.
_ENERGY_THRESHOLD = 300

# Number of consecutive "silent" chunks before we consider speech ended.
# Each chunk is 4096 samples @ 16kHz = 256 ms, so 6 chunks ≈ 1.5 s.
_SILENCE_CHUNKS_TO_END = 6

# Minimum number of "speech" chunks required to trigger transcription.
# Filters out short noise bursts.  3 chunks ≈ 768 ms.
_MIN_SPEECH_CHUNKS = 3

# Maximum speech duration in chunks before forced transcription.
# 120 chunks × 256 ms = ~30 seconds.
_MAX_SPEECH_CHUNKS = 120


def _rms_energy(pcm: bytes) -> float:
    """Compute RMS energy of 16-bit little-endian PCM."""
    n_samples = len(pcm) // SAMPLE_WIDTH
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", pcm[:n_samples * SAMPLE_WIDTH])
    return math.sqrt(sum(s * s for s in samples) / n_samples)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM bytes in a WAV header (16 kHz, 16-bit, mono)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


class ASRClient:
    """Async client for the Parakeet ASR NIM endpoint.

    Parameters
    ----------
    nim_base_url:
        Base URL of the Parakeet NIM (e.g. ``http://host:9000/v1``).
    model:
        Model identifier passed to the NIM API.
    """

    def __init__(
        self,
        *,
        nim_base_url: str = _settings.parakeet.nim_url,
        model: str = _settings.parakeet.model,
    ) -> None:
        self._nim_base_url = nim_base_url
        self._model = model
        self._client: AsyncOpenAI | None = None

    async def start(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self._nim_base_url,
            api_key="not-needed",
        )
        logger.info("ASRClient started (model=%s)", self._model)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("ASRClient stopped")

    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes into text.

        Returns an empty string if the client is not started, the audio
        is empty, or transcription fails.
        """
        if self._client is None or not wav_bytes:
            return ""

        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "speech.wav"

            transcript = await self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
                language="en",
            )
            text = transcript.text.strip()
            if text:
                logger.info("ASR transcript: %s", text[:120])
            return text
        except Exception:
            logger.exception("ASR transcription failed")
            return ""


async def asr_loop(
    audio_queue: asyncio.Queue[bytes],
    asr_client: ASRClient,
    on_transcript: Callable[[str], Awaitable[None]],
) -> None:
    """Continuously buffer PCM from the audio queue, detect speech, and
    transcribe via Parakeet.

    Runs until cancelled.

    Parameters
    ----------
    audio_queue:
        Source of raw 16-bit PCM audio chunks from the iPhone mic.
    asr_client:
        Parakeet ASR client for transcription.
    on_transcript:
        Async callback invoked with each non-empty transcript string.
        Typically ``orchestrator.handle_transcript``.
    """
    logger.info("ASR loop started")

    speech_chunks: list[bytes] = []
    silence_count = 0
    is_speaking = False

    while True:
        pcm = await audio_queue.get()

        energy = _rms_energy(pcm)

        if energy >= _ENERGY_THRESHOLD:
            # Speech detected.
            if not is_speaking:
                logger.debug("VAD: speech start (energy=%.0f)", energy)
            is_speaking = True
            silence_count = 0
            speech_chunks.append(pcm)

            # Force-send if we've been recording too long.
            if len(speech_chunks) >= _MAX_SPEECH_CHUNKS:
                logger.debug("VAD: max duration reached, forcing transcription")
                await _flush_and_transcribe(
                    speech_chunks, asr_client, on_transcript
                )
                speech_chunks = []
                is_speaking = False
        elif is_speaking:
            # Silence while we were recording speech.
            silence_count += 1
            speech_chunks.append(pcm)  # include trailing silence for context

            if silence_count >= _SILENCE_CHUNKS_TO_END:
                logger.debug("VAD: speech end (silence=%d chunks)", silence_count)
                await _flush_and_transcribe(
                    speech_chunks, asr_client, on_transcript
                )
                speech_chunks = []
                silence_count = 0
                is_speaking = False
        # else: silence and not speaking — discard.


async def _flush_and_transcribe(
    chunks: list[bytes],
    asr_client: ASRClient,
    on_transcript: Callable[[str], Awaitable[None]],
) -> None:
    """Package buffered PCM chunks, transcribe, and fire callback."""
    # Filter out very short utterances (noise).
    speech_count = sum(
        1 for c in chunks if _rms_energy(c) >= _ENERGY_THRESHOLD
    )
    if speech_count < _MIN_SPEECH_CHUNKS:
        logger.debug("VAD: too short (%d speech chunks), discarding", speech_count)
        return

    # Concatenate PCM and encode as WAV.
    pcm = b"".join(chunks)
    wav = pcm_to_wav(pcm)

    # Transcribe.
    text = await asr_client.transcribe(wav)
    if text:
        await on_transcript(text)

from __future__ import annotations

import asyncio
import logging
import queue
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.03  # 30ms per VAD frame
SILENCE_THRESHOLD = 0.6  # seconds of silence to end utterance
VAD_THRESHOLD = 0.5


class ASREngine:
    def __init__(
        self,
        voice_queue: asyncio.Queue[str],
        no_mic: bool = False,
        whisper_model: str = "small",
    ) -> None:
        self._voice_queue = voice_queue
        self._no_mic = no_mic
        self._whisper_model_name = whisper_model
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        if self._no_mic:
            logger.info("ASR disabled (--no-mic)")
            return

        self._loop = asyncio.get_running_loop()

        logger.info("Loading Whisper model '%s' …", self._whisper_model_name)
        from faster_whisper import WhisperModel
        whisper = WhisperModel(
            self._whisper_model_name,
            device="auto",
            compute_type="int8",
        )

        logger.info("Loading Silero VAD …")
        import torch
        vad_model, vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        (get_speech_ts, *_) = vad_utils

        audio_q: queue.Queue[np.ndarray] = queue.Queue()
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("Sounddevice status: %s", status)
            audio_q.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=_audio_callback,
        )
        stream.start()
        logger.info("ASR listening on microphone")

        await self._loop.run_in_executor(
            None,
            self._vad_loop,
            audio_q,
            vad_model,
            whisper,
            chunk_samples,
        )

    def _vad_loop(self, audio_q, vad_model, whisper, chunk_samples):
        import torch

        speech_buf: list[np.ndarray] = []
        is_speaking = False
        silence_chunks = 0
        silence_limit = int(SILENCE_THRESHOLD / CHUNK_DURATION)

        while True:
            try:
                chunk = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            tensor = torch.from_numpy(chunk)
            try:
                speech_prob = vad_model(tensor, SAMPLE_RATE).item()
            except Exception:
                continue

            if speech_prob > VAD_THRESHOLD:
                is_speaking = True
                silence_chunks = 0
                speech_buf.append(chunk)
            elif is_speaking:
                silence_chunks += 1
                speech_buf.append(chunk)
                if silence_chunks >= silence_limit and len(speech_buf) > 10:
                    audio = np.concatenate(speech_buf)
                    speech_buf.clear()
                    is_speaking = False
                    silence_chunks = 0
                    self._transcribe(whisper, audio)

    def _transcribe(self, whisper, audio: np.ndarray) -> None:
        try:
            segments, _ = whisper.transcribe(audio, beam_size=1, language="en")
            text = " ".join(s.text.strip() for s in segments).strip()
            if text and len(text) > 2:
                logger.info("ASR transcribed: %s", text)
                if self._loop:
                    self._loop.call_soon_threadsafe(
                        self._voice_queue.put_nowait, text
                    )
        except Exception:
            logger.exception("Transcription error")

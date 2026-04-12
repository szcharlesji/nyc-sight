"""Streaming TTS client + LLM-interleave demo.

Hits a running Kokoro TTS server over HTTP. Two modes:

    # Stream a single sentence to the speakers:
    python client.py --server http://localhost:8880 --text "Hello there."

    # Run the LLM-interleave demo (fake LLM token stream + TTS + playback):
    python client.py --server http://localhost:8880 --demo-llm-interleave

PyAudio is only imported inside the playback path so environments without
portaudio can still run ``--no-play`` for smoke tests.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import random
import re
import sys
import time
from typing import AsyncIterator, Optional

import httpx


log = logging.getLogger("client")

SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Playback helpers
# ---------------------------------------------------------------------------


class _Speaker:
    """Thin wrapper over PyAudio for int16 mono output."""

    def __init__(self, sample_rate: int) -> None:
        import pyaudio  # imported lazily

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
        )

    def write(self, data: bytes) -> None:
        self._stream.write(data)

    def close(self) -> None:
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        self._pa.terminate()


# ---------------------------------------------------------------------------
# HTTP streaming
# ---------------------------------------------------------------------------


async def stream_speech(
    server: str,
    text: str,
    voice: str,
    response_format: str = "pcm",
    speed: float = 1.0,
) -> AsyncIterator[tuple[float, bytes]]:
    """POSTs /v1/audio/speech and yields (elapsed_s_since_request, bytes)."""
    url = f"{server.rstrip('/')}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        t0 = time.perf_counter()
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"server returned {resp.status_code}: {body!r}")
            async for chunk in resp.aiter_bytes(chunk_size=4096):
                if not chunk:
                    continue
                yield time.perf_counter() - t0, chunk


async def stream_to_speaker(
    server: str,
    text: str,
    voice: str,
    speed: float = 1.0,
    play: bool = True,
) -> dict:
    spk: Optional[_Speaker] = None
    if play:
        spk = _Speaker(SAMPLE_RATE)
    ttfa: Optional[float] = None
    total = 0
    try:
        async for elapsed, chunk in stream_speech(server, text, voice, "pcm", speed):
            if ttfa is None:
                ttfa = elapsed
                log.info("ttfa: %.3f s", ttfa)
            total += len(chunk)
            if spk is not None:
                spk.write(chunk)
    finally:
        if spk is not None:
            spk.close()
    samples = total // 2
    return {
        "bytes": total,
        "samples": samples,
        "audio_s": samples / SAMPLE_RATE,
        "ttfa_s": ttfa,
    }


# ---------------------------------------------------------------------------
# LLM interleave demo
# ---------------------------------------------------------------------------


_SENTENCE_END = re.compile(r"[.!?]\s")


async def _fake_llm(prompt: str, tokens_per_sec: float = 40.0) -> AsyncIterator[str]:
    """Simulate an LLM token stream. Emits ~one word per tick with jitter."""
    words = prompt.split()
    period = 1.0 / tokens_per_sec
    for i, w in enumerate(words):
        await asyncio.sleep(period * (0.8 + 0.4 * random.random()))
        yield (w if i == 0 else " " + w)


async def llm_interleave_demo(server: str, voice: str, prompt: str, play: bool) -> None:
    t_start = time.perf_counter()
    log.info("[t=+0.000] llm_start")
    sentences: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def producer() -> None:
        buf = ""
        async for tok in _fake_llm(prompt):
            buf += tok
            # Emit as soon as we see a sentence boundary.
            while True:
                m = _SENTENCE_END.search(buf)
                if not m:
                    break
                end = m.end()
                sent = buf[:end].strip()
                buf = buf[end:]
                if sent:
                    await sentences.put(sent)
                    log.info(
                        "[t=%.3f] sent_complete :: %r",
                        time.perf_counter() - t_start,
                        sent,
                    )
        if buf.strip():
            await sentences.put(buf.strip())
            log.info(
                "[t=%.3f] sent_complete :: %r",
                time.perf_counter() - t_start,
                buf.strip(),
            )
        await sentences.put(None)

    async def consumer() -> None:
        first_audio_logged = False
        while True:
            sent = await sentences.get()
            if sent is None:
                return
            log.info("[t=%.3f] tts_request :: %r", time.perf_counter() - t_start, sent)
            metrics = await stream_to_speaker(server, sent, voice, play=play)
            if not first_audio_logged and metrics.get("ttfa_s") is not None:
                log.info(
                    "[t=%.3f] first_audio (ttfa=%.3fs)",
                    time.perf_counter() - t_start,
                    metrics["ttfa_s"],
                )
                first_audio_logged = True
            log.info("[t=%.3f] playback_done", time.perf_counter() - t_start)

    await asyncio.gather(producer(), consumer())
    log.info("[t=%.3f] llm_interleave_demo done", time.perf_counter() - t_start)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_DEMO_PROMPT = (
    "Welcome to the demo. The agent will speak each sentence as soon as the "
    "language model finishes it. This interleaves generation and synthesis."
)


def main() -> None:
    parser = argparse.ArgumentParser("kokoro-tts-client")
    parser.add_argument("--server", default="http://localhost:8880")
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument("--text", default="Hello from the Kokoro TTS server.")
    parser.add_argument("--format", default="pcm", choices=("pcm", "wav", "mp3", "opus", "flac"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--demo-llm-interleave", action="store_true")
    parser.add_argument("--no-play", action="store_true", help="Skip PyAudio playback.")
    parser.add_argument("--out", help="Write received audio to this file (PCM only, no container).")
    ns = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")

    if ns.demo_llm_interleave:
        asyncio.run(llm_interleave_demo(ns.server, ns.voice, _DEFAULT_DEMO_PROMPT, play=not ns.no_play))
        return

    if ns.out or ns.format != "pcm" or ns.no_play:
        async def run() -> None:
            sink = io.BytesIO()
            ttfa = None
            async for elapsed, chunk in stream_speech(ns.server, ns.text, ns.voice, ns.format, ns.speed):
                if ttfa is None:
                    ttfa = elapsed
                sink.write(chunk)
            log.info("ttfa=%.3fs bytes=%d", ttfa or 0.0, len(sink.getvalue()))
            if ns.out:
                open(ns.out, "wb").write(sink.getvalue())
                log.info("wrote %s", ns.out)

        asyncio.run(run())
        return

    metrics = asyncio.run(stream_to_speaker(ns.server, ns.text, ns.voice, ns.speed, play=True))
    log.info("done: %s", metrics)


if __name__ == "__main__":
    main()

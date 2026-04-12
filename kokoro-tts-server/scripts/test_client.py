#!/usr/bin/env python
"""Smoke-test client for the Kokoro TTS server.

Posts one request, prints time-to-first-byte, and writes received audio to
disk. Useful for CI and manual health checks.

    python scripts/test_client.py --server http://localhost:8880 \
        --voice af_bella --text "Hello there." --format wav --out out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import httpx


async def run(
    server: str,
    voice: str,
    text: str,
    fmt: str,
    out: Path,
    speed: float,
) -> None:
    url = f"{server.rstrip('/')}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": voice,
        "response_format": fmt,
        "speed": speed,
    }
    t0 = time.perf_counter()
    ttfb = None
    total = 0
    with out.open("wb") as fp:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if chunk:
                        if ttfb is None:
                            ttfb = time.perf_counter() - t0
                        fp.write(chunk)
                        total += len(chunk)
    elapsed = time.perf_counter() - t0
    print(f"ttfb   : {ttfb*1000:.1f} ms" if ttfb else "ttfb   : n/a")
    print(f"total  : {elapsed*1000:.1f} ms")
    print(f"bytes  : {total}")
    print(f"written: {out}")


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--server", default="http://localhost:8880")
    p.add_argument("--voice", default="af_bella")
    p.add_argument("--text", default="Hello there. This is a Kokoro smoke test.")
    p.add_argument("--format", default="wav", choices=("pcm", "wav", "mp3", "opus", "flac"))
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("kokoro_smoke.out"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.server, args.voice, args.text, args.format, args.out, args.speed))


if __name__ == "__main__":
    main()

"""FastAPI server exposing an OpenAI-compatible /v1/audio/speech endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from audio_encoder import make_encoder
from backend import KokoroBackend
from config import (
    DEFAULT_FORMAT,
    DEFAULT_VOICE,
    MIME_TYPES,
    SUPPORTED_FORMATS,
    ServerConfig,
    build_cli_parser,
)
from voice_manager import VoiceBlendSyntaxError, VoiceNotFoundError


log = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    model: str = Field(default="kokoro", description="Ignored; always kokoro.")
    input: str = Field(..., min_length=1)
    voice: str = Field(default=DEFAULT_VOICE)
    response_format: Literal["pcm", "wav", "mp3", "opus", "flac"] = DEFAULT_FORMAT
    speed: float = Field(default=1.0, gt=0.25, lt=4.0)
    stream: bool = Field(default=True, description="Ignored for WAV (always buffered).")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def build_app(cfg: ServerConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("starting Kokoro TTS server on %s:%d", cfg.host, cfg.port)
        app.state.cfg = cfg
        app.state.backend = KokoroBackend(cfg)
        app.state.sem = asyncio.Semaphore(cfg.max_concurrent_requests)
        app.state.active_requests = 0
        yield
        log.info("shutting down Kokoro TTS server")

    app = FastAPI(
        title="Kokoro-82M TTS Server",
        description="OpenAI-compatible TTS endpoint for DGX Spark.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        backend: KokoroBackend = request.app.state.backend
        cfg: ServerConfig = request.app.state.cfg
        data = backend.health()
        data["active_requests"] = request.app.state.active_requests
        data["max_concurrent"] = cfg.max_concurrent_requests
        return JSONResponse(data)

    @app.get("/v1/audio/voices")
    async def list_voices(request: Request) -> JSONResponse:
        backend: KokoroBackend = request.app.state.backend
        return JSONResponse({"voices": backend.voices.list_voices()})

    @app.post("/v1/audio/speech")
    async def speech(request: Request, body: SpeechRequest):
        backend: KokoroBackend = request.app.state.backend
        cfg: ServerConfig = request.app.state.cfg
        sem: asyncio.Semaphore = request.app.state.sem

        if body.response_format not in SUPPORTED_FORMATS:
            raise HTTPException(400, f"unsupported response_format {body.response_format!r}")

        # Non-blocking semaphore acquire — 429 immediately if full.
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=429, detail="server busy, try again")

        request.app.state.active_requests += 1

        # Resolve voice eagerly so we 4xx before starting the stream.
        try:
            backend.voices.get(body.voice)
        except (VoiceNotFoundError, VoiceBlendSyntaxError) as exc:
            request.app.state.active_requests -= 1
            sem.release()
            raise HTTPException(400, str(exc)) from exc

        fmt = body.response_format
        mime = MIME_TYPES[fmt]
        sample_rate = cfg.sample_rate

        t_request = time.perf_counter()

        # WAV must be buffered (needs length in header). Everything else streams.
        if fmt == "wav":
            try:
                encoder = make_encoder(
                    fmt,
                    sample_rate,
                    mp3_bitrate=cfg.mp3_bitrate,
                    opus_bitrate=cfg.opus_bitrate,
                    flac_compression=cfg.flac_compression,
                )
                async for pcm in backend.generate(body.input, body.voice, body.speed):
                    await asyncio.to_thread(encoder.feed, pcm)
                payload = await asyncio.to_thread(encoder.close)
            finally:
                request.app.state.active_requests -= 1
                sem.release()
            log.info(
                "speech: wav, %d bytes, %.3fs, voice=%s",
                len(payload),
                time.perf_counter() - t_request,
                body.voice,
            )
            return Response(content=payload, media_type=mime)

        # Streaming path for PCM/FLAC/MP3/Opus.
        async def stream_gen():
            t_first = None
            total = 0
            try:
                encoder = make_encoder(
                    fmt,
                    sample_rate,
                    mp3_bitrate=cfg.mp3_bitrate,
                    opus_bitrate=cfg.opus_bitrate,
                    flac_compression=cfg.flac_compression,
                )
                async for pcm in backend.generate(body.input, body.voice, body.speed):
                    data = await asyncio.to_thread(encoder.feed, pcm)
                    if data:
                        if t_first is None:
                            t_first = time.perf_counter()
                            log.info(
                                "ttfa: %.3fs (%s, voice=%s)",
                                t_first - t_request,
                                fmt,
                                body.voice,
                            )
                        total += len(data)
                        yield data
                tail = await asyncio.to_thread(encoder.close)
                if tail:
                    total += len(tail)
                    yield tail
            finally:
                request.app.state.active_requests -= 1
                sem.release()
                log.info(
                    "speech: %s, %d bytes, %.3fs total, voice=%s",
                    fmt,
                    total,
                    time.perf_counter() - t_request,
                    body.voice,
                )

        headers = {"X-Sample-Rate": str(sample_rate), "X-Audio-Format": fmt}
        return StreamingResponse(stream_gen(), media_type=mime, headers=headers)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


# Module-level ASGI app so `uvicorn server:app` works directly.
# Configure via env vars (KOKORO_HOST, KOKORO_PORT, KOKORO_MODEL_DIR,
# KOKORO_DTYPE, KOKORO_BATCH_SIZE, KOKORO_DEVICE) — see config.py.
_setup_logging("info")
app = build_app(ServerConfig.from_env())


def main() -> None:
    """CLI entry point — use when you want argparse flags instead of env vars."""
    import uvicorn

    parser = build_cli_parser("kokoro-tts-server")
    ns = parser.parse_args()
    _setup_logging(ns.log_level)
    cfg = ServerConfig.from_cli(ns)
    app_obj = build_app(cfg)
    uvicorn.run(app_obj, host=cfg.host, port=cfg.port, log_level=cfg.log_level)


if __name__ == "__main__":
    main()

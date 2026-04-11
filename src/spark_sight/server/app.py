"""FastAPI application — iPhone ↔ GB10 WebSocket bridge.

Serves the iPhone client HTML page and manages a single unified WebSocket
connection that carries camera frames, mic audio, TTS audio, and status
updates between the iPhone and the AI agents on the GB10.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from spark_sight.server.frame_buffer import FrameBuffer
from spark_sight.server.protocol import (
    MessageType,
    pack_status,
    pack_tts,
    unpack_message,
)

logger = logging.getLogger(__name__)

# Resolve the static directory relative to this file.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(
    frame_buffer: FrameBuffer | None = None,
    *,
    debug: bool = False,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    frame_buffer:
        Shared ring buffer for camera frames.  Created internally if not
        provided.
    debug:
        Enable the ``/debug`` endpoint and verbose WebSocket logging.
    """
    frame_buffer = frame_buffer or FrameBuffer()

    app = FastAPI(title="Spark Sight", debug=debug)

    # Queues that feed the WebSocket send loop.
    tts_queue: asyncio.Queue[bytes] = asyncio.Queue()
    status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # Mic audio arrives here; an ASR consumer (not yet wired) will drain it.
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # Track connected clients (currently only one expected).
    connected_clients: set[WebSocket] = set()

    # Debug log — recent events (capped).
    debug_log: deque[dict[str, Any]] = deque(maxlen=200)

    # ── Store shared objects on app.state for external access ────────
    app.state.frame_buffer = frame_buffer
    app.state.tts_queue = tts_queue
    app.state.status_queue = status_queue
    app.state.audio_queue = audio_queue
    app.state.connected_clients = connected_clients
    app.state.debug_log = debug_log

    # ── Public helpers (called by Orchestrator / agents) ─────────────

    async def speak(wav_bytes: bytes) -> None:
        """Enqueue TTS audio to be sent to the iPhone."""
        await tts_queue.put(wav_bytes)

    async def push_status(
        signal: str,
        message: str | None = None,
        mode: str = "patrol",
        goal: str | None = None,
    ) -> None:
        """Enqueue a status update to be sent to the iPhone HUD."""
        payload: dict[str, Any] = {
            "type": "status",
            "signal": signal,
            "message": message,
            "mode": mode,
            "goal": goal,
            "ts": time.time(),
        }
        await status_queue.put(payload)
        if debug:
            debug_log.append(payload)

    app.state.speak = speak
    app.state.push_status = push_status

    # ── Routes ───────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def serve_client() -> HTMLResponse:
        """Serve the iPhone client page."""
        html_path = _STATIC_DIR / "client.html"
        if not html_path.exists():
            return HTMLResponse("<h1>client.html not found</h1>", status_code=404)
        return HTMLResponse(html_path.read_text())

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "clients": len(connected_clients),
            "frames_received": frame_buffer.count,
            "buffer_size": frame_buffer.size,
            "debug": debug,
        }

    if debug:

        @app.get("/debug", response_class=JSONResponse)
        async def debug_info() -> dict[str, Any]:
            return {
                "clients": len(connected_clients),
                "frame_buffer": {
                    "count": frame_buffer.count,
                    "size": frame_buffer.size,
                    "max_size": frame_buffer.max_size,
                },
                "queues": {
                    "tts": tts_queue.qsize(),
                    "status": status_queue.qsize(),
                    "audio": audio_queue.qsize(),
                },
                "recent_events": list(debug_log),
            }

    # ── WebSocket ────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def unified_socket(ws: WebSocket) -> None:
        await ws.accept()
        connected_clients.add(ws)
        logger.info("Client connected (%d total)", len(connected_clients))

        async def send_loop() -> None:
            """Push TTS audio and status updates to the iPhone."""
            while True:
                # Wait for either queue to have something.
                tts_task = asyncio.create_task(tts_queue.get(), name="tts")
                status_task = asyncio.create_task(status_queue.get(), name="status")

                done, pending = await asyncio.wait(
                    {tts_task, status_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

                for task in done:
                    result = task.result()
                    if isinstance(result, bytes):
                        await ws.send_bytes(pack_tts(result))
                    elif isinstance(result, dict):
                        await ws.send_bytes(pack_status(result))

        sender = asyncio.create_task(send_loop())

        try:
            while True:
                data = await ws.receive_bytes()
                try:
                    msg_type, payload = unpack_message(data)
                except ValueError:
                    logger.warning("Bad WebSocket message (len=%d)", len(data))
                    continue

                if msg_type == MessageType.FRAME:
                    frame_buffer.push(payload)
                    if debug:
                        debug_log.append(
                            {
                                "type": "frame",
                                "size": len(payload),
                                "ts": time.time(),
                            }
                        )

                elif msg_type == MessageType.AUDIO:
                    await audio_queue.put(payload)

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket error")
        finally:
            sender.cancel()
            connected_clients.discard(ws)

    return app

"""FastAPI application — iPhone ↔ GB10 WebSocket bridge.

Serves the iPhone client HTML page and manages a single unified WebSocket
connection that carries camera frames, mic audio, TTS audio, and status
updates between the iPhone and the AI agents on the GB10.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
from openai import AsyncOpenAI

from spark_sight.agents.planning.agent import _SYSTEM_PROMPT as _PLANNING_PROMPT
from spark_sight.config import get_settings
from spark_sight.server.frame_buffer import FrameBuffer
from spark_sight.server.protocol import (
    MessageType,
    pack_status,
    pack_tts,
    pack_text_response,
    pack_warning_text,
    unpack_message,
)

logger = logging.getLogger(__name__)

# Resolve the static directory relative to this file.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(
    frame_buffer: FrameBuffer | None = None,
    *,
    debug: bool = False,
    lifespan=None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    frame_buffer:
        Shared ring buffer for camera frames.  Created internally if not
        provided.
    debug:
        Enable the ``/debug`` endpoint and verbose WebSocket logging.
    lifespan:
        Optional async context manager for startup/shutdown (FastAPI lifespan).
    """
    frame_buffer = frame_buffer or FrameBuffer()

    app = FastAPI(title="Spark Sight", debug=debug, lifespan=lifespan)

    # Queues that feed the WebSocket send loop.
    tts_queue: asyncio.Queue[bytes] = asyncio.Queue()
    status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # Mic audio arrives here; an ASR consumer (not yet wired) will drain it.
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    # Text response queue for iOS clients (on-device TTS).
    text_response_queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()

    # Track connected clients (currently only one expected).
    connected_clients: set[WebSocket] = set()

    # Track which clients use on-device STT/TTS (iOS native).
    # When a client sends TRANSCRIPT (0x05), it's flagged as native iOS.
    ios_clients: set[WebSocket] = set()

    # Debug log — recent events (capped).
    debug_log: deque[dict[str, Any]] = deque(maxlen=200)

    # ── Store shared objects on app.state for external access ────────
    app.state.frame_buffer = frame_buffer
    app.state.tts_queue = tts_queue
    app.state.status_queue = status_queue
    app.state.audio_queue = audio_queue
    app.state.text_response_queue = text_response_queue
    app.state.connected_clients = connected_clients
    app.state.ios_clients = ios_clients
    app.state.debug_log = debug_log

    # ── Public helpers (called by Orchestrator / agents) ─────────────

    async def speak(wav_bytes: bytes) -> None:
        """Enqueue TTS audio to be sent to the web client."""
        await tts_queue.put(wav_bytes)

    async def send_text_response(text: str, *, final: bool = True) -> None:
        """Enqueue a text response for iOS clients (on-device TTS)."""
        await text_response_queue.put((text, final))

    async def send_warning_text(text: str, urgency: str = "high") -> None:
        """Send an urgent warning to iOS clients (interrupts speech)."""
        msg = pack_warning_text(text, urgency)
        for ws in list(ios_clients):
            try:
                await ws.send_bytes(msg)
            except Exception:
                logger.warning("Failed to send warning to iOS client")

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
    app.state.send_text_response = send_text_response
    app.state.send_warning_text = send_warning_text
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

    # ── Streaming chat endpoint ──────────────────────────────────────

    def _strip_think(text: str) -> str:
        """Remove chain-of-thought reasoning blocks."""
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Handle missing opening <think> tag.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text

    def _parse_planning_json(text: str) -> dict[str, Any]:
        """Best-effort parse of a Planning Agent JSON response."""
        text = _strip_think(text)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(text)
            return {
                "action": data.get("action", "answer"),
                "message": data.get("message", ""),
                "goal": data.get("goal"),
                "nyc_context": data.get("nyc_context"),
                "inspect_prompt": data.get("inspect_prompt"),
            }
        except (json.JSONDecodeError, ValueError):
            return {"action": "answer", "message": text[:500] if text else ""}

    @app.post("/api/chat")
    async def chat_endpoint(request: Request) -> StreamingResponse:
        """Stream a Planning Agent response for the chat UI.

        Accepts ``{"message": str}``.  Returns an SSE stream of tokens
        followed by a final ``done`` event with the parsed action.
        After streaming, executes any side-effects (set_goal, etc.)
        via the Orchestrator.
        """
        body = await request.json()
        user_message = body.get("message", "")
        if not user_message.strip():
            return JSONResponse({"error": "empty message"}, status_code=400)

        # Build context from current prompt state.
        orchestrator = getattr(app.state, "orchestrator", None)
        prompt_state = getattr(app.state, "prompt_state", None)

        snap_mode, snap_goal, snap_nyc = "patrol", "none", "none"
        if prompt_state:
            snap = prompt_state.get_snapshot()
            snap_mode = snap.mode
            snap_goal = snap.active_goal or "none"
            snap_nyc = snap.nyc_context or "none"

        settings = get_settings()
        user_content = (
            f"User said: {user_message}\n\n"
            f"Current mode: {snap_mode}\n"
            f"Active goal: {snap_goal}\n"
            f"NYC context: {snap_nyc}"
        )

        async def event_stream():
            full_text = ""
            t_start = time.time()
            client = AsyncOpenAI(
                base_url=settings.nemotron.nim_url,
                api_key="not-needed",
            )
            # Emit debug: request info
            yield f"data: {json.dumps({'type': 'debug', 'event': 'request', 'model': settings.nemotron.model, 'endpoint': settings.nemotron.nim_url, 'system_prompt_len': len(_PLANNING_PROMPT), 'user_content': user_content})}\n\n"

            try:
                stream = await client.chat.completions.create(
                    model=settings.nemotron.model,
                    messages=[
                        {"role": "system", "content": _PLANNING_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=512,
                    temperature=0.2,
                    stream=True,
                )
                first_token_time = None
                token_count = 0
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_text += token
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time()
                        # Stream every token — client-side JS handles
                        # <think> stripping and display.
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                t_end = time.time()
                # Parse completed response (strips <think> server-side for action execution).
                parsed = _parse_planning_json(full_text)
                yield f"data: {json.dumps({'type': 'done', **parsed})}\n\n"

                # Emit debug: response stats
                yield f"data: {json.dumps({'type': 'debug', 'event': 'response', 'raw_length': len(full_text), 'raw_output': full_text, 'tokens': token_count, 'ttft_ms': round((first_token_time - t_start) * 1000) if first_token_time else None, 'total_ms': round((t_end - t_start) * 1000), 'parsed_action': parsed.get('action')})}\n\n"

                # Execute side-effects via Orchestrator.
                if orchestrator and parsed.get("action", "answer") != "answer":
                    from spark_sight.bridge.models import (
                        PlanningAction,
                        PlanningResponse,
                    )
                    try:
                        action = PlanningAction(parsed["action"])
                        resp = PlanningResponse(
                            action=action,
                            message=parsed.get("message", ""),
                            goal=parsed.get("goal"),
                            nyc_context=parsed.get("nyc_context"),
                            inspect_prompt=parsed.get("inspect_prompt"),
                        )
                        await orchestrator.handle_planning_response(resp)
                    except (ValueError, Exception):
                        logger.warning("Chat action execution failed", exc_info=True)

            except Exception:
                logger.exception("Chat streaming error")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Inference error — is the NIM running?'})}\n\n"
            finally:
                await client.close()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── WebSocket ────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def unified_socket(ws: WebSocket) -> None:
        await ws.accept()
        connected_clients.add(ws)
        is_ios = False
        logger.info("Client connected (%d total)", len(connected_clients))

        async def send_loop() -> None:
            """Push TTS audio/text and status updates to the client."""
            while True:
                waiters = [
                    asyncio.create_task(tts_queue.get(), name="tts"),
                    asyncio.create_task(status_queue.get(), name="status"),
                    asyncio.create_task(
                        text_response_queue.get(), name="text_resp"
                    ),
                ]

                done, pending = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()

                for task in done:
                    result = task.result()
                    if isinstance(result, tuple):
                        # Text response for iOS client (text, final).
                        text, final = result
                        if is_ios:
                            await ws.send_bytes(
                                pack_text_response(text, final=final)
                            )
                    elif isinstance(result, bytes):
                        # WAV TTS audio for web client.
                        if not is_ios:
                            await ws.send_bytes(pack_tts(result))
                    elif isinstance(result, dict):
                        # Status update for all clients.
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

                elif msg_type == MessageType.TRANSCRIPT:
                    # iOS client sending pre-transcribed text.
                    if not is_ios:
                        is_ios = True
                        ios_clients.add(ws)
                        logger.info("Client identified as iOS (on-device STT)")

                    transcript = payload.decode("utf-8", errors="replace").strip()
                    if transcript:
                        logger.info("iOS transcript: %s", transcript[:120])
                        orchestrator = getattr(app.state, "orchestrator", None)
                        if orchestrator:
                            await orchestrator.handle_transcript(transcript)

                elif msg_type == MessageType.LOCATION:
                    # iOS client GPS update.
                    try:
                        loc = json.loads(payload.decode("utf-8"))
                        lat = loc.get("lat", 0.0)
                        lng = loc.get("lng", 0.0)
                        logger.debug("GPS update: %.6f, %.6f", lat, lng)
                        # Store latest location on app state for NYC data queries.
                        app.state.last_location = (lat, lng)
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("Bad location payload")

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket error")
        finally:
            sender.cancel()
            connected_clients.discard(ws)
            ios_clients.discard(ws)

    return app

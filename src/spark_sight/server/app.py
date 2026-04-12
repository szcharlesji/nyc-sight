"""FastAPI application — iPhone ↔ GB10 WebSocket bridge.

Serves the iPhone client HTML page and manages a single unified WebSocket
connection that carries camera frames, transcript text, and status/speech
updates between the iPhone and the AI agents on the GB10.

Speech is handled natively on the iPhone via iOS SpeechSynthesis / Speech
Recognition — the server only sends text, not audio.
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

    # Queue that feeds the WebSocket send loop.
    status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # Track connected clients (currently only one expected).
    connected_clients: set[WebSocket] = set()

    # Debug log — recent events (capped).
    debug_log: deque[dict[str, Any]] = deque(maxlen=200)

    # ── Store shared objects on app.state for external access ────────
    app.state.frame_buffer = frame_buffer
    app.state.status_queue = status_queue
    app.state.connected_clients = connected_clients
    app.state.debug_log = debug_log

    # ── Public helpers (called by Orchestrator / agents) ─────────────

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

    async def push_speech(priority: str, text: str) -> None:
        """Enqueue speech text to be sent to the iPhone for native TTS."""
        payload: dict[str, Any] = {
            "type": "speech",
            "priority": str(priority),
            "text": text,
            "ts": time.time(),
        }
        await status_queue.put(payload)
        if debug:
            debug_log.append(payload)

    app.state.push_status = push_status
    app.state.push_speech = push_speech

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
                    "status": status_queue.qsize(),
                },
                "recent_events": list(debug_log),
            }

    # ── Streaming chat endpoint ──────────────────────────────────────

    def _strip_think(text: str) -> str:
        """Remove chain-of-thought reasoning blocks."""
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        if "</thought>" in text:
            text = text.split("</thought>", 1)[1].strip()
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
            import os
            client = AsyncOpenAI(
                base_url=settings.nemotron.nim_url,
                api_key=os.environ.get("GEMINI_API_KEY") or "not-needed",
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
                        inspect_text = await orchestrator.handle_planning_response(resp)
                        if inspect_text:
                            yield f"data: {json.dumps({'type': 'inspect_result', 'message': inspect_text})}\n\n"
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

    # ── WebSocket ─────────────────────────────��──────────────────────

    @app.websocket("/ws")
    async def unified_socket(ws: WebSocket) -> None:
        await ws.accept()
        connected_clients.add(ws)
        logger.info("Client connected (%d total)", len(connected_clients))

        # Resolve the orchestrator for transcript routing.
        orchestrator = getattr(app.state, "orchestrator", None)

        async def send_loop() -> None:
            """Push status and speech updates to the iPhone."""
            while True:
                payload = await status_queue.get()
                await ws.send_bytes(pack_status(payload))

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

                elif msg_type == MessageType.TRANSCRIPT:
                    transcript = payload.decode("utf-8", errors="replace").strip()
                    if transcript and orchestrator:
                        asyncio.create_task(
                            orchestrator.handle_transcript(transcript)
                        )

                elif msg_type == MessageType.LOCATION:
                    try:
                        loc = json.loads(payload.decode("utf-8"))
                        loc["ts"] = time.time()
                        if orchestrator:
                            orchestrator.user_location = loc
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket error")
        finally:
            sender.cancel()
            connected_clients.discard(ws)

    return app

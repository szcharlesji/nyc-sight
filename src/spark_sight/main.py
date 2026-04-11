"""Spark Sight — main entry point.

Wires up the Orchestrator, Ambient and Planning agents, the FrameBuffer,
TTS (Magpie), ASR (Parakeet), and the FastAPI server, then starts uvicorn.

Full data flow:
  iPhone camera  → WebSocket → FrameBuffer → AmbientAgent → Orchestrator
  iPhone mic     → WebSocket → audio_queue → ASR loop → Orchestrator → PlanningAgent
  Speech queue   → TTS loop  → Magpie NIM  → tts_queue → WebSocket → iPhone speaker
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn

from spark_sight.agents.ambient import AmbientAgent
from spark_sight.agents.planning import PlanningAgent
from spark_sight.bridge.orchestrator import Orchestrator
from spark_sight.bridge.prompt_state import PromptState
from spark_sight.server.app import create_app
from spark_sight.server.frame_buffer import FrameBuffer
from spark_sight.speech.asr import ASRClient, asr_loop
from spark_sight.speech.tts import TTSClient, tts_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_app(*, debug: bool = False):
    """Construct all components and return the FastAPI app."""
    # Shared state
    state = PromptState()
    frame_buffer = FrameBuffer()

    # Agents
    ambient = AmbientAgent(state)
    planning = PlanningAgent(state)

    # Speech clients
    tts_client = TTSClient()
    asr_client = ASRClient()

    # Server (creates its own queues)
    app = create_app(frame_buffer, debug=debug)

    # Orchestrator — wired to server callbacks for push notifications.
    orchestrator = Orchestrator(
        state,
        ambient_agent=ambient,
        planning_agent=planning,
        frame_buffer=frame_buffer,
        on_status=app.state.push_status,
    )

    # Store references on app.state so lifespan events or tests can access them.
    app.state.orchestrator = orchestrator
    app.state.ambient_agent = ambient
    app.state.planning_agent = planning
    app.state.prompt_state = state
    app.state.tts_client = tts_client
    app.state.asr_client = asr_client

    # Background task handles.
    app.state._bg_tasks: list[asyncio.Task] = []

    @app.on_event("startup")
    async def startup() -> None:
        # Start all clients.
        await ambient.start()
        await planning.start()
        await tts_client.start()
        await asr_client.start()

        # Start background loops.
        tasks = app.state._bg_tasks

        # 1. Ambient frame-processing loop.
        tasks.append(asyncio.create_task(
            orchestrator.run_ambient_loop(),
            name="ambient-loop",
        ))

        # 2. TTS synthesis loop: speech queue → Magpie → WAV → iPhone.
        tasks.append(asyncio.create_task(
            tts_loop(orchestrator, tts_client, app.state.tts_queue),
            name="tts-loop",
        ))

        # 3. ASR transcription loop: iPhone mic → Parakeet → Planning Agent.
        tasks.append(asyncio.create_task(
            asr_loop(
                app.state.audio_queue,
                asr_client,
                orchestrator.handle_transcript,
            ),
            name="asr-loop",
        ))

        logger.info(
            "Spark Sight started — agents live, TTS/ASR active, server ready"
        )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        # Cancel all background tasks.
        for task in app.state._bg_tasks:
            task.cancel()
        for task in app.state._bg_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        app.state._bg_tasks.clear()

        # Stop all clients.
        await asr_client.stop()
        await tts_client.stop()
        await ambient.stop()
        await planning.stop()
        logger.info("Spark Sight shut down")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Sight server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=3000, help="Bind port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--ssl-keyfile", default=None, help="SSL key file (for Safari camera access)")
    parser.add_argument("--ssl-certfile", default=None, help="SSL cert file")
    args = parser.parse_args()

    app = build_app(debug=args.debug)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()

"""Spark Sight — main entry point.

Wires up the Orchestrator, Ambient and Planning agents, the FrameBuffer,
and the FastAPI server, then starts uvicorn.

Full data flow:
  iPhone camera  → WebSocket → FrameBuffer → AmbientAgent → Orchestrator
  iPhone speech  → iOS SpeechRecognition → WebSocket transcript → Orchestrator → PlanningAgent
  Speech text    → WebSocket status/speech → iPhone → iOS SpeechSynthesis
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn

from spark_sight.agents.ambient import AmbientAgent
from spark_sight.agents.planning import PlanningAgent
from spark_sight.agents.warning import WarningAgent
from spark_sight.bridge.orchestrator import Orchestrator
from spark_sight.bridge.prompt_state import PromptState
from spark_sight.server.app import create_app
from spark_sight.server.frame_buffer import FrameBuffer

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
    warning = WarningAgent()

    # Server (creates its own queues) — lifespan handles startup/shutdown.
    app = create_app(frame_buffer, debug=debug, lifespan=_lifespan)

    # Orchestrator — wired to server callbacks for push notifications.
    orchestrator = Orchestrator(
        state,
        ambient_agent=ambient,
        planning_agent=planning,
        frame_buffer=frame_buffer,
        warning_agent=warning,
        on_speech=app.state.push_speech,
        on_status=app.state.push_status,
    )

    # Store references on app.state so lifespan events or tests can access them.
    app.state.orchestrator = orchestrator
    app.state.ambient_agent = ambient
    app.state.planning_agent = planning
    app.state.warning_agent = warning
    app.state.prompt_state = state

    return app


@asynccontextmanager
async def _lifespan(app):
    """Start/stop agents and background loops."""
    ambient = app.state.ambient_agent
    planning = app.state.planning_agent
    warning = app.state.warning_agent
    orchestrator = app.state.orchestrator

    # Start agents.
    await ambient.start()
    await planning.start()
    await warning.start()

    # Start background loops.
    bg_tasks = [
        asyncio.create_task(orchestrator.run_ambient_loop(), name="ambient-loop"),
        asyncio.create_task(orchestrator.run_warning_loop(), name="warning-loop"),
    ]

    logger.info("Spark Sight started — agents live, server ready")
    try:
        yield
    finally:
        for task in bg_tasks:
            task.cancel()
        for task in bg_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        await warning.stop()
        await ambient.stop()
        await planning.stop()
        logger.info("Spark Sight shut down")


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

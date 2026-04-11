"""Spark Sight — main entry point.

Wires up the Orchestrator, Ambient and Planning agents, the FrameBuffer,
and the FastAPI server, then starts uvicorn.
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

    # Server (creates its own queues)
    app = create_app(frame_buffer, debug=debug)

    # Orchestrator — wired to server callbacks for push notifications.
    orchestrator = Orchestrator(
        state,
        ambient_agent=ambient,
        planning_agent=planning,
        frame_buffer=frame_buffer,
        on_speech=None,  # TTS not yet wired
        on_status=app.state.push_status,
    )

    # Store references on app.state so lifespan events or tests can access them.
    app.state.orchestrator = orchestrator
    app.state.ambient_agent = ambient
    app.state.planning_agent = planning
    app.state.prompt_state = state

    # Background task handle for the ambient loop.
    app.state._ambient_loop_task: asyncio.Task | None = None

    @app.on_event("startup")
    async def startup() -> None:
        await ambient.start()
        await planning.start()
        # Start the continuous ambient frame-processing loop.
        app.state._ambient_loop_task = asyncio.create_task(
            orchestrator.run_ambient_loop()
        )
        logger.info("Spark Sight started — agents live, server ready")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        # Stop the ambient loop.
        if app.state._ambient_loop_task is not None:
            app.state._ambient_loop_task.cancel()
            try:
                await app.state._ambient_loop_task
            except asyncio.CancelledError:
                pass
        await ambient.stop()
        await planning.stop()
        logger.info("Spark Sight shut down")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Sight server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
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

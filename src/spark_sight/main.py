"""Spark Sight — main entry point.

Wires up the Orchestrator, Ambient Agent, and Planning Agent, then runs
the main event loop.  Currently a minimal scaffold for testing the bridge.
"""

from __future__ import annotations

import asyncio
import logging

from spark_sight.agents.ambient import AmbientAgent
from spark_sight.agents.planning import PlanningAgent
from spark_sight.bridge.orchestrator import Orchestrator
from spark_sight.bridge.prompt_state import PromptState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    # Shared state
    state = PromptState()

    # Agents
    ambient = AmbientAgent(state)
    planning = PlanningAgent(state)

    # Orchestrator
    _orchestrator = Orchestrator(state, ambient_agent=ambient, planning_agent=planning)

    # Lifecycle
    await ambient.start()
    await planning.start()
    logger.info("Spark Sight started — bridge active, agents in stub mode")

    try:
        # TODO: Replace with real event loop that:
        #   1. Pulls camera frames → Ambient Agent → Orchestrator
        #   2. Listens for ASR transcripts → Planning Agent → Orchestrator
        #   3. Drains speech queue → Magpie TTS → audio output
        #
        # For now, just keep alive so tests and manual poking work.
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await ambient.stop()
        await planning.stop()
        logger.info("Spark Sight shut down")


if __name__ == "__main__":
    asyncio.run(main())

"""Spark Sight — main entry point.

Wires up the Orchestrator, Ambient Agent, and Planning Agent, then runs
the main event loop with the continuous ambient frame-processing loop.
"""

from __future__ import annotations

import asyncio
import logging

from spark_sight.agents.ambient import AmbientAgent
from spark_sight.agents.planning import PlanningAgent
from spark_sight.bridge.frame_buffer import FrameBuffer
from spark_sight.bridge.orchestrator import Orchestrator
from spark_sight.bridge.prompt_state import PromptState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _speech_consumer(orchestrator: Orchestrator) -> None:
    """Drain the speech queue and log (Magpie TTS integration placeholder)."""
    while True:
        priority, text = await orchestrator.next_speech()
        # TODO: Replace with Magpie TTS call.
        # WARNING priority should preempt any currently playing audio.
        logger.info("[TTS %s]: %s", priority, text)


async def main() -> None:
    # Shared state
    state = PromptState()
    frame_buffer = FrameBuffer(maxlen=30)

    # Agents
    ambient = AmbientAgent(state)
    planning = PlanningAgent(state)

    # Orchestrator
    orchestrator = Orchestrator(
        state,
        ambient_agent=ambient,
        planning_agent=planning,
        frame_buffer=frame_buffer,
    )

    # Lifecycle
    await ambient.start()
    await planning.start()
    logger.info("Spark Sight started — agents live")

    try:
        await asyncio.gather(
            orchestrator.run_ambient_loop(),
            _speech_consumer(orchestrator),
            # TODO: _camera_feed(frame_buffer)  — wire iPhone camera stream
            # TODO: _asr_listener(orchestrator, planning)  — wire Parakeet ASR
        )
    except asyncio.CancelledError:
        pass
    finally:
        await ambient.stop()
        await planning.stop()
        logger.info("Spark Sight shut down")


if __name__ == "__main__":
    asyncio.run(main())

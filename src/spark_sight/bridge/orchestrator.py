"""Orchestrator — the central coordination layer.

Routes signals between the Ambient and Planning agents, manages prompt state
transitions, and enforces the speech priority protocol.

The Orchestrator does NOT run inference itself.  It calls into agents via
their public interfaces and updates the shared :class:`PromptState`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from spark_sight.bridge.models import (
    AmbientResponse,
    AmbientSignal,
    PlanningAction,
    PlanningResponse,
)
from spark_sight.bridge.prompt_state import PromptState

if TYPE_CHECKING:
    from spark_sight.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class SpeechPriority(StrEnum):
    """Priority levels for the speech queue."""

    WARNING = "warning"  # preempts everything
    AMBIENT = "ambient"  # normal ambient output
    PLANNING = "planning"  # queues behind ambient


# Callback type aliases for external consumers (e.g. the server layer).
SpeechCallback = Callable[[SpeechPriority, str], Awaitable[None]]
StatusCallback = Callable[[str, str | None, str, str | None], Awaitable[None]]


class Orchestrator:
    """Wires agents together and mediates all communication.

    Responsibilities:
    - Dispatch Ambient Agent signals (GOAL_REACHED → reset, FAILURE → replan).
    - Dispatch Planning Agent actions (set_goal, inspect, reset, etc.).
    - Manage the speech output queue (WARNING preempts, planning queues).

    Parameters
    ----------
    prompt_state:
        The shared mutable state between agents.
    ambient_agent:
        The Ambient Agent instance (or ``None`` during bridge-only testing).
    planning_agent:
        The Planning Agent instance (or ``None`` during bridge-only testing).
    """

    def __init__(
        self,
        prompt_state: PromptState,
        ambient_agent: BaseAgent | None = None,
        planning_agent: BaseAgent | None = None,
        *,
        on_speech: SpeechCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.state = prompt_state
        self.ambient_agent = ambient_agent
        self.planning_agent = planning_agent

        # External callbacks (registered by the server layer).
        self._on_speech = on_speech
        self._on_status = on_status

        # Speech queue: list of (priority, text) — drained by TTS consumer.
        self._speech_queue: asyncio.Queue[tuple[SpeechPriority, str]] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Ambient signal dispatch
    # ------------------------------------------------------------------

    async def handle_ambient_response(self, response: AmbientResponse) -> None:
        """Process a signal emitted by the Ambient Agent.

        Routing rules:
        - CLEAR → do nothing.
        - WARNING → enqueue speech at highest priority.
        - PROGRESS / CORRECTION → enqueue speech at ambient priority.
        - GOAL_REACHED → speak confirmation, reset prompt state to patrol.
        - FAILURE → speak status, trigger Planning Agent replan.
        """
        match response.signal:
            case AmbientSignal.CLEAR:
                return

            case AmbientSignal.WARNING:
                await self._enqueue_speech(SpeechPriority.WARNING, response.message)
                await self._emit_status(response)

            case AmbientSignal.PROGRESS | AmbientSignal.CORRECTION:
                await self._enqueue_speech(SpeechPriority.AMBIENT, response.message)
                await self._emit_status(response)

            case AmbientSignal.GOAL_REACHED:
                await self._enqueue_speech(SpeechPriority.AMBIENT, response.message)
                self.state.reset_goal()
                logger.info("Goal reached — reverted to patrol")
                await self._emit_status(response)

            case AmbientSignal.FAILURE:
                await self._enqueue_speech(SpeechPriority.AMBIENT, response.message)
                self.state.reset_goal()
                logger.warning("Ambient FAILURE: %s", response.message)
                await self._emit_status(response)
                # Trigger replan on the Planning Agent.
                await self._trigger_replan(response.message)

    # ------------------------------------------------------------------
    # Planning action dispatch
    # ------------------------------------------------------------------

    async def handle_planning_response(self, response: PlanningResponse) -> None:
        """Process an action produced by the Planning Agent.

        Routing rules:
        - set_goal / replan → update prompt state, speak to user.
        - inspect → forward one-shot query to Ambient Agent.
        - answer → speak directly (no vision needed).
        - reset → clear goal, revert to patrol.
        """
        match response.action:
            case PlanningAction.SET_GOAL | PlanningAction.REPLAN:
                if response.goal:
                    self.state.set_goal(
                        response.goal,
                        nyc_context=response.nyc_context or "",
                    )
                if response.message:
                    await self._enqueue_speech(
                        SpeechPriority.PLANNING, response.message
                    )

            case PlanningAction.INSPECT:
                # Grab latest frame (placeholder) and query Ambient Agent.
                if response.message:
                    await self._enqueue_speech(
                        SpeechPriority.PLANNING, response.message
                    )
                # TODO: grab frame from buffer, send to ambient with inspect_prompt

            case PlanningAction.ANSWER:
                if response.message:
                    await self._enqueue_speech(
                        SpeechPriority.PLANNING, response.message
                    )

            case PlanningAction.RESET:
                self.state.reset_goal()
                if response.message:
                    await self._enqueue_speech(
                        SpeechPriority.PLANNING, response.message
                    )

    # ------------------------------------------------------------------
    # Speech queue
    # ------------------------------------------------------------------

    async def _enqueue_speech(self, priority: SpeechPriority, text: str) -> None:
        """Add a message to the speech output queue and notify callbacks."""
        if not text:
            return
        await self._speech_queue.put((priority, text))
        logger.debug("Speech enqueued [%s]: %s", priority, text[:80])
        if self._on_speech:
            await self._on_speech(priority, text)

    async def next_speech(self) -> tuple[SpeechPriority, str]:
        """Consume the next speech item (blocks until available).

        In the full system this feeds into Magpie TTS.
        WARNING-priority items should preempt any currently playing audio.
        """
        return await self._speech_queue.get()

    @property
    def speech_pending(self) -> bool:
        return not self._speech_queue.empty()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit_status(self, response: AmbientResponse) -> None:
        """Push a status update to the registered callback (if any)."""
        if self._on_status is None:
            return
        snap = self.state.get_snapshot()
        await self._on_status(
            response.signal,
            response.message or None,
            snap.mode,
            snap.active_goal,
        )

    async def _trigger_replan(self, failure_reason: str) -> None:
        """Ask the Planning Agent to generate a new plan after a FAILURE."""
        if self.planning_agent is None:
            logger.warning("No planning agent attached — cannot replan")
            return
        logger.info("Triggering replan: %s", failure_reason)
        # The planning agent's process() will be called by the main loop
        # with the failure reason as the trigger text.
        # For now we just log — actual invocation handled by the run loop.

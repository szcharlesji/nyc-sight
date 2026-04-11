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
    from spark_sight.agents.ambient import AmbientAgent
    from spark_sight.agents.base import BaseAgent
    from spark_sight.server.frame_buffer import FrameBuffer

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
    - Run the continuous ambient frame-processing loop.

    Parameters
    ----------
    prompt_state:
        The shared mutable state between agents.
    ambient_agent:
        The Ambient Agent instance (or ``None`` during bridge-only testing).
    planning_agent:
        The Planning Agent instance (or ``None`` during bridge-only testing).
    frame_buffer:
        Shared camera frame ring buffer (or ``None`` during bridge-only testing).
    on_speech:
        Async callback fired when speech is enqueued (e.g. to push to WebSocket).
    on_status:
        Async callback fired on ambient status changes (e.g. to push to HUD).
    """

    def __init__(
        self,
        prompt_state: PromptState,
        ambient_agent: AmbientAgent | None = None,
        planning_agent: BaseAgent | None = None,
        frame_buffer: FrameBuffer | None = None,
        *,
        on_speech: SpeechCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.state = prompt_state
        self.ambient_agent = ambient_agent
        self.planning_agent = planning_agent
        self.frame_buffer = frame_buffer

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
        - CLEAR → emit status (for frame counter) but no speech.
        - WARNING → enqueue speech at highest priority.
        - PROGRESS / CORRECTION → enqueue speech at ambient priority.
        - GOAL_REACHED → speak confirmation, reset prompt state to patrol.
        - FAILURE → speak status, trigger Planning Agent replan.
        """
        match response.signal:
            case AmbientSignal.CLEAR:
                await self._emit_status(response)
                return

            case AmbientSignal.WARNING:
                await self._enqueue_speech(SpeechPriority.WARNING, response.message)
                await self._emit_status(response)

            case AmbientSignal.PROGRESS | AmbientSignal.CORRECTION:
                await self._enqueue_speech(SpeechPriority.AMBIENT, response.message)
                await self._emit_status(response)

            case AmbientSignal.GOAL_REACHED:
                await self._enqueue_speech(SpeechPriority.AMBIENT, response.message)
                # Emit status BEFORE resetting so the client sees the completed goal.
                await self._emit_status(response)
                self.state.reset_goal()
                logger.info("Goal reached — reverted to patrol")

            case AmbientSignal.FAILURE:
                await self._enqueue_speech(SpeechPriority.AMBIENT, response.message)
                # Emit status BEFORE resetting so the client sees the failed goal.
                await self._emit_status(response)
                self.state.reset_goal()
                logger.warning("Ambient FAILURE: %s", response.message)
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
        logger.info(
            "[Planning] action=%s | msg=%s | goal=%s",
            response.action,
            (response.message or "-")[:80],
            (response.goal or "-")[:40],
        )
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
                if response.message:
                    await self._enqueue_speech(
                        SpeechPriority.PLANNING, response.message
                    )
                # Grab latest frame and query the Ambient Agent.
                if (
                    self.ambient_agent is not None
                    and self.frame_buffer is not None
                    and response.inspect_prompt
                ):
                    frame = self.frame_buffer.latest_base64()
                    if frame:
                        inspect_result = await self.ambient_agent.inspect(
                            frame, response.inspect_prompt
                        )
                        if inspect_result.message:
                            await self._enqueue_speech(
                                SpeechPriority.PLANNING, inspect_result.message
                            )
                    else:
                        await self._enqueue_speech(
                            SpeechPriority.PLANNING,
                            "I can't see anything right now — no camera frame available.",
                        )

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
    # Transcript handling (ASR → Planning Agent)
    # ------------------------------------------------------------------

    async def handle_transcript(self, transcript: str) -> None:
        """Route a voice transcript from ASR to the Planning Agent.

        This is the callback used by :func:`asr_loop` when Parakeet
        produces a non-empty transcription.
        """
        if not transcript.strip():
            return

        logger.info("Transcript received: %s", transcript[:120])

        # Push transcript to the iPhone HUD.
        if self._on_status:
            snap = self.state.get_snapshot()
            await self._on_status(
                "TRANSCRIPT",
                transcript,
                snap.mode,
                snap.active_goal,
            )

        if self.planning_agent is None:
            logger.warning("No planning agent attached — cannot process transcript")
            return

        planning_response = await self.planning_agent.process(
            {"transcript": transcript}
        )
        await self.handle_planning_response(planning_response)

    # ------------------------------------------------------------------
    # Ambient processing loop
    # ------------------------------------------------------------------

    async def run_ambient_loop(self) -> None:
        """Continuous frame-processing loop for the Ambient Agent.

        Only processes the **latest** frame and skips if it hasn't changed
        since the last inference (same timestamp).  This ensures the model
        always sees the freshest view and never re-processes stale frames.

        The loop is naturally throttled by NIM inference latency (~1-3s
        per frame on Cosmos Reason2-8B).
        """
        import time as _time

        if self.ambient_agent is None or self.frame_buffer is None:
            logger.warning("Ambient loop requires ambient_agent and frame_buffer")
            return

        logger.info("Ambient processing loop started")
        frame_num = 0
        last_ts = 0.0  # timestamp of the last frame we processed

        while True:
            # Grab the freshest frame.
            frame_obj = self.frame_buffer.latest()
            if frame_obj is None:
                await asyncio.sleep(0.1)  # no frames yet — back off
                continue

            # Skip if we already processed this exact frame.
            if frame_obj.timestamp <= last_ts:
                await asyncio.sleep(0.05)  # wait for a new frame
                continue

            last_ts = frame_obj.timestamp
            frame_num += 1

            # Re-grab the absolute latest right before inference
            # (a newer frame may have arrived while we were awaiting).
            fresh = self.frame_buffer.latest()
            if fresh is not None and fresh.timestamp > last_ts:
                frame_obj = fresh
                last_ts = fresh.timestamp

            import base64
            frame_b64 = base64.b64encode(frame_obj.jpeg).decode("ascii")

            t0 = _time.time()
            try:
                response = await self.ambient_agent.process(
                    {"frame_base64": frame_b64}
                )
                dt = round((_time.time() - t0) * 1000)
                snap = self.state.get_snapshot()
                logger.info(
                    "[Ambient #%d] %s | %dms | mode=%s | goal=%s | msg=%s",
                    frame_num,
                    response.signal,
                    dt,
                    snap.mode,
                    (snap.active_goal or "-")[:40],
                    (response.message or "-")[:60],
                )
                await self.handle_ambient_response(response)
            except Exception:
                logger.exception("Error in ambient loop iteration #%d", frame_num)

            # Yield to the event loop.
            await asyncio.sleep(0)

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
        # Ensure signal is a plain string (not StrEnum) for JSON serialization.
        await self._on_status(
            str(response.signal),
            response.message or None,
            str(snap.mode),
            snap.active_goal,
        )

    async def _trigger_replan(self, failure_reason: str) -> None:
        """Ask the Planning Agent to generate a new plan after a FAILURE."""
        if self.planning_agent is None:
            logger.warning("No planning agent attached — cannot replan")
            return
        logger.info("Triggering replan: %s", failure_reason)
        replan_response = await self.planning_agent.process(
            {"failure_reason": failure_reason}
        )
        await self.handle_planning_response(replan_response)

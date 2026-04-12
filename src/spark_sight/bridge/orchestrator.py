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
    from spark_sight.agents.warning import WarningAgent
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
        warning_agent: WarningAgent | None = None,
        *,
        on_speech: SpeechCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.state = prompt_state
        self.ambient_agent = ambient_agent
        self.planning_agent = planning_agent
        self.frame_buffer = frame_buffer
        self.warning_agent = warning_agent

        # External callbacks (registered by the server layer).
        self._on_speech = on_speech
        self._on_status = on_status

        # Latest GPS location from the iPhone (set by server app).
        self.user_location: dict | None = None  # {lat, lng, accuracy, ts}

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

    async def handle_planning_response(self, response: PlanningResponse) -> str | None:
        """Process an action produced by the Planning Agent.

        Routing rules:
        - set_goal / replan → update prompt state, speak to user.
        - inspect → forward one-shot query to Ambient Agent.
        - answer → speak directly (no vision needed).
        - reset → clear goal, revert to patrol.

        Returns the inspect result text for INSPECT actions, ``None`` otherwise.
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
                            return inspect_result.message
                        else:
                            fallback = "I couldn't get a clear reading — please try again."
                            await self._enqueue_speech(
                                SpeechPriority.PLANNING, fallback
                            )
                            return fallback
                    else:
                        no_frame = "I can't see anything right now — no camera frame available."
                        await self._enqueue_speech(
                            SpeechPriority.PLANNING, no_frame
                        )
                        return no_frame

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

            case PlanningAction.FIND_RESTROOM:
                if response.message:
                    await self._enqueue_speech(
                        SpeechPriority.PLANNING, response.message
                    )
                await self._handle_find_restroom()

    # ------------------------------------------------------------------
    # Transcript handling (ASR → Planning Agent)
    # ------------------------------------------------------------------

    async def handle_transcript(self, transcript: str) -> None:
        """Route a voice transcript to the Planning Agent.

        Called when the client sends a transcript (from native iOS speech
        recognition) over the WebSocket.
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

            import base64
            frame_b64 = base64.b64encode(frame_obj.jpeg).decode("ascii")

            t0 = _time.time()
            try:
                response = await self.ambient_agent.process(
                    {"frame_base64": frame_b64}
                )
                dt = round((_time.time() - t0) * 1000)
                frame_age = round((_time.time() - frame_obj.timestamp) * 1000)
                snap = self.state.get_snapshot()
                logger.info(
                    "[Ambient #%d] %s | %dms | age=%dms | mode=%s | goal=%s | msg=%s",
                    frame_num,
                    response.signal,
                    dt,
                    frame_age,
                    snap.mode,
                    (snap.active_goal or "-")[:40],
                    (response.message or "-")[:60],
                )
                await self.handle_ambient_response(response)
            except Exception:
                logger.exception("Error in ambient loop iteration #%d", frame_num)

            # Wait at least 4 seconds between frames.
            elapsed = _time.time() - t0
            if elapsed < 4.0:
                await asyncio.sleep(4.0 - elapsed)

    # ------------------------------------------------------------------
    # Warning loop (YOLO — fast, parallel to ambient loop)
    # ------------------------------------------------------------------

    async def run_warning_loop(self) -> None:
        """Fast obstacle-detection loop powered by the YOLO11 service.

        Runs in parallel with :meth:`run_ambient_loop`.  Processes every new
        frame at up to ~10 fps (capped by a 0.1 s sleep).  Only forwards
        WARNING signals — CLEAR responses are discarded silently.

        Gracefully no-ops if the YOLO service is unreachable.
        """
        import base64 as _base64

        if self.warning_agent is None or self.frame_buffer is None:
            logger.warning("Warning loop requires warning_agent and frame_buffer")
            return

        logger.info("Warning loop started")
        last_ts = 0.0

        while True:
            frame_obj = self.frame_buffer.latest()
            if frame_obj is None:
                await asyncio.sleep(0.1)
                continue

            if frame_obj.timestamp <= last_ts:
                await asyncio.sleep(1.0)
                continue

            last_ts = frame_obj.timestamp
            frame_b64 = _base64.b64encode(frame_obj.jpeg).decode("ascii")

            try:
                response = await self.warning_agent.process({"frame_base64": frame_b64})
                if response.signal != AmbientSignal.CLEAR:
                    await self.handle_ambient_response(response)
            except Exception:
                logger.exception("Error in warning loop")

            await asyncio.sleep(1.0)  # 2 fps

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    async def _enqueue_speech(self, priority: SpeechPriority, text: str) -> None:
        """Push speech text to the client via the on_speech callback.

        The client handles synthesis using native iOS SpeechSynthesis.
        """
        if not text:
            return
        logger.debug("Speech [%s]: %s", priority, text[:80])
        if self._on_speech:
            await self._on_speech(priority, text)

    # ------------------------------------------------------------------
    # NYC data lookups
    # ------------------------------------------------------------------

    async def _handle_find_restroom(self) -> None:
        """Query NYC Open Data for nearby restrooms and set a navigation goal."""
        if not self.user_location:
            await self._enqueue_speech(
                SpeechPriority.PLANNING,
                "I need your location to find restrooms. Please enable location services.",
            )
            return

        from spark_sight.data.restrooms import find_nearby_restrooms

        lat = self.user_location["lat"]
        lng = self.user_location["lng"]
        restrooms = await find_nearby_restrooms(lat, lng, limit=3)

        if not restrooms:
            await self._enqueue_speech(
                SpeechPriority.PLANNING,
                "Sorry, I couldn't find any nearby public restrooms right now.",
            )
            return

        nearest = restrooms[0]
        dist = nearest["distance_ft"]
        if dist < 528:  # < 0.1 miles
            dist_str = f"about {dist} feet"
        else:
            dist_str = f"about {dist / 5280:.1f} miles"

        speech = (
            f"The nearest public restroom is {nearest['name']}, "
            f"{dist_str} away. Hours: {nearest['hours']}."
        )
        if len(restrooms) > 1:
            speech += f" There's also one at {restrooms[1]['name']}."

        await self._enqueue_speech(SpeechPriority.PLANNING, speech)

        # Build context for the ambient agent.
        ctx_lines = []
        for r in restrooms:
            ctx_lines.append(
                f"- {r['name']} ({r['location_type']}, {r['operator']}): "
                f"{r['distance_ft']} ft away, hours: {r['hours']}, open: {r['open']}"
            )
        nyc_ctx = "Nearby public restrooms:\n" + "\n".join(ctx_lines)

        self.state.set_goal(
            f"Guide user to the nearest restroom at {nearest['name']}",
            nyc_context=nyc_ctx,
        )
        logger.info("Restroom goal set: %s", nearest["name"])

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

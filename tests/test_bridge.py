"""Tests for the core bridge: PromptState and Orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from spark_sight.agents.ambient import AmbientAgent
from spark_sight.server.frame_buffer import FrameBuffer
from spark_sight.bridge.models import (
    AgentMode,
    AmbientResponse,
    AmbientSignal,
    PlanningAction,
    PlanningResponse,
)
from spark_sight.bridge.orchestrator import Orchestrator, SpeechPriority
from spark_sight.bridge.prompt_state import PromptState


# ---------------------------------------------------------------------------
# PromptState
# ---------------------------------------------------------------------------


class TestPromptState:
    def test_initial_mode_is_patrol(self) -> None:
        state = PromptState()
        assert state.get_mode() == AgentMode.PATROL

    def test_set_goal_transitions_to_goal_mode(self) -> None:
        state = PromptState()
        state.set_goal("Go north", nyc_context="scaffolding ahead")
        assert state.get_mode() == AgentMode.GOAL
        snap = state.get_snapshot()
        assert snap.active_goal == "Go north"
        assert snap.nyc_context == "scaffolding ahead"

    def test_reset_goal_reverts_to_patrol(self) -> None:
        state = PromptState()
        state.set_goal("Go north")
        state.reset_goal()
        assert state.get_mode() == AgentMode.PATROL
        snap = state.get_snapshot()
        assert snap.active_goal is None
        assert snap.nyc_context == ""

    def test_compiled_prompt_patrol(self) -> None:
        state = PromptState()
        prompt = state.get_compiled_prompt()
        assert "navigation assistant" in prompt.lower()
        assert "ACTIVE GOAL" not in prompt

    def test_compiled_prompt_goal(self) -> None:
        state = PromptState()
        state.set_goal("Find 72nd St subway", nyc_context="elevator out")
        prompt = state.get_compiled_prompt()
        assert "ACTIVE GOAL: Find 72nd St subway" in prompt
        assert "NYC CONTEXT: elevator out" in prompt

    def test_snapshot_is_a_copy(self) -> None:
        state = PromptState()
        snap = state.get_snapshot()
        snap.mode = AgentMode.GOAL  # mutate the copy
        assert state.get_mode() == AgentMode.PATROL  # original unchanged

    def test_custom_base_goal(self) -> None:
        state = PromptState(base_goal="Custom base")
        assert "Custom base" in state.get_compiled_prompt()


# ---------------------------------------------------------------------------
# Helper: collect speech callback calls
# ---------------------------------------------------------------------------


def _make_speech_collector() -> tuple[AsyncMock, list[tuple[SpeechPriority, str]]]:
    """Return an (on_speech callback, list of (priority, text) calls)."""
    calls: list[tuple[SpeechPriority, str]] = []

    async def _on_speech(priority: SpeechPriority, text: str) -> None:
        calls.append((priority, text))

    return AsyncMock(side_effect=_on_speech), calls


# ---------------------------------------------------------------------------
# Orchestrator — Ambient signal routing
# ---------------------------------------------------------------------------


class TestOrchestratorAmbient:
    @pytest.mark.asyncio
    async def test_clear_produces_no_speech(self) -> None:
        state = PromptState()
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(state, on_speech=on_speech)
        resp = AmbientResponse(signal=AmbientSignal.CLEAR)
        await orch.handle_ambient_response(resp)
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_warning_enqueues_speech(self) -> None:
        state = PromptState()
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(state, on_speech=on_speech)
        resp = AmbientResponse(signal=AmbientSignal.WARNING, message="Watch out!")
        await orch.handle_ambient_response(resp)
        assert len(calls) == 1
        assert calls[0] == (SpeechPriority.WARNING, "Watch out!")

    @pytest.mark.asyncio
    async def test_goal_reached_resets_to_patrol(self) -> None:
        state = PromptState()
        state.set_goal("Find subway")
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(state, on_speech=on_speech)
        resp = AmbientResponse(
            signal=AmbientSignal.GOAL_REACHED, message="You arrived!"
        )
        await orch.handle_ambient_response(resp)
        assert state.get_mode() == AgentMode.PATROL
        assert len(calls) == 1
        assert calls[0][1] == "You arrived!"

    @pytest.mark.asyncio
    async def test_failure_resets_and_logs(self) -> None:
        state = PromptState()
        state.set_goal("Find subway")
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(state, on_speech=on_speech)
        resp = AmbientResponse(signal=AmbientSignal.FAILURE, message="Path blocked")
        await orch.handle_ambient_response(resp)
        assert state.get_mode() == AgentMode.PATROL
        assert len(calls) == 1
        assert calls[0][1] == "Path blocked"


# ---------------------------------------------------------------------------
# Orchestrator — Planning action routing
# ---------------------------------------------------------------------------


class TestOrchestratorPlanning:
    @pytest.mark.asyncio
    async def test_set_goal_updates_state(self) -> None:
        state = PromptState()
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(state, on_speech=on_speech)
        resp = PlanningResponse(
            action=PlanningAction.SET_GOAL,
            message="Guiding you to 72nd St",
            goal="Navigate to 72nd St subway",
            nyc_context="elevator working at north entrance",
        )
        await orch.handle_planning_response(resp)
        assert state.get_mode() == AgentMode.GOAL
        snap = state.get_snapshot()
        assert snap.active_goal == "Navigate to 72nd St subway"
        assert len(calls) == 1
        assert "72nd" in calls[0][1]

    @pytest.mark.asyncio
    async def test_answer_speaks_without_state_change(self) -> None:
        state = PromptState()
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(state, on_speech=on_speech)
        resp = PlanningResponse(
            action=PlanningAction.ANSWER,
            message="Yes, there is scaffolding nearby.",
        )
        await orch.handle_planning_response(resp)
        assert state.get_mode() == AgentMode.PATROL  # unchanged
        assert len(calls) == 1
        assert "scaffolding" in calls[0][1]

    @pytest.mark.asyncio
    async def test_reset_clears_goal(self) -> None:
        state = PromptState()
        orch = Orchestrator(state)
        state.set_goal("Active goal")
        resp = PlanningResponse(
            action=PlanningAction.RESET, message="Goal cancelled."
        )
        await orch.handle_planning_response(resp)
        assert state.get_mode() == AgentMode.PATROL


# ---------------------------------------------------------------------------
# Orchestrator — Inspect routing
# ---------------------------------------------------------------------------


class TestOrchestratorInspect:
    @pytest.mark.asyncio
    async def test_inspect_calls_ambient_agent(self) -> None:
        state = PromptState()
        buf = FrameBuffer()
        buf.push(b"test_frame")

        mock_ambient = AsyncMock(spec=AmbientAgent)
        mock_ambient.inspect.return_value = AmbientResponse(
            signal=AmbientSignal.CLEAR, message="The sign says Broadway"
        )

        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(
            state, ambient_agent=mock_ambient, frame_buffer=buf,
            on_speech=on_speech,
        )
        resp = PlanningResponse(
            action=PlanningAction.INSPECT,
            message="Let me look at that for you.",
            inspect_prompt="Read all visible text",
        )
        result = await orch.handle_planning_response(resp)

        # The orchestrator calls latest_base64() which base64-encodes the raw bytes.
        expected_b64 = buf.latest_base64()
        mock_ambient.inspect.assert_called_once_with(expected_b64, "Read all visible text")
        # Two speech calls: the planning message + the inspect result.
        assert len(calls) == 2
        assert "look at that" in calls[0][1]
        assert "Broadway" in calls[1][1]
        # Return value carries the inspect result for the chat endpoint.
        assert result == "The sign says Broadway"

    @pytest.mark.asyncio
    async def test_inspect_no_frame_speaks_fallback(self) -> None:
        state = PromptState()
        buf = FrameBuffer()  # empty

        mock_ambient = AsyncMock(spec=AmbientAgent)
        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(
            state, ambient_agent=mock_ambient, frame_buffer=buf,
            on_speech=on_speech,
        )
        resp = PlanningResponse(
            action=PlanningAction.INSPECT,
            inspect_prompt="Read all visible text",
        )
        result = await orch.handle_planning_response(resp)

        mock_ambient.inspect.assert_not_called()
        assert len(calls) == 1
        assert "no camera frame" in calls[0][1].lower()
        assert result is not None
        assert "no camera frame" in result.lower()

    @pytest.mark.asyncio
    async def test_inspect_empty_message_returns_error(self) -> None:
        state = PromptState()
        buf = FrameBuffer()
        buf.push(b"test_frame")

        mock_ambient = AsyncMock(spec=AmbientAgent)
        mock_ambient.inspect.return_value = AmbientResponse(
            signal=AmbientSignal.CLEAR, message=""
        )

        on_speech, calls = _make_speech_collector()
        orch = Orchestrator(
            state, ambient_agent=mock_ambient, frame_buffer=buf,
            on_speech=on_speech,
        )
        resp = PlanningResponse(
            action=PlanningAction.INSPECT,
            inspect_prompt="Read all visible text",
        )
        result = await orch.handle_planning_response(resp)

        assert result is not None
        assert "try again" in result.lower()
        assert len(calls) == 1
        assert "try again" in calls[0][1].lower()


# ---------------------------------------------------------------------------
# Orchestrator — Replan wiring
# ---------------------------------------------------------------------------


class TestOrchestratorReplan:
    @pytest.mark.asyncio
    async def test_failure_triggers_replan_on_planning_agent(self) -> None:
        state = PromptState()
        state.set_goal("Find subway")

        mock_planning = AsyncMock()
        mock_planning.process.return_value = PlanningResponse(
            action=PlanningAction.REPLAN,
            message="Rerouting via Broadway.",
            goal="Take Broadway north to 72nd St",
        )

        orch = Orchestrator(state, planning_agent=mock_planning)

        failure_resp = AmbientResponse(
            signal=AmbientSignal.FAILURE, message="Path blocked by construction"
        )
        await orch.handle_ambient_response(failure_resp)

        # Planning agent should have been called with the failure reason.
        mock_planning.process.assert_called_once_with(
            {"failure_reason": "Path blocked by construction"}
        )
        # The replan response should have set a new goal.
        assert state.get_mode() == AgentMode.GOAL
        snap = state.get_snapshot()
        assert snap.active_goal == "Take Broadway north to 72nd St"

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
# Orchestrator — Ambient signal routing
# ---------------------------------------------------------------------------


class TestOrchestratorAmbient:
    @pytest.fixture
    def setup(self) -> tuple[PromptState, Orchestrator]:
        state = PromptState()
        orch = Orchestrator(state)
        return state, orch

    @pytest.mark.asyncio
    async def test_clear_produces_no_speech(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        _, orch = setup
        resp = AmbientResponse(signal=AmbientSignal.CLEAR)
        await orch.handle_ambient_response(resp)
        assert not orch.speech_pending

    @pytest.mark.asyncio
    async def test_warning_enqueues_speech(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        _, orch = setup
        resp = AmbientResponse(signal=AmbientSignal.WARNING, message="Watch out!")
        await orch.handle_ambient_response(resp)
        assert orch.speech_pending
        priority, text = await orch.next_speech()
        assert priority == SpeechPriority.WARNING
        assert text == "Watch out!"

    @pytest.mark.asyncio
    async def test_goal_reached_resets_to_patrol(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        state, orch = setup
        state.set_goal("Find subway")
        resp = AmbientResponse(
            signal=AmbientSignal.GOAL_REACHED, message="You arrived!"
        )
        await orch.handle_ambient_response(resp)
        assert state.get_mode() == AgentMode.PATROL
        _, text = await orch.next_speech()
        assert text == "You arrived!"

    @pytest.mark.asyncio
    async def test_failure_resets_and_logs(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        state, orch = setup
        state.set_goal("Find subway")
        resp = AmbientResponse(signal=AmbientSignal.FAILURE, message="Path blocked")
        await orch.handle_ambient_response(resp)
        assert state.get_mode() == AgentMode.PATROL
        _, text = await orch.next_speech()
        assert text == "Path blocked"


# ---------------------------------------------------------------------------
# Orchestrator — Planning action routing
# ---------------------------------------------------------------------------


class TestOrchestratorPlanning:
    @pytest.fixture
    def setup(self) -> tuple[PromptState, Orchestrator]:
        state = PromptState()
        orch = Orchestrator(state)
        return state, orch

    @pytest.mark.asyncio
    async def test_set_goal_updates_state(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        state, orch = setup
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
        _, text = await orch.next_speech()
        assert "72nd" in text

    @pytest.mark.asyncio
    async def test_answer_speaks_without_state_change(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        state, orch = setup
        resp = PlanningResponse(
            action=PlanningAction.ANSWER,
            message="Yes, there is scaffolding nearby.",
        )
        await orch.handle_planning_response(resp)
        assert state.get_mode() == AgentMode.PATROL  # unchanged
        _, text = await orch.next_speech()
        assert "scaffolding" in text

    @pytest.mark.asyncio
    async def test_reset_clears_goal(
        self, setup: tuple[PromptState, Orchestrator]
    ) -> None:
        state, orch = setup
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

        orch = Orchestrator(
            state, ambient_agent=mock_ambient, frame_buffer=buf
        )
        resp = PlanningResponse(
            action=PlanningAction.INSPECT,
            message="Let me look at that for you.",
            inspect_prompt="Read all visible text",
        )
        await orch.handle_planning_response(resp)

        # The orchestrator calls latest_base64() which base64-encodes the raw bytes.
        expected_b64 = buf.latest_base64()
        mock_ambient.inspect.assert_called_once_with(expected_b64, "Read all visible text")
        # Two speech items: the planning message + the inspect result.
        _, text1 = await orch.next_speech()
        assert "look at that" in text1
        _, text2 = await orch.next_speech()
        assert "Broadway" in text2

    @pytest.mark.asyncio
    async def test_inspect_no_frame_speaks_fallback(self) -> None:
        state = PromptState()
        buf = FrameBuffer()  # empty

        mock_ambient = AsyncMock(spec=AmbientAgent)
        orch = Orchestrator(
            state, ambient_agent=mock_ambient, frame_buffer=buf
        )
        resp = PlanningResponse(
            action=PlanningAction.INSPECT,
            inspect_prompt="Read all visible text",
        )
        await orch.handle_planning_response(resp)

        mock_ambient.inspect.assert_not_called()
        _, text = await orch.next_speech()
        assert "no camera frame" in text.lower()


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

"""Tests for the Planning Agent: response parsing and NIM integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from spark_sight.agents.planning import PlanningAgent
from spark_sight.bridge.models import PlanningAction
from spark_sight.bridge.prompt_state import PromptState


def _make_completion(content: str):
    """Build a fake OpenAI ChatCompletion-like object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestPlanningResponseParsing:
    def setup_method(self) -> None:
        self.state = PromptState()
        self.agent = PlanningAgent(self.state)

    def test_parse_set_goal(self) -> None:
        resp = _make_completion(json.dumps({
            "action": "set_goal",
            "message": "I'll guide you to the subway.",
            "goal": "Navigate to 72nd St subway entrance",
            "nyc_context": "Elevator working at north entrance",
            "inspect_prompt": None,
        }))
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.SET_GOAL
        assert "subway" in result.message.lower()
        assert result.goal == "Navigate to 72nd St subway entrance"
        assert result.nyc_context == "Elevator working at north entrance"

    def test_parse_inspect(self) -> None:
        resp = _make_completion(json.dumps({
            "action": "inspect",
            "message": "Let me check that for you.",
            "goal": None,
            "nyc_context": None,
            "inspect_prompt": "Read all visible text and signs",
        }))
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.INSPECT
        assert result.inspect_prompt == "Read all visible text and signs"

    def test_parse_answer(self) -> None:
        resp = _make_completion(json.dumps({
            "action": "answer",
            "message": "The closest subway is at 72nd and Broadway.",
            "goal": None,
            "nyc_context": None,
            "inspect_prompt": None,
        }))
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.ANSWER
        assert "72nd" in result.message

    def test_parse_reset(self) -> None:
        resp = _make_completion(json.dumps({
            "action": "reset",
            "message": "Goal cancelled.",
        }))
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.RESET

    def test_parse_replan(self) -> None:
        resp = _make_completion(json.dumps({
            "action": "replan",
            "message": "Rerouting via Broadway.",
            "goal": "Take Broadway north to 72nd St",
        }))
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.REPLAN
        assert result.goal == "Take Broadway north to 72nd St"

    def test_invalid_json_falls_back_to_answer(self) -> None:
        resp = _make_completion("This is just plain text, not JSON")
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.ANSWER
        assert "plain text" in result.message

    def test_markdown_wrapped_json(self) -> None:
        content = '```json\n{"action": "answer", "message": "Hello!"}\n```'
        resp = _make_completion(content)
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.ANSWER
        assert result.message == "Hello!"

    def test_empty_content(self) -> None:
        resp = _make_completion("")
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.ANSWER

    def test_none_content(self) -> None:
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        result = self.agent._parse_response(resp)
        assert result.action == PlanningAction.ANSWER

    def test_unknown_action_falls_back(self) -> None:
        resp = _make_completion(json.dumps({
            "action": "unknown_action",
            "message": "Something unexpected",
        }))
        result = self.agent._parse_response(resp)
        # Should fall back to ANSWER since "unknown_action" is not a valid PlanningAction.
        assert result.action == PlanningAction.ANSWER


class TestPlanningProcess:
    @pytest.mark.asyncio
    async def test_empty_input_returns_answer(self) -> None:
        state = PromptState()
        agent = PlanningAgent(state)
        result = await agent.process({})
        assert result.action == PlanningAction.ANSWER

    @pytest.mark.asyncio
    async def test_stub_without_client(self) -> None:
        state = PromptState()
        agent = PlanningAgent(state)
        # Don't call start() — _client stays None.
        result = await agent.process({"transcript": "Hello"})
        assert result.action == PlanningAction.ANSWER
        assert "stub" in result.message.lower()

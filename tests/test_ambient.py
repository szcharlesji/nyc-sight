"""Tests for the Ambient Agent: frame buffer, response parsing, NIM integration, and loop."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from spark_sight.agents.ambient import AmbientAgent
from spark_sight.bridge.models import AmbientResponse, AmbientSignal
from spark_sight.bridge.orchestrator import Orchestrator
from spark_sight.bridge.prompt_state import PromptState
from spark_sight.server.frame_buffer import FrameBuffer


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_push_and_latest(self) -> None:
        buf = FrameBuffer(max_size=5)
        buf.push(b"frame1")
        buf.push(b"frame2")
        assert buf.latest() is not None
        assert buf.latest().jpeg == b"frame2"

    def test_empty_returns_none(self) -> None:
        buf = FrameBuffer(max_size=5)
        assert buf.latest() is None
        assert buf.latest_base64() is None

    def test_size(self) -> None:
        buf = FrameBuffer(max_size=5)
        assert buf.size == 0
        buf.push(b"a")
        buf.push(b"b")
        assert buf.size == 2

    def test_maxlen_eviction(self) -> None:
        buf = FrameBuffer(max_size=3)
        for i in range(5):
            buf.push(f"frame{i}".encode())
        assert buf.size == 3
        assert buf.latest().jpeg == b"frame4"

    def test_latest_base64(self) -> None:
        buf = FrameBuffer(max_size=5)
        buf.push(b"test")
        b64 = buf.latest_base64()
        assert b64 is not None
        import base64
        assert base64.b64decode(b64) == b"test"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _make_completion(content: str) -> Any:
    """Build a fake OpenAI ChatCompletion-like object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestAmbientResponseParsing:
    def setup_method(self) -> None:
        self.state = PromptState()
        self.agent = AmbientAgent(self.state)

    def test_parse_clear(self) -> None:
        resp = _make_completion(json.dumps({
            "signal": "CLEAR", "message": "", "reasoning": "nothing here"
        }))
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.CLEAR
        assert result.message == ""

    def test_parse_warning(self) -> None:
        resp = _make_completion(json.dumps({
            "signal": "WARNING", "message": "Bicycle approaching fast on your left",
            "reasoning": "detected cyclist"
        }))
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.WARNING
        assert "Bicycle" in result.message

    def test_parse_progress(self) -> None:
        resp = _make_completion(json.dumps({
            "signal": "PROGRESS", "message": "Subway entrance 50 feet ahead on your right",
            "reasoning": "sign visible"
        }))
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.PROGRESS

    def test_parse_goal_reached(self) -> None:
        resp = _make_completion(json.dumps({
            "signal": "GOAL_REACHED", "message": "You've arrived at the subway entrance",
            "reasoning": "entrance visible"
        }))
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.GOAL_REACHED

    def test_invalid_json_returns_clear(self) -> None:
        resp = _make_completion("this is not json at all")
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.CLEAR

    def test_unknown_signal_returns_clear(self) -> None:
        resp = _make_completion(json.dumps({
            "signal": "UNKNOWN_SIGNAL", "message": "what"
        }))
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.CLEAR

    def test_markdown_wrapped_json(self) -> None:
        content = '```json\n{"signal": "WARNING", "message": "Watch out!", "reasoning": "x"}\n```'
        resp = _make_completion(content)
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.WARNING
        assert result.message == "Watch out!"

    def test_empty_content_returns_clear(self) -> None:
        resp = _make_completion("")
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.CLEAR

    def test_none_content_returns_clear(self) -> None:
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        result = self.agent._parse_response(resp)
        assert result.signal == AmbientSignal.CLEAR


# ---------------------------------------------------------------------------
# AmbientAgent.process()
# ---------------------------------------------------------------------------


class TestAmbientProcess:
    @pytest.mark.asyncio
    async def test_no_frame_returns_clear(self) -> None:
        state = PromptState()
        agent = AmbientAgent(state)
        await agent.start()
        result = await agent.process({"frame_base64": None})
        assert result.signal == AmbientSignal.CLEAR
        await agent.stop()

    @pytest.mark.asyncio
    async def test_no_client_returns_clear(self) -> None:
        state = PromptState()
        agent = AmbientAgent(state)
        # Don't call start() — _client is None
        result = await agent.process({"frame_base64": "abc123"})
        assert result.signal == AmbientSignal.CLEAR

    @pytest.mark.asyncio
    async def test_process_calls_nim_and_parses(self) -> None:
        state = PromptState()
        agent = AmbientAgent(state)

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            json.dumps({"signal": "WARNING", "message": "Obstacle ahead", "reasoning": "x"})
        )
        agent._client = mock_client

        result = await agent.process({"frame_base64": "dGVzdA=="})
        assert result.signal == AmbientSignal.WARNING
        assert result.message == "Obstacle ahead"

        # Verify the NIM call was made with correct structure.
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"]  # model name is set from config
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_inspect_uses_override_prompt(self) -> None:
        state = PromptState()
        agent = AmbientAgent(state)

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            json.dumps({"signal": "CLEAR", "message": "The sign says Exit", "reasoning": "x"})
        )
        agent._client = mock_client

        result = await agent.inspect("dGVzdA==", "Read all visible text")
        assert result.message == "The sign says Exit"

        # Verify the system prompt uses the inspect prompt, not the normal one.
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "Read all visible text" in system_content
        assert "ACTIVE GOAL" not in system_content

    @pytest.mark.asyncio
    async def test_nim_exception_returns_clear(self) -> None:
        state = PromptState()
        agent = AmbientAgent(state)

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("NIM down")
        agent._client = mock_client

        result = await agent.process({"frame_base64": "dGVzdA=="})
        assert result.signal == AmbientSignal.CLEAR


# ---------------------------------------------------------------------------
# Ambient loop (on Orchestrator)
# ---------------------------------------------------------------------------


class TestAmbientLoop:
    @pytest.mark.asyncio
    async def test_loop_processes_frames(self) -> None:
        state = PromptState()
        buf = FrameBuffer()
        buf.push(b"frame_data")

        mock_agent = AsyncMock(spec=AmbientAgent)
        mock_agent.process.return_value = AmbientResponse(
            signal=AmbientSignal.WARNING, message="Watch out!"
        )

        speech_calls = []
        async def on_speech(priority, text):
            speech_calls.append((priority, text))

        orch = Orchestrator(
            state, ambient_agent=mock_agent, frame_buffer=buf,
            on_speech=on_speech,
        )

        # Run the loop long enough for at least one iteration (4s min interval).
        loop_task = asyncio.create_task(orch.run_ambient_loop())
        await asyncio.sleep(0.5)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        # The agent should have been called at least once.
        assert mock_agent.process.call_count >= 1
        # Speech callback should have been invoked.
        assert len(speech_calls) >= 1

    @pytest.mark.asyncio
    async def test_loop_backs_off_when_empty(self) -> None:
        state = PromptState()
        buf = FrameBuffer()  # empty

        mock_agent = AsyncMock(spec=AmbientAgent)

        orch = Orchestrator(state, ambient_agent=mock_agent, frame_buffer=buf)

        loop_task = asyncio.create_task(orch.run_ambient_loop())
        await asyncio.sleep(0.15)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        # Agent should NOT have been called — no frames.
        mock_agent.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_survives_agent_exception(self) -> None:
        state = PromptState()
        buf = FrameBuffer()

        call_count = 0

        async def flaky_process(input_data: dict) -> AmbientResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return AmbientResponse(signal=AmbientSignal.CLEAR)

        mock_agent = AsyncMock(spec=AmbientAgent)
        mock_agent.process.side_effect = flaky_process

        orch = Orchestrator(state, ambient_agent=mock_agent, frame_buffer=buf)

        # Push distinct frames so the loop processes each one (it skips duplicates).
        buf.push(b"frame_1")
        loop_task = asyncio.create_task(orch.run_ambient_loop())
        # The loop has a 4s minimum interval; wait long enough for two iterations.
        await asyncio.sleep(0.5)
        buf.push(b"frame_2")  # new frame after the first error
        await asyncio.sleep(5.0)
        loop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop_task

        # Should have recovered from the first error and processed the second frame.
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_loop_returns_without_agent(self) -> None:
        state = PromptState()
        orch = Orchestrator(state)  # no agent, no buffer
        # Should return immediately, not loop forever.
        await orch.run_ambient_loop()

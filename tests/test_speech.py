"""Tests for the speech modules: TTS (Kokoro) and ASR (Parakeet-EOU)."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spark_sight.speech.asr import (
    ASRClient,
    _i16le_to_f32le,
    asr_loop,
)
from spark_sight.speech.tts import TTSClient, tts_loop
from spark_sight.bridge.orchestrator import Orchestrator, SpeechPriority
from spark_sight.bridge.prompt_state import PromptState


# ---------------------------------------------------------------------------
# _i16le_to_f32le
# ---------------------------------------------------------------------------


class TestI16ToF32:
    def test_silence_converts_to_zeros(self) -> None:
        pcm_i16 = b"\x00" * 4096  # 2048 zero samples
        pcm_f32 = _i16le_to_f32le(pcm_i16)
        # f32le: 4 bytes per sample, so 2048 * 4 = 8192 bytes
        assert len(pcm_f32) == 8192
        samples = struct.unpack(f"<{len(pcm_f32) // 4}f", pcm_f32)
        assert all(s == 0.0 for s in samples)

    def test_max_positive_normalizes_near_one(self) -> None:
        # 16-bit max positive is 32767
        pcm_i16 = struct.pack("<1h", 32767)
        pcm_f32 = _i16le_to_f32le(pcm_i16)
        (sample,) = struct.unpack("<1f", pcm_f32)
        assert sample == pytest.approx(32767 / 32768.0)

    def test_negative_normalizes(self) -> None:
        pcm_i16 = struct.pack("<1h", -16384)
        pcm_f32 = _i16le_to_f32le(pcm_i16)
        (sample,) = struct.unpack("<1f", pcm_f32)
        assert sample == pytest.approx(-0.5)

    def test_empty_input(self) -> None:
        assert _i16le_to_f32le(b"") == b""


# ---------------------------------------------------------------------------
# TTSClient
# ---------------------------------------------------------------------------


class TestTTSClient:
    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self) -> None:
        client = TTSClient()
        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.read.return_value = b"RIFF\x00\x00\x00\x00WAVEfake-audio"
        mock_openai.audio.speech.create.return_value = mock_response
        client._client = mock_openai

        result = await client.synthesize("Hello world")
        assert result is not None
        assert result.startswith(b"RIFF")

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self) -> None:
        client = TTSClient()
        client._client = AsyncMock()
        result = await client.synthesize("")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_no_client(self) -> None:
        client = TTSClient()
        result = await client.synthesize("Hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_synthesize_handles_exception(self) -> None:
        client = TTSClient()
        mock_openai = AsyncMock()
        mock_openai.audio.speech.create.side_effect = RuntimeError("TTS down")
        client._client = mock_openai

        result = await client.synthesize("Hello")
        assert result is None


# ---------------------------------------------------------------------------
# ASRClient
# ---------------------------------------------------------------------------


class TestASRClient:
    def test_not_available_before_start(self) -> None:
        client = ASRClient()
        assert not client.available

    @pytest.mark.asyncio
    async def test_send_audio_when_not_connected(self) -> None:
        client = ASRClient()
        # Should not raise.
        await client.send_audio(b"\x00" * 4096)

    @pytest.mark.asyncio
    async def test_send_audio_converts_and_sends(self) -> None:
        client = ASRClient()
        mock_ws = AsyncMock()
        client._ws = mock_ws
        client._connected = True

        pcm_i16 = struct.pack("<4h", 0, 16384, -16384, 32767)
        await client.send_audio(pcm_i16)

        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        # 4 samples * 4 bytes = 16 bytes of f32le
        assert len(sent_data) == 16


# ---------------------------------------------------------------------------
# tts_loop
# ---------------------------------------------------------------------------


class TestTTSLoop:
    @pytest.mark.asyncio
    async def test_loop_drains_speech_and_enqueues_wav(self) -> None:
        state = PromptState()
        orch = Orchestrator(state)
        await orch._enqueue_speech(SpeechPriority.PLANNING, "Test message")

        mock_tts = AsyncMock(spec=TTSClient)
        mock_tts.synthesize.return_value = b"RIFF-wav-data"

        tts_queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = asyncio.create_task(tts_loop(orch, mock_tts, tts_queue))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_tts.synthesize.assert_called_once_with("Test message")
        assert tts_queue.qsize() == 1
        wav = await tts_queue.get()
        assert wav == b"RIFF-wav-data"


# ---------------------------------------------------------------------------
# asr_loop
# ---------------------------------------------------------------------------


class TestASRLoop:
    @pytest.mark.asyncio
    async def test_loop_forwards_audio_to_client(self) -> None:
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        async def noop_transcript(text: str) -> None:
            pass

        mock_asr = ASRClient()
        mock_asr._connected = True
        mock_asr._ws = AsyncMock()
        mock_asr._on_transcript = noop_transcript

        pcm = struct.pack("<1024h", *([100] * 1024))
        await audio_queue.put(pcm)

        task = asyncio.create_task(asr_loop(audio_queue, mock_asr, noop_transcript))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Audio should have been forwarded to the WebSocket.
        mock_asr._ws.send.assert_called_once()


# ---------------------------------------------------------------------------
# Orchestrator.handle_transcript
# ---------------------------------------------------------------------------


class TestOrchestratorTranscript:
    @pytest.mark.asyncio
    async def test_handle_transcript_routes_to_planning(self) -> None:
        from spark_sight.bridge.models import PlanningAction, PlanningResponse

        state = PromptState()
        mock_planning = AsyncMock()
        mock_planning.process.return_value = PlanningResponse(
            action=PlanningAction.ANSWER,
            message="The nearest subway is 72nd St.",
        )
        orch = Orchestrator(state, planning_agent=mock_planning)

        await orch.handle_transcript("Where is the subway?")

        mock_planning.process.assert_called_once_with(
            {"transcript": "Where is the subway?"}
        )
        _, text = await orch.next_speech()
        assert "72nd" in text

    @pytest.mark.asyncio
    async def test_handle_empty_transcript_is_noop(self) -> None:
        state = PromptState()
        mock_planning = AsyncMock()
        orch = Orchestrator(state, planning_agent=mock_planning)

        await orch.handle_transcript("")
        await orch.handle_transcript("   ")

        mock_planning.process.assert_not_called()

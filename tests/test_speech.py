"""Tests for the speech modules: TTS (Magpie) and ASR (Parakeet)."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from spark_sight.speech.asr import (
    ASRClient,
    _rms_energy,
    pcm_to_wav,
    asr_loop,
    _ENERGY_THRESHOLD,
    _SILENCE_CHUNKS_TO_END,
    _MIN_SPEECH_CHUNKS,
)
from spark_sight.speech.tts import TTSClient, tts_loop
from spark_sight.bridge.orchestrator import Orchestrator, SpeechPriority
from spark_sight.bridge.prompt_state import PromptState


# ---------------------------------------------------------------------------
# pcm_to_wav
# ---------------------------------------------------------------------------


class TestPcmToWav:
    def test_produces_valid_wav_header(self) -> None:
        pcm = b"\x00" * 8192
        wav = pcm_to_wav(pcm)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_wav_contains_pcm_data(self) -> None:
        pcm = b"\x01\x02" * 1000
        wav = pcm_to_wav(pcm)
        assert len(wav) > len(pcm)  # WAV has header overhead

    def test_empty_pcm(self) -> None:
        wav = pcm_to_wav(b"")
        assert wav[:4] == b"RIFF"


# ---------------------------------------------------------------------------
# _rms_energy
# ---------------------------------------------------------------------------


class TestRmsEnergy:
    def test_silence_is_zero(self) -> None:
        pcm = b"\x00" * 4096
        assert _rms_energy(pcm) == 0.0

    def test_loud_signal(self) -> None:
        # 2048 samples of constant 10000
        pcm = struct.pack("<2048h", *([10000] * 2048))
        assert _rms_energy(pcm) == pytest.approx(10000.0)

    def test_quiet_signal(self) -> None:
        pcm = struct.pack("<2048h", *([50] * 2048))
        assert _rms_energy(pcm) == pytest.approx(50.0)

    def test_empty_returns_zero(self) -> None:
        assert _rms_energy(b"") == 0.0


# ---------------------------------------------------------------------------
# TTSClient
# ---------------------------------------------------------------------------


class TestTTSClient:
    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self) -> None:
        client = TTSClient()
        # Mock the OpenAI client.
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
        mock_openai.audio.speech.create.side_effect = RuntimeError("NIM down")
        client._client = mock_openai

        result = await client.synthesize("Hello")
        assert result is None


# ---------------------------------------------------------------------------
# ASRClient
# ---------------------------------------------------------------------------


class TestASRClient:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self) -> None:
        client = ASRClient()
        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_openai.audio.transcriptions.create.return_value = mock_response
        client._client = mock_openai

        result = await client.transcribe(pcm_to_wav(b"\x00" * 4096))
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_empty_wav(self) -> None:
        client = ASRClient()
        client._client = AsyncMock()
        result = await client.transcribe(b"")
        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_no_client(self) -> None:
        client = ASRClient()
        result = await client.transcribe(b"some-wav-data")
        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_handles_exception(self) -> None:
        client = ASRClient()
        mock_openai = AsyncMock()
        mock_openai.audio.transcriptions.create.side_effect = RuntimeError("NIM down")
        client._client = mock_openai

        result = await client.transcribe(pcm_to_wav(b"\x00" * 4096))
        assert result == ""


# ---------------------------------------------------------------------------
# tts_loop
# ---------------------------------------------------------------------------


class TestTTSLoop:
    @pytest.mark.asyncio
    async def test_loop_drains_speech_and_enqueues_wav(self) -> None:
        state = PromptState()
        orch = Orchestrator(state)
        # Enqueue a speech item.
        await orch._enqueue_speech(SpeechPriority.PLANNING, "Test message")

        mock_tts = AsyncMock(spec=TTSClient)
        mock_tts.synthesize.return_value = b"RIFF-wav-data"

        tts_queue: asyncio.Queue[bytes] = asyncio.Queue()

        task = asyncio.create_task(tts_loop(orch, mock_tts, tts_queue))
        # Wait for the TTS to process.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_tts.synthesize.assert_called_once_with("Test message")
        assert tts_queue.qsize() == 1
        wav = await tts_queue.get()
        assert wav == b"RIFF-wav-data"


# ---------------------------------------------------------------------------
# asr_loop (VAD integration)
# ---------------------------------------------------------------------------


class TestASRLoop:
    @pytest.mark.asyncio
    async def test_loop_detects_speech_and_transcribes(self) -> None:
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcripts: list[str] = []

        async def on_transcript(text: str) -> None:
            transcripts.append(text)

        mock_asr = AsyncMock(spec=ASRClient)
        mock_response = MagicMock()
        mock_response.text = "Take me to the subway"
        mock_asr.transcribe.return_value = "Take me to the subway"

        # Simulate speech: loud chunks followed by silence.
        loud_chunk = struct.pack("<4096h", *([int(_ENERGY_THRESHOLD * 2)] * 4096))
        silent_chunk = struct.pack("<4096h", *([10] * 4096))

        # Push enough loud chunks to exceed MIN_SPEECH_CHUNKS.
        for _ in range(_MIN_SPEECH_CHUNKS + 1):
            await audio_queue.put(loud_chunk)
        # Push silence to trigger end-of-speech.
        for _ in range(_SILENCE_CHUNKS_TO_END + 1):
            await audio_queue.put(silent_chunk)

        task = asyncio.create_task(asr_loop(audio_queue, mock_asr, on_transcript))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert mock_asr.transcribe.call_count >= 1
        assert "Take me to the subway" in transcripts

    @pytest.mark.asyncio
    async def test_loop_ignores_pure_silence(self) -> None:
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        transcripts: list[str] = []

        async def on_transcript(text: str) -> None:
            transcripts.append(text)

        mock_asr = AsyncMock(spec=ASRClient)

        silent_chunk = struct.pack("<4096h", *([10] * 4096))
        for _ in range(20):
            await audio_queue.put(silent_chunk)

        task = asyncio.create_task(asr_loop(audio_queue, mock_asr, on_transcript))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_asr.transcribe.assert_not_called()
        assert transcripts == []


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

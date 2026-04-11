# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Spark Sight is a real-time accessibility assistant for visually impaired New Yorkers. It runs a two-agent AI system on an NVIDIA GB10 — an **Ambient Agent** (Cosmos Reason2-8B VLM) that continuously monitors camera frames for hazards and navigation, and a **Planning Agent** (Nemotron Super) that interprets voice commands and queries NYC accessibility data. An iPhone connects via WebSocket to stream camera frames and mic audio; the server responds with TTS audio and status updates.

## Build & Development Commands

Python 3.13, managed with `uv`. All commands run from the project root.

```sh
# Install dependencies (creates .venv automatically)
uv sync

# Copy and edit .env with your GB10 IP (required for NIM endpoints)
cp .env.example .env

# Run the server
uv run python -m spark_sight.main --debug

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_bridge.py

# Run a single test class or method
uv run pytest tests/test_bridge.py::TestPromptState::test_set_goal_transitions_to_goal_mode

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Launch NIM containers on GB10 (requires NGC_API_KEY, NIM_LLM_MODEL_FREE_IMAGE env vars)
bash scripts/cos_nemo_docker.sh
```

## Architecture

### Package Layout (`src/spark_sight/`)

- **`agents/`** — AI agent implementations. Each agent subclasses `BaseAgent` (in `agents/base.py`) which defines the lifecycle: `start()` → `process()` → `stop()`.
  - `ambient/agent.py` — **AmbientAgent**: evaluates each camera frame against a dynamic goal prompt via the Cosmos Reason2 NIM endpoint (OpenAI-compatible API at `localhost:8000`). Returns structured JSON with a signal (`CLEAR`, `WARNING`, `PROGRESS`, `CORRECTION`, `GOAL_REACHED`, `FAILURE`) and an optional spoken message. Strips markdown fences from VLM output before parsing.
  - `planning/agent.py` — **PlanningAgent**: parses user voice transcripts or FAILURE replan triggers via the Nemotron NIM endpoint (`localhost:8005`). Returns a `PlanningResponse` with an action (`set_goal`, `inspect`, `answer`, `reset`, `replan`). Currently a stub — NIM inference is placeholder-commented.

- **`bridge/`** — The coordination layer between agents. No agent talks to another directly; everything flows through here.
  - `models.py` — All Pydantic data models and enums: `AgentMode`, `AmbientSignal`, `PlanningAction`, `PromptStateData`, `AmbientResponse`, `PlanningResponse`. This is the contract between all components.
  - `prompt_state.py` — **PromptState**: thread-safe (threading.Lock) shared mutable state. The Planning Agent writes goals/context, the Ambient Agent reads the `compiled_prompt` each frame. Mutations are atomic. `compiled_prompt` is a computed property that concatenates `base_goal + active_goal + nyc_context`.
  - `orchestrator.py` — **Orchestrator**: the central router. Dispatches Ambient signals (e.g. `GOAL_REACHED` → reset state, `FAILURE` → trigger replan on Planning Agent). Dispatches Planning actions (e.g. `set_goal` → update PromptState, `inspect` → grab latest frame from buffer → one-shot query to Ambient Agent). Manages a speech priority queue (`WARNING` preempts all). Runs the continuous `run_ambient_loop()` that pulls frames from the buffer.

- **`server/`** — FastAPI WebSocket bridge between the iPhone and agents.
  - `protocol.py` — Binary WebSocket protocol: 1-byte type tag prefix. iPhone sends `0x01` (JPEG frame) and `0x02` (PCM audio); server sends `0x03` (TTS WAV) and `0x04` (JSON status).
  - `frame_buffer.py` — **FrameBuffer**: thread-safe ring buffer (`deque(maxlen=30)`) storing raw JPEG frames. The Ambient Agent reads the latest frame via `latest_base64()`.
  - `app.py` — `create_app()` factory returns the FastAPI app. Routes: `/` (client HTML), `/health`, `/debug` (when debug=True), `/ws` (unified WebSocket). Queues for TTS, status, and audio are created here and exposed on `app.state`.

- **`speech/`** — Speech synthesis and recognition clients for Magpie TTS and Parakeet ASR.
  - `tts.py` — **TTSClient**: wraps OpenAI-compatible `/v1/audio/speech` endpoint. `tts_loop()` drains the Orchestrator speech queue → Magpie NIM → WAV bytes → `tts_queue` → iPhone.
  - `asr.py` — **ASRClient**: wraps Whisper-compatible `/v1/audio/transcriptions`. `asr_loop()` buffers PCM from `audio_queue`, runs energy-based VAD, packages WAV, sends to Parakeet, routes transcript to `Orchestrator.handle_transcript()` → Planning Agent.

- **`config.py`** — Settings module. Reads 4 NIM endpoints from `.env` via `python-dotenv`. Dataclass-based with defaults for all fields.

- **`main.py`** — Entry point. Wires PromptState, FrameBuffer, both agents, TTSClient, ASRClient, and the Orchestrator. Starts 3 background loops on startup: ambient frame processing, TTS synthesis, ASR transcription.

- **`static/client.html`** — iPhone web client served at `/`.

### Data Flow

1. iPhone camera → WebSocket (`/ws`) → `FrameBuffer`
2. `Orchestrator.run_ambient_loop()` → latest frame → `AmbientAgent.process()` → `handle_ambient_response()` → speech queue / state transitions
3. iPhone mic → WebSocket → `audio_queue` → `asr_loop()` (VAD + Parakeet) → `Orchestrator.handle_transcript()` → `PlanningAgent.process()` → `handle_planning_response()` → PromptState / speech queue
4. Speech queue → `tts_loop()` → Magpie TTS NIM → WAV → `tts_queue` → WebSocket → iPhone speaker

### Key Design Patterns

- **Agents never communicate directly.** All inter-agent communication flows through the Orchestrator and PromptState.
- **PromptState is the only shared mutable state.** Planning writes, Ambient reads, Orchestrator mediates.
- **All 4 models use OpenAI-compatible APIs** (`AsyncOpenAI` client) — chat completions for LLM/VLM, audio endpoints for ASR/TTS. No API keys needed for local NIM.
- **Graceful degradation**: if NIM inference fails or returns unparseable output, the Ambient Agent defaults to `CLEAR` (silent). The Planning Agent falls back to raw text or error messages. TTS/ASR failures are logged and skipped.
- **Configuration via `.env`**: all 4 NIM endpoint URLs and model names are configurable. Copy `.env.example` and set your GB10 IP.

### NIM Container Configuration

`scripts/cos_nemo_docker.sh` launches two Docker containers sequentially (to avoid vLLM memory profiling race conditions):
1. Nemotron-3-Nano-30B on port **8005** (`NIM_KVCACHE_PERCENT=0.35`, `NIM_MAX_MODEL_LEN=32768`)
2. Cosmos-Reason2-8B on port **8000** (`NIM_KVCACHE_PERCENT=0.35`, `NIM_MAX_MODEL_LEN=16384`, forced bf16 profile to avoid FP8 kernel bugs on SM121/GB10)

### Testing Conventions

- Tests use `pytest` + `pytest-asyncio` for async tests and `unittest.mock.AsyncMock` for mocking NIM clients and agents.
- `httpx` is a dev dependency used by FastAPI's `TestClient` (via Starlette).
- Test files mirror source structure: `test_bridge.py` (PromptState + Orchestrator), `test_ambient.py` (AmbientAgent + ambient loop), `test_planning.py` (PlanningAgent response parsing), `test_speech.py` (TTS/ASR clients + loops + VAD), `test_config.py` (settings), `test_server.py` (HTTP + WebSocket), `test_frame_buffer.py`, `test_protocol.py`.

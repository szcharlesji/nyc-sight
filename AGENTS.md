# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app (all flags optional)
uv run main.py
uv run main.py --no-mic --no-tts       # text-only mode for dev
uv run main.py --video demo.mp4        # use a video file instead of webcam

# Add dependencies
uv add <package>

# Never use pip, conda, or venv — always uv
```

## Architecture

Three concurrent loops run in a background `asyncio` event loop (inside a daemon `threading.Thread` started by `App._start_backend`). Gradio runs in the main thread.

```
CameraLoop (threading.Thread)
  └─ pushes Frame → FrameBuffer (ring buffer, maxlen=30)

YOLODetector (async task)
  └─ reads FrameBuffer.latest() every 30ms
  └─ proximity alert → alert_queue
  └─ unusual object → vlm_trigger_queue

ASREngine (async task, runs VAD loop in executor)
  └─ sounddevice → Silero VAD → faster-whisper
  └─ transcribed text → voice_queue (via call_soon_threadsafe)

Orchestrator (async task)
  └─ alert_queue  → PiperTTS → audio file → UI callback
  └─ voice_queue  → VLMClient.describe() → PiperTTS → UI callback
  └─ vlm_trigger_queue → same as voice_queue path
```

**Thread-safety rule**: anything putting items into an `asyncio.Queue` from outside the backend event loop (e.g., from Gradio callbacks or the camera thread) must use `loop.call_soon_threadsafe(queue.put_nowait, item)`. The camera thread uses `FrameBuffer.push()` which already does this correctly via `call_soon_threadsafe`.

## Key design decisions

- **VLM is Gemini API** (`gemma-4-31b-it`) via Google's OpenAI-compatible endpoint at `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`. API key is read from `.env` (`GEMINI_API_KEY`). The `.env` is loaded manually in `src/vlm.py` without python-dotenv.
- **Streaming is required**: `VLMClient` always uses `stream=True` + SSE to prevent TCP connections being dropped during long generation silences.
- **Two VLM entry points**: `describe()` (used by the Orchestrator, returns a complete string) and `describe_stream()` (used by the Gradio ChatInterface `chat_fn`, async generator that yields cumulative strings). `describe_stream` creates its own `httpx.AsyncClient` per call to avoid cross-event-loop issues with Gradio's async context.
- **YOLO alerts fire instantly** without VLM — they go straight to TTS. VLM is only triggered by voice commands, YOLO escalation for unusual objects, or the ambient timer.
- **NYCData is a stub** (`src/nyc_data.py`) — it always returns an empty context string. The full spatial SQLite implementation is not yet built.

## Gradio UI

The Gradio `gr.Blocks` app has:
- A live webcam feed (polled every 0.5s via `gr.Timer`)
- Status and alert textboxes updated by `App._ui_update` (called from the Orchestrator via `ui_callback`)
- A `gr.ChatInterface` for text queries — its `chat_fn` calls `vlm.describe_stream()` directly, bypassing the queue system

The backend event loop reference is stored as `App._loop` after `_start_backend` creates it, making it available for `call_soon_threadsafe` calls from Gradio handlers.

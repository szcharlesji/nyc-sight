# Ambient Agent — Cosmos Reason2-8B

## Role
The system's **eyes**. Runs a continuous frame-processing loop evaluating each camera frame against a dynamic goal prompt. Speaks only when something is actionable.

## Model
- **Cosmos Reason2-8B** — NVIDIA's physical AI reasoning VLM
- Post-trained on EgoExo4D, PerceptionTest, CLEVRER for spatial understanding
- Outputs 2D/3D point localization, bounding boxes with reasoning
- Local NIM endpoint: `http://localhost:8000/v1`

## Input
Each `process()` call receives:
```json
{
  "frame_base64": "<base64-encoded JPEG from camera>"
}
```
The agent also reads the shared `PromptState` every frame to get the compiled prompt (base goal + active goal + NYC context).

## Output
Returns an `AmbientResponse`:
```json
{
  "signal": "CLEAR | WARNING | PROGRESS | CORRECTION | GOAL_REACHED | FAILURE",
  "message": "Natural language to speak (empty for CLEAR)",
  "reasoning": "Internal chain-of-thought (logged, not spoken)"
}
```

## Response Protocol
**Patrol Mode** (no active goal):
- `CLEAR` — nothing to report, stay silent
- `WARNING` — danger or useful info detected, speak to user

**Goal Mode** (active goal set by Planning Agent):
- `CLEAR` — on track, no hazards, stay silent
- `WARNING` — danger detected, speak warning (overrides goal tracking)
- `PROGRESS` — meaningful progress toward goal, speak update
- `CORRECTION` — user veering off course, speak correction
- `GOAL_REACHED` — goal achieved, speak confirmation, revert to Patrol
- `FAILURE` — goal cannot be achieved, speak status, trigger replan

Safety warnings (`WARNING`) always fire regardless of mode.

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `nim_base_url` | from `COSMOS_NIM_URL` env | Cosmos Reason2 NIM endpoint |
| `model` | from `COSMOS_MODEL` env (`nvidia/cosmos-reason2-8b`) | Model identifier |

## Status

### Built
- [x] Agent class implementing `BaseAgent` interface
- [x] `start()` / `stop()` lifecycle with async OpenAI client
- [x] `process()` method with NIM inference via OpenAI-compatible API
- [x] Live NIM inference against Cosmos Reason2-8B endpoint (configured via `.env`)
- [x] `_parse_response()` — parses structured JSON, strips markdown fences, graceful fallback to CLEAR
- [x] Integration with shared `PromptState` (reads compiled prompt each frame)
- [x] `FrameBuffer` ring buffer — thread-safe, stores last 30 JPEG frames with timestamps
- [x] `FrameBuffer.latest_base64()` — returns frame in the format `process()` expects
- [x] iPhone → WebSocket → FrameBuffer pipeline (frames arrive via `server/app.py`)
- [x] Orchestrator status callbacks — ambient signals push to iPhone HUD in real time
- [x] Continuous loop runner — `Orchestrator.run_ambient_loop()` pulls frames at ~2-4 FPS
- [x] One-shot inspect mode — `inspect()` method handles visual queries from Planning Agent

### TODO
- [x] ~~Wire NIM inference~~ — live against Cosmos Reason2-8B
- [x] ~~Response parsing~~ — `_parse_response()` with JSON + markdown fence handling
- [x] ~~Continuous loop runner~~ — `Orchestrator.run_ambient_loop()`
- [x] ~~One-shot inspect mode~~ — `AmbientAgent.inspect()`
- [ ] Frame preprocessing — resize/encode frames for optimal VLM input
- [ ] Safety classifier — ensure WARNING signals are never suppressed
- [ ] Performance tuning — optimize inference latency on GB10

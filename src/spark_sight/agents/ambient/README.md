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
| `nim_base_url` | `http://localhost:8000/v1` | Cosmos Reason2 NIM endpoint |
| `model` | `nvidia/cosmos-reason2-8b` | Model identifier |

## Status

### Built
- [x] Agent class implementing `BaseAgent` interface
- [x] `start()` / `stop()` lifecycle with async OpenAI client
- [x] `process()` method with input parsing and stub response
- [x] NIM inference call structure (commented placeholder)
- [x] Integration with shared `PromptState` (reads compiled prompt)

### TODO
- [ ] Wire NIM inference — uncomment and test against live Cosmos Reason2 container
- [ ] Response parsing — parse model output into `AmbientResponse` structured format
- [ ] Frame preprocessing — resize/encode frames for optimal VLM input
- [ ] Continuous loop runner — async loop that pulls frames from ring buffer at ~2-4 FPS
- [ ] Ring buffer integration — connect to iPhone camera frame buffer
- [ ] One-shot inspect mode — handle `inspect` queries from Planning Agent
- [ ] Safety classifier — ensure WARNING signals are never suppressed
- [ ] Performance tuning — optimize inference latency on GB10

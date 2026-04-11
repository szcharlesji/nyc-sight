# Planning Agent — Nemotron-3-Nano-30B

## Role
The system's **brain**. Sits idle until activated by user voice or an Ambient Agent `FAILURE` signal. Interprets natural language, queries NYC Open Data, reasons about routes, and programs the Ambient Agent's active goal.

## Model
- **Nemotron-3-Nano-30B** — NVIDIA's efficient agentic LLM
- Optimized for intent parsing, structured output, and tool calling
- No vision capabilities — all visual tasks delegated to Ambient Agent
- NIM endpoint configured via `NEMOTRON_NIM_URL` env var (default: `http://localhost:8005/v1`)

## Input
Each `process()` call receives one of:
```json
{"transcript": "Take me to the nearest subway station"}
```
```json
{"failure_reason": "Path blocked by construction at 71st & Broadway"}
```
The agent also reads the shared `PromptState` for current mode/goal context.

## Output
Returns a `PlanningResponse`:
```json
{
  "action": "set_goal | inspect | answer | reset | replan",
  "message": "Text to speak to the user",
  "goal": "New goal text (set_goal/replan only)",
  "nyc_context": "NYC data context string (set_goal/replan only)",
  "inspect_prompt": "One-shot prompt for Ambient Agent (inspect only)",
  "metadata": {}
}
```

## Actions
| Action | Effect |
|--------|--------|
| `set_goal` | Inject new goal + NYC context into Ambient Agent's prompt |
| `inspect` | Grab latest frame, send to Ambient Agent with a one-shot question |
| `answer` | Speak directly to user (no vision needed) |
| `reset` | Clear Ambient Agent's goal, revert to Patrol Mode |
| `replan` | Formulate new goal after FAILURE signal |

## Tools
| Tool | Description |
|------|-------------|
| `nyc_accessibility_lookup` | Spatial SQL query on local SQLite — scaffolding, elevators, APS, 311, ramps |
| `set_ambient_goal` | Write new goal into Ambient Agent's prompt state |
| `reset_ambient_goal` | Clear goal, revert to Patrol |
| `web_search` | Web search for info not in NYC DB (WiFi only) |

## Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `nim_base_url` | from `NEMOTRON_NIM_URL` env | Nemotron NIM endpoint |
| `model` | from `NEMOTRON_MODEL` env (`nemotron-nano`) | Model identifier |

## Status

### Built
- [x] Agent class implementing `BaseAgent` interface
- [x] `start()` / `stop()` lifecycle with async OpenAI client
- [x] `process()` method with transcript/failure input parsing
- [x] System prompt with structured JSON output schema
- [x] Live NIM inference via OpenAI-compatible API (Nemotron-3-Nano-30B)
- [x] `_parse_response()` — parses model JSON into `PlanningResponse`, strips markdown fences
- [x] Integration with shared `PromptState` (reads snapshot for context)
- [x] Graceful degradation: falls back to stub if NIM client is None, to raw text if JSON parse fails
- [x] Mic audio → WebSocket → `audio_queue` pipeline (PCM chunks arrive via `server/app.py`)
- [x] Orchestrator status callbacks — planning responses push to iPhone HUD

### TODO
- [x] ~~Wire NIM inference~~ — live against Nemotron-3-Nano-30B
- [x] ~~Response parsing~~ — `_parse_response()` with JSON + markdown fence handling
- [ ] Tool execution — implement `nyc_accessibility_lookup` with SQLite spatial queries
- [ ] Tool execution — implement `set_ambient_goal` / `reset_ambient_goal` via Orchestrator
- [ ] Tool execution — implement `web_search` (optional, WiFi-dependent)
- [ ] NYC Open Data SQLite — schema design, data import pipeline, spatial indexing
- [ ] GPS coordinate handling — receive and pass user location to tools
- [ ] Parakeet ASR bridge — drain `audio_queue`, stream PCM to Parakeet NIM gRPC, emit transcripts
- [ ] Conversation memory — maintain short-term context across multiple interactions
- [ ] Replan logic — sophisticated replanning when FAILURE reasons require route changes

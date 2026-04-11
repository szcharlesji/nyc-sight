# Planning Agent — Nemotron Super 120B

## Role
The system's **brain**. Sits idle until activated by user voice or an Ambient Agent `FAILURE` signal. Interprets natural language, queries NYC Open Data, reasons about routes, and programs the Ambient Agent's active goal.

## Model
- **Nemotron Super 120B** — NVIDIA's open-source agentic LLM
- Optimized for intent parsing, structured output, and tool calling
- No vision capabilities — all visual tasks delegated to Ambient Agent
- Local NIM endpoint: `http://localhost:8001/v1`

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
| `nim_base_url` | `http://localhost:8001/v1` | Nemotron Super NIM endpoint |
| `model` | `nvidia/nemotron-super-120b` | Model identifier |

## Status

### Built
- [x] Agent class implementing `BaseAgent` interface
- [x] `start()` / `stop()` lifecycle with async OpenAI client
- [x] `process()` method with transcript/failure input parsing
- [x] System prompt with tool definitions and output schema
- [x] NIM inference call structure (commented placeholder)
- [x] Integration with shared `PromptState` (reads snapshot for context)
- [x] Stub responses for testing without NIM

### TODO
- [ ] Wire NIM inference — uncomment and test against live Nemotron Super container
- [ ] Response parsing — parse model JSON output into `PlanningResponse`
- [ ] Tool execution — implement `nyc_accessibility_lookup` with SQLite spatial queries
- [ ] Tool execution — implement `set_ambient_goal` / `reset_ambient_goal` via Orchestrator
- [ ] Tool execution — implement `web_search` (optional, WiFi-dependent)
- [ ] NYC Open Data SQLite — schema design, data import pipeline, spatial indexing
- [ ] GPS coordinate handling — receive and pass user location to tools
- [ ] Parakeet ASR integration — receive streaming transcripts as triggers
- [ ] Conversation memory — maintain short-term context across multiple interactions
- [ ] Replan logic — sophisticated replanning when FAILURE reasons require route changes

# Spark Sight

## Problem We Are Solving

800,000 New Yorkers live with a vision disability. They navigate a city where only 23% of subway stations are accessible, elevators break without warning, and over 8,000 scaffolding sheds turn sidewalks into obstacle courses.

No existing app combines real-time scene understanding with live city infrastructure data. Be My Eyes requires cloud connectivity and human volunteers. Seeing AI describes snapshots but doesn't track goals or warn about hazards proactively. Neither knows that the elevator at your station is broken or that there's scaffolding narrowing the sidewalk ahead.

**Spark Sight** is a fully local, private AI assistant that gives blind users real-time spatial awareness fused with live NYC data. It runs entirely on the NVIDIA GB10 — no cloud, no internet, no images ever leave the device.

Our two-agent architecture pairs **Cosmos Reason2** as a continuous ambient visual monitor with **Nemotron Super** as an intelligent planner. When you say "take me to the subway," the Planning Agent checks which elevators are working, finds active scaffolding on your route, and programs the Ambient Agent to guide you there — warning about obstacles and tracking your progress until you arrive.

Four NVIDIA models. All multilingual. One device. Zero compromise on privacy. Designed to run entirely on a single NVIDIA GB10 with 128GB unified memory.

|Component|Model|Role|
|---|---|---|
|Ambient Agent|Cosmos Reason2-8B|Continuous visual monitoring, spatial reasoning, goal tracking|
|Planning Agent|Nemotron Super 120B|Intent parsing, route planning, NYC data queries, goal management|
|Speech-to-Text|Parakeet 1.1B RNNT Multilingual|Streaming ASR with built-in VAD|
|Text-to-Speech|Magpie TTS Multilingual|Expressive multi-voice synthesis|

---

## How It Works

The user wears a chest-mounted iPhone in a MagSafe holder and headphones. They walk normally, speak naturally, and hear spoken responses.

The **Ambient Agent** runs continuously in the background — always watching through the camera, always ready to warn. It speaks only when something matters: an obstacle ahead, a course correction needed, or a goal reached. Most of the time, it is silent.

The **Planning Agent** wakes when the user speaks. It interprets what they want, searches NYC's accessibility data — subway stations, scaffolding permits, elevator outages, accessible pedestrian signals — and programs the Ambient Agent with a specific goal. Then it goes back to sleep.

The core innovation is this separation: the Ambient Agent is fast but narrow. The Planning Agent is slow but broad. Together they form a complete ambient intelligence.

---

## Ambient Agent: Cosmos Reason2-8B

**Role**: The system's eyes. Runs a continuous frame-processing loop. Every frame is evaluated against an active goal prompt. Most frames produce silence — the agent only speaks when there is something actionable.

**Why Cosmos**: Cosmos Reason2 is NVIDIA's physical AI reasoning VLM, post-trained on embodied reasoning datasets (EgoExo4D, PerceptionTest, CLEVRER) for spatial understanding. It natively outputs 2D/3D point localization and bounding box coordinates with reasoning explanations. It understands space, time, object permanence, and physical common sense — exactly what a navigation assistant needs.

**How it works**:

1. The iPhone camera feeds frames into a ring buffer (last 30 frames).
2. The Ambient Agent pulls the latest frame on each loop iteration (~2–4 FPS, governed by inference speed).
3. Each frame is evaluated against a **goal prompt** — a dynamic text block composed of three layers:
 - **Base goal** (always present): detect safety hazards, warn about close obstacles, read signs.
 - **Active goal** (set by the Planning Agent): "guide the user north on Broadway toward the 72nd St subway entrance on the right side."
 - **NYC context** (injected by the Planning Agent): "active scaffolding at 234 W 71st St narrowing the west sidewalk; accessible pedestrian signal at 72nd & Broadway."
4. The Ambient Agent evaluates the frame and responds with one of these signals:

### Ambient Agent Response Protocol

**In Patrol Mode** (no active goal — default state):

|Signal|Meaning|Action|
|---|---|---|
|`CLEAR`|Nothing to report|Stay silent. Continue loop.|
|`WARNING`|Potential danger or useful information detected|Speak to user via Magpie TTS. Continue loop.|

**In Goal Mode** (active goal set by Planning Agent):

|Signal|Meaning|Action|
|---|---|---|
|`CLEAR`|On track, no hazards|Stay silent. Continue loop.|
|`WARNING`|Danger detected (overrides goal tracking)|Speak warning to user. Continue loop. Goal persists.|
|`PROGRESS`|Meaningful progress toward goal|Speak update to user (e.g., "intersection ahead, subway entrance 50 feet on your right"). Continue loop.|
|`CORRECTION`|User is veering off course or needs to adjust|Speak correction to user. Continue loop. Goal persists.|
|`GOAL_REACHED`|Goal has been achieved|Speak confirmation to user. Send signal to Orchestrator. Revert to Patrol Mode.|
|`FAILURE`|Goal cannot be achieved (target lost, path blocked)|Speak status to user. Send signal + reason to Planning Agent for replanning. Revert to Patrol Mode.|

**Safety warnings always fire regardless of mode.** If a bicycle is approaching fast while the agent is tracking a navigation goal, the `WARNING` fires immediately. The goal is not interrupted — the agent reports both in the same or consecutive frames.

---

## Planning Agent: Nemotron Super 120B

**Role**: The system's brain. Sits idle until activated by a user voice command, or by a `FAILURE` signal from the Ambient Agent. Interprets natural language requests, queries NYC Open Data for spatial context, reasons about routes and conditions, and manages the Ambient Agent's active goal.

**Why Nemotron Super**: NVIDIA's open-source 120B LLM, optimized for agentic tasks — parsing intent, generating structured outputs, and making tool calls. It doesn't need vision capabilities. It only needs the textual context for planning; all real-time visual tasks are handled by the Ambient Agent.

**How it works**:

1. **Trigger**: user speaks → Parakeet ASR transcribes → transcript arrives at the Planning Agent. Alternatively, a `FAILURE` signal from the Ambient Agent triggers replanning.
2. **Context assembly**: the Planning Agent receives:
 - The user's transcribed text (or the Ambient Agent's failure reason).
 - The user's current GPS coordinates.
 - The Ambient Agent's current state (Patrol or Goal, and what the active goal is).
3. **Reasoning + tool use**: the Planning Agent uses tools to gather information (NYC data lookups, optional web search), then decides on an action.
4. **Output**: a structured JSON response specifying the action and any speech to deliver.

### Planning Agent Actions

|Action|What happens|Example|
|---|---|---|
|`set_goal`|Injects a new active goal + NYC context into the Ambient Agent's prompt. The Ambient Agent begins tracking on the next frame.|User: "Take me to the subway." → Planning Agent queries elevator status, finds scaffolding, sets goal: "Guide user north to 72nd St station north entrance."|
|`inspect`|Grabs the latest camera frame from the buffer and sends it to the Ambient Agent with a one-shot question. Answer is spoken, then the Ambient Agent returns to its previous mode.|User: "Read that sign." → Latest frame sent to Cosmos with prompt "Read all visible text." → Answer spoken.|
|`answer`|The Planning Agent answers directly without involving the Ambient Agent. For questions that don't require vision.|User: "Is there scaffolding nearby?" → SQL lookup → "Yes, active scaffolding at 234 Broadway, about 100 feet ahead."|
|`reset`|Clears the Ambient Agent's active goal. Reverts to Patrol Mode.|User: "Cancel." → Goal cleared.|
|`replan`|Triggered by a `FAILURE` from the Ambient Agent. The Planning Agent formulates a new goal based on the failure reason and current context.|Ambient Agent: "FAILURE — path blocked by construction." → Planning Agent finds alternative route, sets new goal.|

### Planning Agent Tools

|Tool|Description|
|---|---|
|`nyc_accessibility_lookup`|Spatial query on SQLite database. Returns scaffolding, elevator outages, accessible pedestrian signals, 311 complaints, and subway station info within a radius of the user's coordinates.|
|`set_ambient_goal`|Writes a new active goal into the Ambient Agent's prompt state.|
|`reset_ambient_goal`|Clears the Ambient Agent's active goal and reverts to Patrol Mode.|
|`web_search`|Searches the web for information not in the NYC database (business hours, weather, transit schedules). Available when WiFi is connected.|

---

## Agent Communication Protocol

The two agents do not communicate directly. All communication flows through a shared **Prompt State** managed by the **Orchestrator**. This is the contract:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR                               │
│                                                                     │
│  Manages shared state, routes signals, prevents speech collisions   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      PROMPT STATE                             │  │
│  │                                                               │  │
│  │  mode: "patrol" | "goal"                                      │  │
│  │  base_goal: "warn about obstacles, read signs..."             │  │
│  │  active_goal: "guide user to 72nd St subway" | null           │  │
│  │  nyc_context: "elevator out at 72nd south entrance..." | ""   │  │
│  │  compiled_prompt: [base_goal + active_goal + nyc_context]     │  │
│  │                                                               │  │
│  │  ← Planning Agent WRITES (set_goal, reset)                    │  │
│  │  → Ambient Agent READS (every frame iteration)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Signal flows:                                                      │
│                                                                     │
│  User voice ──→ Parakeet ASR ──→ Planning Agent ──→ Prompt State   │
│                                        │                            │
│                                        └──→ Magpie TTS ──→ User    │
│                                                                     │
│  Camera ──→ Frame Buffer ──→ Ambient Agent ──→ Magpie TTS ──→ User │
│                                    │                                │
│                                    └── FAILURE ──→ Planning Agent   │
│                                    └── GOAL_REACHED ──→ reset state │
│                                                                     │
│  Speech collision rule:                                             │
│  • Only one audio stream plays at a time.                           │
│  • WARNING from the Ambient Agent preempts all other speech.        │
│  • Planning Agent responses queue behind active Ambient speech.     │
└─────────────────────────────────────────────────────────────────────┘
```

### Communication Rules

1. **The Ambient Agent never calls the Planning Agent directly.** It emits signals (`GOAL_REACHED`, `FAILURE`) to the Orchestrator, which routes them.
2. **The Planning Agent never sees camera frames directly.** For `inspect` actions, the Orchestrator retrieves a frame from the buffer and sends it to the Ambient Agent on the Planning Agent's behalf.
3. **The Prompt State is the only shared mutable state.** The Planning Agent writes to it. The Ambient Agent reads from it. The Orchestrator mediates.
4. **Goal transitions are atomic.** When the Planning Agent calls `set_goal`, the Ambient Agent picks up the new goal on its next frame iteration. There is no partial state.
5. **Patrol Mode is the ground state.** Every goal eventually resolves back to Patrol — either via `GOAL_REACHED`, `FAILURE` + replan, or user `reset`. The system never gets stuck in Goal Mode.

---

## NYC Open Data Integration

All data is pre-cached in a local SQLite database on the GB10's SSD. No internet required at runtime.

|Dataset|What it provides|
|---|---|
|Accessible Pedestrian Signals|Intersections with audio crossing signals for safe street crossing|
|MTA Elevator/Escalator Status|Which station elevators are working right now|
|Scaffolding Permits (DOB NOW)|Active sidewalk sheds that narrow walkways|
|311 Service Requests|Blocked sidewalks, broken ramps, curb hazards, scaffold complaints|
|Pedestrian Ramp Locations|Curb cut locations across 185,000 corners|
|Subway Station Locations + ADA|Which stations are accessible, entrance coordinates|

The Planning Agent queries this data with spatial SQL lookups — not RAG, not vector search. Given the user's GPS coordinates, it finds everything within ~200 meters. Sub-millisecond response time.

## The Pitch (60 seconds)

"800,000 New Yorkers live with a vision disability. They navigate a city where only 23% of subway stations are accessible, elevators break daily, and 8,000 scaffolding sheds turn sidewalks into obstacle courses.

Spark Sight gives blind users something no app offers today: real-time spatial awareness fused with live city data — running entirely on the GB10. No cloud. No internet. No images ever leave the device.

Our two-agent system pairs Cosmos Reason2 as a continuous visual monitor with Nemotron Super as an intelligent planner. When you say 'take me to the subway,' the planner checks which elevators are working, finds scaffolding on your route, and programs the visual agent to guide you there — warning about obstacles and tracking your progress until you arrive. If the path is blocked, it replans automatically.

Four NVIDIA models. All multilingual. One device. Zero compromise on privacy. Built in 48 hours on the hardware you gave us."

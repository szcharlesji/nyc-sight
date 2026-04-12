# LLM Tool: `find_closure`

Instructs the Planning Agent LLM when and how to invoke the street closure lookup.

---

## System Prompt Addition

Insert the following block into the Planning Agent system prompt alongside the other action definitions:

```
- find_closure: Search for active NYC street construction closures by street name.
  Use this when the user asks whether a street is blocked, mentions construction,
  or wants to know if a specific route is passable.
  Add these fields to your JSON output:
    "closure_street"  — street name in ALL CAPS (e.g. "BROADWAY", "SHORE ROAD").
                        Required.
    "closure_borough" — single letter: M=Manhattan, X=Bronx, B=Brooklyn,
                        Q=Queens, S=Staten Island. Omit if unknown.
  Set "message" to a short spoken confirmation like
  "Let me check for closures on Broadway."
```

---

## Expected LLM Output Format

The LLM must emit a single JSON object. Two fields are added on top of the standard schema:

```json
{
  "action": "find_closure",
  "message": "Let me check for closures on Broadway.",
  "closure_street": "BROADWAY",
  "closure_borough": "M"
}
```

Borough is optional:

```json
{
  "action": "find_closure",
  "message": "Checking for construction on Atlantic Avenue.",
  "closure_street": "ATLANTIC AVENUE"
}
```

---

## Trigger Examples

These user utterances should produce `find_closure`:

| User says | `closure_street` | `closure_borough` |
|---|---|---|
| "Is Broadway blocked?" | `BROADWAY` | *(from context or omit)* |
| "Any construction on Shore Road in Staten Island?" | `SHORE ROAD` | `S` |
| "Is Atlantic Avenue passable?" | `ATLANTIC AVENUE` | *(omit)* |
| "I heard Canal Street is closed" | `CANAL STREET` | *(omit)* |
| "How's the sidewalk on 5th Avenue?" | `5 AVENUE` | `M` |

---

## How the Planning Agent Handles This

After the LLM emits `find_closure`, the Planning Agent:

1. Extracts `closure_street` and `closure_borough` from the JSON.
2. Calls the local data server:
   ```
   GET http://localhost:8010/search
       ?street={closure_street}
       &borough={closure_borough}   ← omitted if absent
       &date={today}
       &limit=5
   ```
3. Formats the response into spoken language (see below).
4. Returns `action=ANSWER` with the formatted message — no further orchestrator handling needed.

---

## Spoken Response Templates

**Results found:**

```
I found {count} active closure{s} on {street}{in borough}.
The first is between {from_street} and {to_street},
for {purpose}, through {end_date}.
{There are N more — want me to list them?}
```

Example:
> "I found 3 active closures on Broadway in Manhattan. The first is between Canal Street and Howard Street, for DOT in-house paving, through April 30th. There are 2 more — want me to list them?"

**Single result:**

> "Broadway is closed between Canal Street and Howard Street for DOT in-house paving through April 30th."

**No results:**

> "No active construction closures found on {street} right now."

**Server unreachable:**

> "I can't reach the closure data right now. Please try again in a moment."

---

## Data Server Reference

The closure data server must be running locally before queries are made.

```
Base URL : http://localhost:8010
Health   : GET /health  → {"status":"ok","rows":3314}
Search   : GET /search?street=BROADWAY&borough=M&date=2026-04-12&limit=5
```

See `closure-data/README.md` for full API documentation and startup instructions.

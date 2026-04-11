## Case 1: "Can I Get In?" — Real-Time Station Accessibility Check

### Problem

For blind and low-vision pedestrians, arriving at a subway station only to find the elevator is broken is more than an inconvenience — it can strand them entirely. MTA's official elevator status page is often minutes behind reality. And even when the data is correct, it doesn't answer the actual question: **is *this* station, *right now*, actually accessible through any of its entrances?**

No existing app cross-references elevator outage data with 311 complaint logs to answer that compound question. They report status fields — they don't reason about what those fields mean together.

---

### Why This Requires Reasoning, Not Just Lookup

A single data source cannot answer "can I get in?" reliably:

- **MTA elevator data** tells you the official status — but lags reality by minutes or hours
- **311 complaint data** is crowdsourced and timely — but unstructured and sometimes refers to the same entrance by different names
- **Station ADA structure data** tells you how many accessible entrances exist — but not which are currently operational

The Planning Agent must **synthesize across all three**, resolve naming ambiguities, and reason about what a partial outage actually means for access. The conclusion cannot be read from any single table — it has to be *derived*.

---

### User Profile

> *This assumption is what makes the solution viable — and it's grounded in reality.*

Blind pedestrians frequently need to make a binary decision at a station entrance: **go in, or reroute**. This decision must be made before committing to stairs, turnstiles, or a platform with no way back up.

The user doesn't need navigation. They need **ground truth about a specific place, right now** — delivered before they take the next step.

---

### Demonstrated Approach

**Scene:** User is standing outside a subway station. Cosmos reads the station signage. User asks a single question.

---

**Step 1 — Visual Recognition + Voice Input**

Cosmos (Ambient Agent, Patrol Mode) reads station signage visible in frame:

```
Signal: PROGRESS
Cosmos: "Jay Street-MetroTech station detected. A, C, F lines."
```

> User says: *"Can I get into this station?"*

Parakeet ASR transcribes. Planning Agent wakes up.

**Planning Agent initial reasoning:**
> *"User wants to know if this station is accessible right now. Station identity is known from Cosmos — no GPS needed. I need to check elevator status. But elevator status alone isn't enough: I need to know how many accessible entrances this station has, and whether the outage affects all of them or just one."*

---

**Step 2 — Check Official Elevator Status** *(Tool Call 1)*

```
tool: mta_elevator_status
input: station_name: "Jay St-MetroTech"

output: {
  "station": "Jay St-MetroTech",
  "elevators": [
    {
      "entrance": "A/C/F — Jay Street entrance",
      "status": "OUT OF SERVICE",
      "reason": "Maintenance",
      "since": "06:14 AM"
    },
    {
      "entrance": "A/C — Court Street entrance",
      "status": "UNKNOWN",
      "last_updated": "11:47 PM (yesterday)"
    }
  ]
}
```

**Planning Agent reasoning after Tool Call 1:**
> *"The Jay Street entrance elevator is confirmed out. The Court Street entrance shows UNKNOWN — the data hasn't been updated since last night. That's a data gap, not a confirmation of working. I can't tell the user 'Court Street is fine' based on stale data. 311 complaints may have more recent signal."*

→ **Decides to query 311, not skip it.**

---

**Step 3 — Check 311 Recent Complaints** *(Tool Call 2)*

```
tool: nyc_311_lookup
input: {
  location: "Jay St-MetroTech",
  category: "elevator",
  within_hours: 12
}

output: {
  "complaints": [
    {
      "submitted": "08:22 AM today",
      "description": "Elevator broken at Court Street entrance, 
                       couldn't get my wheelchair through",
      "status": "Open"
    },
    {
      "submitted": "07:55 AM today", 
      "description": "Both elevators at Jay St station out, 
                       no staff to help",
      "status": "Open"
    }
  ]
}
```

**Planning Agent reasoning after Tool Call 2:**
> *"Two complaints this morning — both filed after 6 AM, both still open. The 7:55 AM complaint says 'both elevators out.' The 8:22 AM complaint specifically names the Court Street entrance. This crowdsourced data is more recent than MTA's own records. The UNKNOWN status on Court Street is now effectively confirmed broken by real users. But before I conclude the station is fully inaccessible, I should check whether there are any other entrances I haven't accounted for."*

→ **Decides to check station structure, not conclude yet.**

---

**Step 4 — Check Station ADA Structure** *(Tool Call 3)*

```
tool: nyc_subway_ada_lookup
input: station_name: "Jay St-MetroTech"

output: {
  "station": "Jay St-MetroTech",
  "ada_status": "Partially Accessible",
  "accessible_entrances": [
    {
      "name": "Jay Street (A/C/F)",
      "type": "elevator",
      "notes": "Primary ADA entrance"
    },
    {
      "name": "Court Street (A/C)",
      "type": "elevator", 
      "notes": "Secondary ADA entrance"
    }
  ],
  "non_ada_entrances": [
    {
      "name": "Willoughby Street (F)",
      "type": "stairs only"
    }
  ]
}
```

**Planning Agent reasoning after Tool Call 3:**
> *"Confirmed: this station has exactly two ADA-accessible entrances. Tool Call 1 showed the first is officially out. Tool Calls 2 confirmed the second is also out based on this morning's 311 reports. The third entrance is stairs only — not usable. The station has no currently functional accessible entry point. The conclusion is now fully supported across three independent data sources."*

---

**Step 5 — Response to User**

Planning Agent synthesizes and speaks via Magpie TTS:

> *"If you don't want to use stairs, not right now. Both accessible entrances at Jay Street are out of service. The Jay Street elevator has been down since 6 AM — that's official. The Court Street elevator shows unknown status in MTA's system, but two people filed complaints about it this morning, so it's likely broken too. The only other entrance here is stairs only. Borough Hall station is two blocks east on Joralemon Street — it's fully accessible and showing no outage reports."*

---

### The "So What" for Judges

> MTA's app would have shown one elevator "Out of Service" and one "Unknown" — and left the user to figure out what that means. Spark Sight read the gap in the official data, filled it with 311 signal, cross-referenced the station's entrance structure, and delivered a single actionable answer: **don't go in here, go two blocks east instead.**

> No GPS. No navigation. No memorized route. Just a station name read from a sign — and three datasets reasoned into one clear answer.

---
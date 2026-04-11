
## Case 1: "Take Me Home" — Personalized Route Safety Check


### Problem

For blind pedestrians, crossing intersections is the highest-risk moment of any journey. Audio pedestrian signals are the primary safety mechanism — but they can be disrupted by construction without warning, and no existing navigation tool tracks this in real time.

Standard apps like Google Maps treat all users the same. They optimize for speed, not for whether every crossing on the route has a functioning audio signal. They don't know your specific route. They don't know which intersections you depend on every day.

---

### User Profile

> *This assumption is what makes the solution viable — and it's grounded in reality.*

Blind pedestrians overwhelmingly rely on a **small set of memorized routes** built up over time. Each route is learned through repetition: the number of steps to the first curb cut, which corner has the audio signal, which block to avoid. Deviation is costly and stressful.

This means:
- The set of intersections that matter to any individual user is **small and known in advance**
- The risk isn't "finding a route" — it's **validating that today's conditions on a familiar route are still safe**
- A system that monitors those specific intersections proactively is orders of magnitude more useful than a general-purpose navigation app

---

### Showed Approach

**Scene:** User is about to leave work and head home. A route they walk every day.

---

**Step 1 — Voice Input**

> User says: *"Take me home."*

Parakeet ASR transcribes. Planning Agent wakes up.

---

**Step 2 — Memory Retrieval** *(Tool Call 1)*

```
tool: query_user_memory
input: "home route"

output: {
  "route_name": "home",
  "description": "South on Broadway from W86th to W72nd, 
                  cross Broadway at W72nd St heading east,
                  then south on Amsterdam Ave to W68th",
  "key_intersections": [
    "Broadway & W79th St",
    "Broadway & W72nd St",   ← primary crossing
    "Amsterdam Ave & W68th St"
  ],
  "notes": "User prefers north entrance of 72nd St station 
             as landmark. Usually crosses at W72nd."
}
```

Planning Agent now knows exactly which intersections to check.

---

**Step 3 — Intersection Safety Check** *(Tool Calls 2–4)*

```
tool: nyc_accessibility_lookup
input: "Broadway & W79th St"
output: audio_signal ✅  |  construction ✅ none  |  311 complaints ✅ none
→ CLEAR
```

```
tool: nyc_accessibility_lookup
input: "Broadway & W72nd St"
output: audio_signal ✅  |  construction ⚠️ ACTIVE  |  permit_type: "street closure"
→ FLAG
```

```
tool: nyc_accessibility_lookup
input: "Amsterdam & W68th St"
output: audio_signal ✅  |  construction ✅ none  |  311 complaints ✅ none
→ CLEAR
```

---

**Step 4 — Alternative Crossing** *(Tool Call 5, triggered by FLAG)*

W72nd is the user's usual crossing but has active construction. Planning Agent looks for the nearest safe alternative.

```
tool: nyc_accessibility_lookup
input: "Broadway & W73rd St"
output: audio_signal ✅  |  construction ✅ none  |  311 complaints ✅ none
→ CLEAR — recommend as substitute crossing
```

---

**Step 5 — Response to User**

Planning Agent synthesizes findings and speaks via Magpie TTS:

> *"Your usual route home looks clear today — except for one thing. There's active construction at your usual crossing on 72nd Street. I'm routing you to 73rd Street instead, which has a confirmed audio signal and no construction. Everything else on your route is clear. Setting your goal now."*

Then: `set_goal` → Ambient Agent enters Goal Mode, programmed with the adjusted route and the W72nd Street warning injected into NYC context.

---

### The "So What" for Judges

> Google Maps would have sent her to 72nd Street. Spark Sight checked ahead, found the problem, and already picked a safer crossing — before she left the building.

---

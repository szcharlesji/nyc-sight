## Case 2: "Eyes That Never Blink" — Real-Time Hazard Detection During Active Navigation

### Problem

A white cane sweeps the ground plane — 2D, floor-level only. It cannot detect what's at head height, and it cannot react to what's moving toward you from the side.

NYC sidewalks have both. Scaffolding sheds leave crossbeams at head and chest level. Pedestrians move at speed in unpredictable directions.

No existing navigation app watches the full spatial volume a person moves through. They give directions — they don't watch.

---

### User Profile

> *This assumption is what makes the solution viable — and it's grounded in reality.*

The white cane's sweep arc covers the ground plane — curb cuts, steps, uneven pavement. Its physics are fixed at roughly knee height and below.

This means:

- **Above-ground hazards are invisible** — scaffold crossbeams and protruding hardware exist entirely outside the cane's detection range
- **Moving objects require anticipation** — fast-approaching pedestrians give no tactile warning; hearing partially compensates, but loud urban environments mask approach sounds
- A system with **continuous vision at camera height** fills exactly the gap the cane leaves

---

### Showed Approach

**Scene:** Continue from Case 1 "Take me home" and user has just said *"I don't want to change route."* Planning Agent has set an active goal with progress checkpoints form Case1, and loaded Phase 1 context.

**Active goal:** *"Guide the user north on Broadway toward the 72nd St subway entrance on the right side. Checkpoints: "*

**NYC context — Phase 1:** *"Active scaffolding at 234 W 71st St narrowing the west sidewalk."*

---

**Step 1 — Walking Start** 

```
Signal: PROGRESS
Ambient Agent: [Injected by Alan Agent] checking for road signs and landmarks matching active checkpoints
Message: "There's scaffolding ahead at 234 W 71st, you will approach it in about 5 minutes. I'll let you know when you're close."
```

**Step 2 — Overhead Beam Detected** *(Base goal fires, WARNING overrides)*

Cosmos detects a scaffold crossbeam at head height, 10m ahead. The white cane passes underneath without contact.

```
Signal: WARNING
Spatial: 0.9m height, 10m ahead, center of path
Message: "Slow down. Overhead beam at head height, ten meter ahead. Slow down please."
```

Active goal persists. WARNING preempts all other speech.

---

**Step 3 — Overhead Beam Detected** 

Cosmos detects a scaffold crossbeam at head height, 1.2m ahead. The white cane passes underneath without contact.

```
Signal: WARNING 
Spatial: 0.9m height, 1.2m ahead, center of path
Message: "Stop. Overhead beam at head height, one meter ahead."

Signal: PROGRESS
Ambient Agent: [Injected by Alan Agent] Leading user through the hazard. 
Message: "Step right slightly to avoid the beam, then continue straight. I'll let you know when you're clear."
```

Active goal persists. WARNING preempts all other speech.

---

**Step 4 — All Clear**

Cosmos detects a scaffold crossbeam passed

```
Signal: CLEAR
Message: "You have passed the overhead beam."
```


Ambient Agent picks up Phase 2 context on the next frame.

---

**Step 4 — Pedestrian Collision Warning** *(Base goal fires)*

A pedestrian rounds a corner and closes fast from the front-left.

```
Signal: WARNING
Spatial: front-left, closing fast, ~2 seconds to contact
Message: "Someone coming fast from your left. Stop."
```

> *"Someone coming fast from your left. Stop."*

Person passes. CLEAR. Navigation continues.

---

**Step 5 — Goal Reached**

Cosmos sees the subway entrance signage on the right — matching the active goal landmark.

```
Signal: GOAL_REACHED
Message: "You've arrived. 72nd Street subway entrance is on your right."
```

> *"You've arrived. 72nd Street subway entrance is on your right."*

Ambient Agent reverts to Patrol Mode.

---

### The "So What" for Judges

> The cane never touched the beam — it had no way to. The user's voice confirmed she was past it, and the system updated its context in real time for the next phase of the route. Two hazards, two agents, one device. No cloud. No images left the device.

---
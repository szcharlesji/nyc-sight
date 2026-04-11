## Case 2: "Eyes That Never Blink" — Real-Time Hazard Detection

### Problem

A white cane sweeps the ground plane — knee height and below. It cannot detect what's at head height, and it cannot react to what's moving toward you from the side.

NYC sidewalks have both. Scaffold crossbeams exist entirely outside the cane's detection range. Fast-moving pedestrians give no tactile warning; loud urban environments mask approach sounds.

No existing app watches the full spatial volume a person moves through. Spark Sight does.

---

### Demonstrated Approach

**Scene:** User is walking on a sidewalk. Ambient Agent is in Patrol Mode — no active goal, no navigation. It watches. It waits. It speaks only when something matters.

---

**Step 1 — Overhead Beam, 10 Meters**

Cosmos detects a scaffold crossbeam at head height ahead. The white cane passes underneath — no contact, no signal.

```
Signal: WARNING
Spatial: 0.9m height | 10m ahead | center of path
```
> *"Overhead beam at head height, ten meters ahead. Slow down."*

---

**Step 2 — Overhead Beam, 1 Meter**

User has slowed but the beam is now immediate.

```
Signal: WARNING
Spatial: 0.9m height | 1.2m ahead | center of path
```
> *"Stop. Beam one meter ahead. Step right to clear it."*

---

**Step 3 — All Clear**

Cosmos confirms the hazard is behind the user.

```
Signal: CLEAR
```
> *"You're clear."*

---

**Step 4 — Pedestrian Collision Warning**

A pedestrian rounds a corner and closes fast from the front-left. Two seconds to contact.

```
Signal: WARNING
Spatial: front-left | closing fast | ~2s to contact
```
> *"Someone coming fast from your left. Stop."*

Person passes. Patrol Mode continues.

---

### The "So What" for Judges

> The cane never touched the beam — it had no way to. Two hazards, both invisible to every other tool the user carries. Cosmos caught them both. No cloud. No images left the device.
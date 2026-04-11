## Case 3: "Just Tell Me" — Private, Instant Visual Reasoning at Home

### Problem

Be My Eyes connects blind users to sighted volunteers for visual questions. The concept is generous. The reality is complicated.

**Privacy.** The moment you open Be My Eyes, a stranger sees your home. Your bedroom. The medication on your nightstand. The pile of laundry on the floor. The personal letter on your desk. For questions that arise in private spaces — which happens to be where most daily life occurs — the cost of asking is exposure you didn't choose.

**Latency.** Be My Eyes depends on volunteer availability. At 11 PM, when you're packing for a trip and need to know which shirt is which color, no one may answer. When someone does, you've already waited. The moment has a cost: time, interruption, the quiet indignity of needing to schedule your own independence.

**Dependency.** Every call to Be My Eyes is a reminder that you cannot do this alone — and that your ability to do it depends on a stranger's willingness to pick up. For questions that feel small ("is this Coke or Sprite?"), the social weight of asking a human is disproportionate to the task. Users report avoiding the app for "minor" questions precisely because asking feels like too much to ask.

Spark Sight eliminates all three frictions. No stranger. No wait. No images leave the device.

---

### User Profile

> *This assumption is what makes the solution viable — and it's grounded in reality.*

Blind and low-vision users navigate dozens of small visual decisions every day — selecting items, reading status indicators, distinguishing between nearly identical objects. These decisions don't require navigation or planning. They require a fast, accurate answer to a specific visual question, asked in a private moment, answered instantly.

The barrier is never the complexity of the question. It's always the cost of asking.

---

### Demonstrated Approach

The user speaks. Cosmos looks. The answer comes back in under two seconds. No call placed. No stranger involved. No image transmitted.

These are three moments from a single morning.

---

### Scene A — "Which one is red?"

**The moment:** Getting dressed. Two T-shirts are laid out on the bed.

> User says: *"Which one is the red shirt?"*

Parakeet transcribes. Planning Agent routes to `inspect` action — Cosmos receives the latest frame with a focused prompt.

**Cosmos reasoning:**
```
Two garments visible on bed surface.
Left item: red short-sleeve crew neck. 
Right item: blue short-sleeve crew neck.


Camera angle check: both items positioned toward right side of frame.
Camera is not facing the surface straight-on — slight leftward offset detected.
In-frame "left" does not reliably map to user's physical left.


Answer: The red one is on the left, relative to the other shirt, but you may not face them them directly. I can double check the color after you pick one.

```

**Magpie TTS responds:**
> *"The red one is on the left, relative to the other shirt, but you may not face them them directly. I can double check the color after you pick one."*

**Why this matters:** Be My Eyes would require a volunteer to see the interior of the user's bedroom. Spark Sight sees only what the camera points at, processes it locally, and forgets it immediately. The bedroom stays private.

---

### Scene B — "Which is the Coke?"

**The moment:** Two cans pulled from the fridge. Both feel identical in the hand.

> User says: *"Which one is the Coke?"*

**Cosmos reasoning:**
```
Two aluminum cans visible, held upright.
Front can in the hand:
  Coca-Cola.
Background can on the desk 2 steps ahead:
  Sprite.

Answer: You are holding Coca-Cola and the Sprite is 2 steps ahead.
```

**Magpie TTS responds:**
> *"You are holding Coca-Cola and the Sprite is 2 steps ahead."*

**Why this matters:** This question takes three seconds. A Be My Eyes call — dial, wait, explain the situation, hold still while the volunteer reads the can — takes three minutes. For a question this small, the human cost of asking another human is simply too high. Most users don't make the call. They guess.

---

### Scene C — "Is my power bank charged?"

**The moment:** About to leave the house. Power bank is sitting on the desk.

> User says: *"Is my power bank charged?"*

**Cosmos reasoning:**
```
Rectangular device on desk surface. 
LED indicator array visible on front face: 4 LEDs in row.
LED status:
  LED 1: illuminated (solid white)
  LED 2: illuminated (solid white)
  LED 3: illuminated (solid white)
  LED 4: illuminated (solid white)
All 4 LEDs lit = full charge indicator on standard 4-bar battery display.

Answer: fully charged.
```

**Magpie TTS responds:**
> *"All four lights are on. It's fully charged."*

**Why this matters:** LED indicators are designed to be read visually — they communicate through a code that has no audio equivalent. Be My Eyes can read them, but requires availability and privacy exposure. Spark Sight reads the same visual code, faster, and tells the user not just the answer but the evidence: *"all four lights"* — so the user understands what was seen, not just what was concluded.

---

### The "So What" for Judges

> Three questions. Three seconds each. No stranger saw the bedroom, the fridge, or the desk. No call was placed. No app was opened and closed in frustration when no one picked up.

> Be My Eyes is an act of asking for help. Spark Sight is an act of knowing.

> The difference isn't capability — Be My Eyes volunteers are accurate and generous. The difference is **when** you can use it, **where** you can use it, and **whether asking feels worth it**. Spark Sight removes every condition on the answer. You ask. You know. You move on.

---
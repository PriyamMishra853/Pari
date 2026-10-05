# PARIKSHAK — demo video script (≈ 6 minutes)

A complete tour: landing page → console → three live experiments → camera-rotation proof →
AI copilot & voice → creating a new experiment → logs, recordings, reports → deployment.

---

## Before you press record (10 minutes of setup)

**Run locally** — the laptop gives 3–4 fps; the cloud adds network delay.
```bash
.venv/Scripts/python.exe -m backend.server --port 8766
```
Open **Chrome** → `http://localhost:8766` → allow **camera** and **microphone** when asked.

**Props on the desk**
- a water bottle (clear or labelled, ~0.5 L)
- an open cardboard box (the "container") + a **red** and a **yellow** box (solid, saturated colours, fist-sized)
- a cup (for the custom-experiment scene)
- **AprilTag 10** printed from `http://localhost:8766/tags`, black square measured (e.g. 8 cm), taped
  **upright on the wall or a box behind you, near the middle of the camera view**

**Camera & light**
- Face the window, never sit with the window behind you (backlight kills pose detection).
- Laptop at arm's length; head, shoulders, both hands and the desk in frame.
- Plain background, nothing red/yellow besides the two boxes.

**Console settings** (System tab) → Tag size = your measured size → Apply.
Voice popover 🎙 → *Speak guidance* ON. Hands-free optional; **Hold V to talk** is the reliable one.

**Screen recorder**: Windows **Win + Alt + R** (Xbox Game Bar) or OBS, 1080p, system audio ON
(so the copilot's voice is recorded) and microphone ON (your narration / voice commands).
Close other tabs; reload the console once; press **↺ Reset** before each experiment.

---

## Scene 1 — The problem & landing page (0:00–0:40)

**Do:** open `http://localhost:8766/`, pause on the hero, then scroll slowly to the bottom.
**Show:** the ship with the astronaut descending as you scroll, the altimeter on the right
counting down, phases ORBIT → DE-ORBIT → POWERED DESCENT → TOUCHDOWN, crew egress with the flag.
**Say:**
> "On the space station or the Moon, ground control can be minutes away. PARIKSHAK is an on-board
> witness: it watches every experiment step, guides the astronaut by voice, and flags anything wrong
> or skipped — with no ground link. As we scroll, we descend from orbit to the lunar surface — where
> the experiment begins."

Click **Begin experiment**.

## Scene 2 — Console tour (0:40–1:15)

**Do:** point at each area without starting the camera.
**Say:**
> "Experiment selector with ten procedures — water intake, two-box collision, multi-object
> workstation, and seven modelled on ISRO microgravity research themes. On the right, the live
> procedure: current step, the checks still missing, and the AI flight copilot powered by Groq.
> Below, the 3D rack world, body metrics, the rotation test, datasets, logs and system status."

Open **System** tab briefly: *"Everything runs on this CPU laptop: 3D pose, YOLO on ONNX Runtime,
AprilTag rack frame. SAM 3D Body needs an NVIDIA GPU, so the system says MODEL UNAVAILABLE instead of
faking it."*

## Scene 3 — WBP-1 crew hydration, live (1:15–2:20)

**Do:** select **WBP-1**, ↺ Reset, **Start camera**. Bottle upright on the desk, hands away.
1. Hold still 1 s → S01 verifies, voice says step two.
2. Grasp and lift the bottle to chest height → S02 ✓.
3. Bring it to your lips and hold ~1 s → S03 ✓ (activity **DRINK** on the HUD).
4. Put it down, hands away → S04 ✓, "protocol verified nominal".

**Point at:** mesh + skeleton over your body, bottle box with IN HAND / LIFTED / AT MOUTH tags,
hand→bottle distance line, ACTIVITY chip, INCLINATION chip.
**Say:** *"Every tick is measured — 33 body joints in 3D, the bottle, and the hand-to-mouth distance."*

**Deviation (20 s):** ↺ Reset, repeat but **skip drinking** (lift, then put it down).
After a few seconds: red banner **"Step S03 skipped"**, spoken warning, and a Groq recovery message
in the copilot feed. *"It doesn't just track — it knows the order and says which step was missed."*

## Scene 4 — BCX-1 two-box collision (2:20–3:15)

**Do:** select **BCX-1**, ↺ Reset. Container centred on the desk.
1. Container in view → S01 locked (blue box overlay).
2. Hold up red and yellow boxes, apart → S02 colours verified.
3. Red box into the container → S03. 4. Yellow box in, with a gap → S04.
5. Push them together, hold ~1 s → S05 **collision confirmed**. 6. Pull apart → S06 complete.

**Deviation:** ↺ Reset, after S02 put the **yellow box in first** → **out-of-order** alert + voice.

## Scene 5 — Rack-as-gravity: rotate the laptop (3:15–4:00)

**Do:** select **NBP-1** (neutral body posture). Tag 10 visible behind you → pill **RACK LOCKED**.
Open the **Rotation test** tab → **Start rotation test**. Sit upright, then slowly rotate the
laptop ~90° and back, then ~180° (upside down) and back. Click **⟲ Rack-aligned view** while rotated.
Stop the test.
**Show:** the chart — camera roll (yellow) sweeps, camera-image lean (red) sweeps, but **body
inclination in the rack frame (green) stays flat**; the summary numbers underneath.
**Say:**
> "In microgravity there is no up. PARIKSHAK uses the payload rack as gravity: an AprilTag gives the
> rack's pose, and every joint is expressed in rack coordinates. Turn the camera upside down — the
> camera numbers change completely, the rack-relative posture doesn't."

Then open **3D rack world → Expand**: astronaut mesh, rack axes, and the yellow camera frustum
showing where the laptop is. Drag to orbit.

## Scene 6 — AI copilot & voice (4:00–4:40)

**Do:** with any experiment running, stand still (don't do the step) for ~15 s.
→ the copilot notices the stall and **speaks guidance** for exactly that step (source tag *Groq*).
Then hold **V** and say **"what's next"** → it speaks the step. Say **"review"** → assessment of
your current attempt. Say **"status"** → progress summary.
**Say:** *"Groq only reasons over measurements from the local vision pipeline — it never sees the
video. If the network drops, the copilot falls back to on-board rules."*

## Scene 7 — Create an experiment instantly (4:40–5:25)

**Do:** **+ New experiment** → *Describe in words*, type:
`Pick up the bottle. Shake it three times. Hold it at eye level. Pour it into the cup. Put the bottle down.`
→ **Generate steps** (≈2 s) → show the generated checks → **Create & load** (keep *Start dataset
recording* ticked). The new experiment is in the dropdown and running.
Mention the second tab: *"Or upload a video of yourself doing it once — the pipeline segments the
demonstration into steps and keeps the labelled frames as a training dataset."*

## Scene 8 — Records: logs, recordings, reports (5:25–5:50)

**Do:** click **⏺ Record** at some point earlier (e.g. during Scene 3) and stop it here.
Open **Reports & recordings**: the recording (also auto-downloaded), **Generate witness report**,
and **Flight logs** with the green **CHAIN INTACT** badge — open the *text* version.
**Say:** *"Every session writes a tamper-evident, hash-chained log — a few kilobytes that can be
downlinked instead of gigabytes of video."*

## Scene 9 — Close (5:50–6:10)

**Do:** show the deployed URL (`https://pari-production.up.railway.app` or the Vercel URL) and
press **Present** for the split-screen view.
**Say:**
> "PARIKSHAK — rack-centric activity recognition, step-by-step verification and an AI flight
> copilot, running on the edge. Built for the Bharatiya Antariksh Station and beyond."

---

## If something goes wrong on camera

| Problem | Quick fix |
|---|---|
| ASTRONAUT: NOT DETECTED | face the light, move back so head + shoulders are in view |
| Bottle not boxed | turn the label to the camera, plain background, don't cover it fully with the hand |
| Red/yellow box not found | brighter light, no other red/yellow things in view, hold boxes away from your face |
| RACK LOST | tag flat, well lit, in the middle third of the frame; re-check Tag size |
| Voice does nothing | Hold **V** to talk (works in every browser); speak short commands |
| Steps feel slow to verify | hold each pose ~1 s — the system needs a stable observation, not a flash |
| Something stuck | ↺ Reset the procedure; refresh the page if needed |

**Voice commands:** "what's next" · "review" · "repeat" · "status" · "reset procedure" ·
"start camera" / "stop camera" · "start recording" / "stop recording" · "show mesh" / "hide mesh" ·
"open 3D" / "close 3D" · "mute voice" / "unmute".

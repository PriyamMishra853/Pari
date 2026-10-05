"""Draft a new experiment from a text description or from a demonstration video.

Both produce the same draft the manual builder edits:
    {"title", "category", "objects": {alias: [coco_class]}, "steps": [{name, prompt, hint, hold_s, require, expected_activity}]}

Text  -> Groq turns the description into steps using ONLY the engine's checks
         and the detector's classes; a keyword parser is the offline fallback.
Video -> the real pipeline watches the demonstration (pose, objects, hands),
         the per-frame interaction state is segmented into stable phases, and
         each phase becomes a step. The labelled frames are kept so the new
         experiment starts with a training dataset.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from parikshak.zerog.procedure import PREDICATES, TEXT

# ------------------------------------------------------------------ catalogue
SYNONYMS = {
    "bottle": ["bottle", "flask", "vial", "water", "drink", "canister", "culture"],
    "cup": ["cup", "mug", "glass", "beaker", "receiver"],
    "cell phone": ["phone", "mobile", "smartphone", "tablet", "display", "camera"],
    "laptop": ["laptop", "computer", "notebook computer"],
    "keyboard": ["keyboard"], "mouse": ["mouse"], "remote": ["remote", "controller"],
    "book": ["book", "manual", "checklist", "logbook", "notebook"],
    "scissors": ["scissors", "cutter"], "spoon": ["spoon", "spatula"], "knife": ["knife"],
    "potted plant": ["plant", "seed", "sprout", "chamber", "pot"], "bowl": ["bowl", "tray", "dish", "petri"],
    "red_box": ["red box", "red cube", "red block"], "yellow_box": ["yellow box", "yellow cube", "yellow block"],
    "chair": ["chair", "seat"], "backpack": ["backpack", "bag", "kit"], "clock": ["clock", "timer"],
    "apple": ["apple"], "banana": ["banana"], "orange": ["orange"], "toothbrush": ["toothbrush", "brush"],
    "sports ball": ["ball"], "teddy bear": ["teddy", "toy"], "vase": ["vase", "jar"],
}

ACTIVITIES = ["IDLE", "APPROACH_PAYLOAD", "REACH_TOOL", "GRASP_TOOL", "MOVE_TOOL", "PLACE_TOOL", "RELEASE_TOOL",
              "INSPECT_PAYLOAD", "DRINK", "RETURN_POSITION", "UNEXPECTED_INTERACTION"]

# (regex on the clause, builder(a, b) -> (requires, hint, activity))
VERBS: list[tuple[str, Any]] = [
    (r"\b(pour|tilt|transfer)\b", lambda a, b: ([{"tilted": a}] + ([{"near": [a, b]}] if b else []),
                                                 f"Tilt the {a} over the {b or 'target'}.", "MOVE_TOOL")),
    (r"\b(drink|sip)\b", lambda a, b: ([{"near_mouth": a}], f"Bring the {a} to your mouth.", "DRINK")),
    (r"\b(shake|agitate|mix|stir)\b", lambda a, b: ([{"shaken": a}], f"Shake the {a} several times.", "MOVE_TOOL")),
    (r"\b(above (your |the )?head|overhead|toward(s)? the light|raise .* up high)\b",
     lambda a, b: ([{"above_head": a}] if a else [{"hands_above_head": "both"}], "Lift it above your head.", "MOVE_TOOL")),
    (r"\b(inspect|examine|look at|eye level|read|check)\b",
     lambda a, b: ([{"near_face": a}], f"Hold the {a} in front of your eyes.", "INSPECT_PAYLOAD")),
    (r"\b(put|place|return|stow|set|keep|drop)\b.*\b(into|inside|in)\b",
     lambda a, b: ([{"inside": [a, b]}] if b else [{"resting": a}], f"Place the {a} into the {b or 'container'}.", "PLACE_TOOL")),
    (r"\b(put|place|return|stow|set|keep)\b.*\b(next to|beside|near|by)\b",
     lambda a, b: ([{"near": [a, b]}, {"resting": a}] if b else [{"resting": a}], f"Place the {a} beside the {b or 'target'}.", "PLACE_TOOL")),
    (r"\b(put|place|return|stow|set|keep)\b.*\b(down|back|away|table|bench)\b|\brelease\b|\blet go\b",
     lambda a, b: ([{"resting": a}, {"hands_free": True}] if a else [{"hands_free": True}],
                   f"Put the {a or 'item'} down and move your hands away.", "RELEASE_TOOL")),
    (r"\b(both hands|two hands)\b", lambda a, b: ([{"both_hands_on": a}], f"Hold the {a} with both hands.", "GRASP_TOOL")),
    (r"\b(pick up|pick|grab|grasp|take|lift|hold|retrieve|remove)\b",
     lambda a, b: ([{"in_hand": a}, {"lifted": a}], f"Pick up the {a} and lift it.", "GRASP_TOOL")),
    (r"\b(show|present|locate|identify|place in view|bring .* into view|stage|position)\b",
     lambda a, b: ([{"visible": a}] + ([{"visible": b}] if b else []), f"Hold the {a} in clear view.", "APPROACH_PAYLOAD")),
]
BODY: list[tuple[str, Any]] = [
    (r"\b(raise|lift|put) (both |your )?(hands|arms)\b", ([{"hands_above_head": "both"}], "Raise both hands above your head.", "REACH_TOOL")),
    (r"\b(curl|bend (your )?elbow)", ([{"curl_reps": 3}], "Bend the elbow fully, then straighten it. Three times.", "MOVE_TOOL")),
    (r"\b(lean|tilt your body|bend sideways)\b", ([{"inclined": 25}], "Tilt your upper body to one side.", "APPROACH_PAYLOAD")),
    (r"\b(stand|sit) (straight|upright)|\bneutral (posture|stance)|\bupright\b", ([{"upright": 20}], "Straighten your body.", "IDLE")),
]


def _alias(cls: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cls.lower()).strip("_")


def _find_objects(text: str, known: set[str]) -> list[tuple[int, str]]:
    t = text.lower()
    hits = []
    for cls, words in SYNONYMS.items():
        if cls not in known:
            continue
        for w in words:
            for m in re.finditer(rf"\b{re.escape(w)}s?\b", t):
                hits.append((m.start(), cls))
    hits.sort()
    seen, out = set(), []
    for pos, cls in hits:
        if cls not in seen:
            seen.add(cls)
        out.append((pos, cls))
    return out


def checks_for_clause(clause: str, known: set[str], last: str | None) -> tuple[list | None, str | None, str | None, list[str], str | None]:
    """(requires, hint, activity, classes mentioned, primary object alias) for one sentence.
    'it'/'them' refers to the last object mentioned, so "Shake it" after "Pick up
    the bottle" checks the bottle."""
    objs = []
    for _, cls in _find_objects(clause, known):
        if cls not in objs:
            objs.append(cls)
    aliases = [_alias(c) for c in objs]
    pron = re.search(r"\b(it|them|this|that)\b", clause, re.I) is not None
    if pron and last:
        a = last
        b = next((x for x in aliases if x != last), None)
    else:
        a = aliases[0] if aliases else (last if pron else None)
        b = aliases[1] if len(aliases) > 1 else None
    for pat, fn in BODY:
        if re.search(pat, clause, re.I):
            req, hint, act = fn
            return req, hint, act, objs, a
    if a:
        for pat, fn in VERBS:
            if re.search(pat, clause, re.I):
                req, hint, act = fn(a, b)
                return req, hint, act, objs, a
    return None, None, None, objs, a


def heuristic_from_text(text: str, known: set[str]) -> dict[str, Any]:
    clauses = [c.strip(" .,-") for c in re.split(r"[\n.;]+|\bthen\b|\bafter that\b|\bfinally\b|\d+[.)]\s", text, flags=re.I)]
    clauses = [c for c in clauses if len(c) > 2]
    objects: dict[str, list[str]] = {}
    steps = []
    last = None
    for c in clauses:
        req, hint, act, objs, a = checks_for_clause(c, known, last)
        for cls in objs:
            objects.setdefault(_alias(cls), [cls])
        if a:
            last = a
        if req is None and a:
            req, hint, act = [{"visible": a}], f"Show the {a} to the camera.", "APPROACH_PAYLOAD"
        if req is None:
            continue
        name = c[:1].upper() + c[1:]
        steps.append({"name": name[:48], "prompt": name if name.endswith(".") else name + ".", "hint": hint,
                      "hold_s": 0.8, "require": req, "expected_activity": act})
    title = (clauses[0][:40] if clauses else "Custom procedure").strip().capitalize()
    return {"title": title, "category": "Custom", "objects": objects, "steps": steps[:10]}


CONTRADICT = [("resting", "absent"), ("visible", "absent"), ("in_hand", "hands_free"), ("lifted", "resting"),
              ("in_hand", "resting"), ("near_mouth", "near_face")]


def _prune(req: list[dict[str, Any]], keep: int = 2) -> list[dict[str, Any]]:
    """At most `keep` checks, dropping any that contradict an earlier one."""
    out: list[dict[str, Any]] = []
    for p in req:
        k = next(iter(p))
        if any({k, next(iter(q))} == set(pair) for q in out for pair in CONTRADICT):
            continue
        out.append(p)
    return out[:keep]


def _sanitize(draft: dict[str, Any], known: set[str]) -> dict[str, Any]:
    objects = {}
    for alias, cl in (draft.get("objects") or {}).items():
        cl = [cl] if isinstance(cl, str) else list(cl or [])
        cl = [c for c in cl if c in known]
        if cl:
            objects[_alias(str(alias))] = cl
    steps = []
    for s in draft.get("steps") or []:
        req = []
        for p in s.get("require") or []:
            if not isinstance(p, dict) or len(p) != 1:
                continue
            (k, v), = p.items()
            inner_k = k
            if k == "not" and isinstance(v, dict) and len(v) == 1:
                inner_k, v2 = next(iter(v.items()))
            else:
                v2 = v
            if inner_k not in PREDICATES:
                continue
            args = v2 if isinstance(v2, list) else [v2]
            if inner_k in ("visible", "absent", "in_hand", "both_hands_on", "lifted", "resting", "near_mouth", "near_face",
                           "above_head", "tilted", "shaken", "moving", "near", "inside"):
                args = [_alias(str(x)) for x in args]
                if not all(x in objects for x in args):
                    continue
                v2 = args if inner_k in ("near", "inside") else args[0]
            req.append({"not": {inner_k: v2}} if k == "not" else {inner_k: v2})
        if not req:
            continue
        act = s.get("expected_activity")
        steps.append({"name": str(s.get("name") or "Step")[:60], "prompt": str(s.get("prompt") or s.get("name") or ""),
                      "hint": str(s.get("hint") or ""), "hold_s": float(min(3.0, max(0.3, float(s.get("hold_s") or 0.8)))),
                      "require": req, "expected_activity": act if act in ACTIVITIES else None})
    return {"title": str(draft.get("title") or "Custom procedure")[:70], "category": str(draft.get("category") or "Custom"),
            "objects": objects, "steps": steps[:10]}


TEXT_PROMPT = """Convert this experiment description into a procedure the vision system can verify.
Description: {text}

Rules:
- "objects": map short lowercase names to detector classes. Allowed classes ONLY: {classes}
- Each step "require" is a list of 1-2 checks - the MINIMUM that proves the step. Never add a check the
  description does not ask for (no "both_hands_on" unless both hands are mentioned, no "hands_above_head" unless
  raising hands is asked). Never combine contradictory checks. Each check is a one-key object. Allowed checks ONLY:
{checks}
  Object checks take an object name; "near"/"inside" take [object, object]; "hands_above_head"/"elbow_flexed"/
  "arms_extended"/"wrist_above_shoulder" take "left"|"right"|"both"|"any"; "upright"/"inclined" take degrees;
  "curl_reps" takes a count; "person"/"rack_frame"/"hands_free" take true; wrap a check in {{"not": {{...}}}} to negate.
- 2 to 8 steps, in order. Each step: "name" (<=6 words), "prompt" (spoken instruction), "hint" (what to do if stuck),
  "hold_s" (0.5-1.5), "require" (1-3 checks), "expected_activity" (one of {acts}).
Return JSON: {{"title": "...", "category": "...", "objects": {{...}}, "steps": [...]}}"""


def draft_from_text(text: str, known: set[str]) -> dict[str, Any]:
    from backend.app.groq.client import get_client

    client = get_client()
    meta: dict[str, Any] = {"source": "keyword-parser"}
    if client.available:
        checks = "\n".join(f"  - {k}: {TEXT[k]}" for k in sorted(PREDICATES))
        user = TEXT_PROMPT.format(text=text.strip()[:2000], classes=", ".join(sorted(known)), checks=checks,
                                  acts=", ".join(ACTIVITIES))
        data, m = client.chat_json("You design verifiable lab procedures. Reply with JSON only.", user,
                                   max_tokens=1500, timeout=20.0)
        if data:
            d = _sanitize(data, known)
            if d["steps"]:
                # Groq is good at splitting and wording steps but over-specifies
                # checks; derive each step's checks from its own sentence where a
                # known verb is present, otherwise keep Groq's (pruned to 2).
                last = None
                for st in d["steps"]:
                    req, hint, act, objs, a = checks_for_clause(f"{st['name']}. {st['prompt']}", known, last)
                    for cls in objs:
                        d["objects"].setdefault(_alias(cls), [cls])
                    if a:
                        last = a
                    if req:
                        st["require"] = req
                        st["expected_activity"] = st.get("expected_activity") or act
                        st["hint"] = st.get("hint") or hint
                    else:
                        st["require"] = _prune(st["require"])
                d = _sanitize(d, known)
                return {**d, "source": m.get("source") + " + verb checks", "latency_ms": m.get("latency_ms")}
        meta = {"source": "keyword-parser", "groq_error": m.get("error")}
    d = _sanitize(heuristic_from_text(text, known), known)
    return {**d, **meta}


# ----------------------------------------------------------------- from video
JOBS: dict[str, dict[str, Any]] = {}
IGNORE_CLASSES = {"person", "dining table", "couch", "bed", "tv", "toilet", "refrigerator", "oven", "sink",
                  "microwave", "car", "bench", "chair"}

TOKEN_STEP = {
    "pour": lambda a, b: ("Pour {a} into {b}", "Tilt the {a} over the {b} to transfer.", [{"tilted": a}, {"near": [a, b]}], "MOVE_TOOL"),
    "near_mouth": lambda a, b: ("Bring {a} to the mouth", "Bring the {a} to your mouth.", [{"near_mouth": a}], "DRINK"),
    "shaken": lambda a, b: ("Agitate the {a}", "Shake the {a} several times.", [{"shaken": a}], "MOVE_TOOL"),
    "above_head": lambda a, b: ("Raise {a} overhead", "Lift the {a} above your head and hold.", [{"above_head": a}], "MOVE_TOOL"),
    "near_face": lambda a, b: ("Inspect {a} at eye level", "Hold the {a} in front of your eyes.", [{"near_face": a}], "INSPECT_PAYLOAD"),
    "lifted": lambda a, b: ("Pick up the {a}", "Pick up the {a} and lift it.", [{"in_hand": a}, {"lifted": a}], "GRASP_TOOL"),
    "in_hand": lambda a, b: ("Grasp the {a}", "Take hold of the {a}.", [{"in_hand": a}], "GRASP_TOOL"),
    "near": lambda a, b: ("Place {a} beside {b}", "Put the {a} next to the {b} and let go.", [{"near": [a, b]}, {"resting": a}], "PLACE_TOOL"),
    "resting": lambda a, b: ("Put the {a} down", "Put the {a} down and move your hands away.", [{"resting": a}, {"hands_free": True}], "RELEASE_TOOL"),
    "visible": lambda a, b: ("Stage the {a}", "Place the {a} in clear view.", [{"visible": a}], "APPROACH_PAYLOAD"),
    "hands_up": lambda a, b: ("Raise both hands", "Raise both hands above your head.", [{"hands_above_head": "both"}], "REACH_TOOL"),
    "inclined": lambda a, b: ("Lean to the side", "Tilt your upper body to one side.", [{"inclined": 25}], "APPROACH_PAYLOAD"),
}


def _token(facts, aliases: list[str]) -> tuple[str, str | None, str | None]:
    """The single most specific interaction state in this frame."""
    held = [a for a in aliases if facts.in_hand(a)]
    for a in held:
        for b in aliases:
            if b != a and facts.tilted(a) and facts.near([a, b]):
                return "pour", a, b
    for a in aliases:
        if facts.near_mouth(a):
            return "near_mouth", a, None
    for name in ("shaken", "above_head", "near_face"):
        for a in held:
            if getattr(facts, name)(a):
                return name, a, None
    for a in held:
        if facts.lifted(a):
            return "lifted", a, None
    if held:
        return "in_hand", held[0], None
    if facts.hands_above_head("both"):
        return "hands_up", None, None
    if facts.inclined(30):
        return "inclined", None, None
    for a in aliases:
        for b in aliases:
            if a != b and facts.near([a, b]) and facts.resting(a):
                return "near", a, b
    vis = [a for a in aliases if facts.visible(a)]
    if vis:
        return "visible", vis[0], None
    return "none", None, None


def start_video_job(path: str | Path, det_model) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "status": "running", "progress": 0.0, "message": "starting", "draft": None, "t": time.time()}
    threading.Thread(target=_run_video_job, args=(job_id, Path(path), det_model), daemon=True).start()
    return job_id


def _run_video_job(job_id: str, path: Path, det_model) -> None:
    job = JOBS[job_id]
    try:
        job["draft"] = draft_from_video(path, det_model, lambda p, m: job.update(progress=round(p, 3), message=m))
        job["status"] = "done" if job["draft"]["steps"] else "failed"
        if not job["draft"]["steps"]:
            job["message"] = "No clear object interactions found - hold objects in view and move deliberately."
        else:
            job["message"] = f"{len(job['draft']['steps'])} steps found"
    except Exception as exc:
        job["status"], job["message"] = "failed", f"{type(exc).__name__}: {exc}"
    job["progress"] = 1.0


def draft_from_video(path: Path, det_model, progress=lambda p, m: None, max_frames: int = 120) -> dict[str, Any]:
    import cv2

    from parikshak.zerog.body import body_metrics
    from parikshak.zerog.colors import detect_color_boxes
    from parikshak.zerog.pipeline import ZeroGPipeline
    from parikshak.zerog.procedure import GenericTracker

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if not 1 <= fps <= 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / 4.0)))  # ~4 samples per second
    if total and total / step > max_frames:
        step = max(1, total // max_frames)
    pl = ZeroGPipeline()
    frames: list[dict[str, Any]] = []
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, raw = cap.retrieve()
            if not ok:
                break
            raw = cv2.resize(raw, (640, 480))
            t = i / fps
            ctx = pl.begin(raw, t)
            res = det_model(ctx.view, imgsz=320, verbose=False)[0]
            dets = [{"cls": det_model.names[int(b.cls[0])], "conf": float(b.conf[0]), "box": [float(v) for v in b.xyxy[0].tolist()]}
                    for b in res.boxes if float(b.conf[0]) >= 0.3]
            dets = [d for d in dets if d["cls"] not in IGNORE_CLASSES]
            dets += detect_color_boxes(ctx.view, {"red_box", "yellow_box"})
            pose = _pose_from_ctx(ctx)
            body = body_metrics(ctx.obs, ctx.rack) if (ctx.obs is not None and ctx.obs.cam is not None) else {}
            joints = None
            if ctx.obs is not None and ctx.obs.cam is not None:
                J = ctx.rack.cam_to_rack(ctx.obs.cam) if ctx.rack.valid else ctx.obs.cam * np.array([[1, -1, -1]])
                joints = [[round(float(x), 3) for x in p] for p in J]
            frames.append({"t": t, "dets": dets, "pose": pose, "body": body, "rack": ctx.rack.valid, "joints": joints})
            progress(min(0.85, 0.85 * len(frames) / max(1, (total // step) or max_frames)), f"analysed {len(frames)} frames")
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise ValueError("video has no readable frames")

    counts = Counter(c for f in frames for c in {d["cls"] for d in f["dets"]})
    chosen = [c for c, n in counts.most_common() if n >= max(2, 0.12 * len(frames))][:3]
    aliases = [_alias(c) for c in chosen]
    spec = {"id": "DRAFT", "title": "draft", "objects": {a: [c] for a, c in zip(aliases, chosen)},
            "steps": [{"id": "S01", "name": "observe", "require": [{"person": True}], "hold_s": 1e9}]}
    g = GenericTracker(spec, det_model)
    tokens = []
    for f in frames:
        g.process(None, f["t"], f["pose"], f["body"], f["rack"], f["dets"])
        tokens.append(_token(g.facts, aliases) if g.facts else ("none", None, None))
    progress(0.9, "segmenting phases")

    # majority filter (window 5) then run-length segmentation
    keys = [tk for tk in tokens]
    smooth = []
    for k in range(len(keys)):
        win = keys[max(0, k - 2):k + 3]
        smooth.append(Counter(win).most_common(1)[0][0])
    runs: list[list[Any]] = []
    for k, tk in enumerate(smooth):
        if runs and runs[-1][0] == tk:
            runs[-1][2] = k
        else:
            runs.append([tk, k, k])
    min_len = max(2, int(0.6 * 4))
    phases = [r for r in runs if r[0][0] != "none" and (r[2] - r[1] + 1) >= min_len]
    merged: list[list[Any]] = []
    for r in phases:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    # "in_hand" right before "lifted" of the same object is the same action
    collapsed: list[list[Any]] = []
    for r in merged:
        if collapsed and collapsed[-1][0][0] == "in_hand" and r[0][0] == "lifted" and collapsed[-1][0][1] == r[0][1]:
            collapsed[-1] = [r[0], collapsed[-1][1], r[2]]
        else:
            collapsed.append(r)
    # an object back on the bench after it was handled = "put it down", not "stage it"
    handled: set[str] = set()
    for r in collapsed:
        name, a, b = r[0]
        if name == "visible" and a in handled:
            r[0] = ("resting", a, None)
        if name not in ("visible", "none", "hands_up", "inclined") and a:
            handled.add(a)
    collapsed = collapsed[:8]

    steps = []
    frame_step = [None] * len(frames)
    for si, (tok, a0, a1) in enumerate(collapsed):
        name, a, b = tok
        title, hint, req, act = TOKEN_STEP[name](a, b)
        lbl = lambda x: (x or "").replace("_", " ")  # noqa: E731
        steps.append({"name": title.format(a=lbl(a), b=lbl(b)).capitalize(), "prompt": hint.format(a=lbl(a), b=lbl(b)),
                      "hint": hint.format(a=lbl(a), b=lbl(b)), "hold_s": 0.8, "require": req, "expected_activity": act})
        for k in range(a0, a1 + 1):
            frame_step[k] = si
    progress(1.0, "done")
    samples = [{"i": k, "t": round(f["t"], 3), "step_index": frame_step[k], "token": smooth[k][0],
                "frame": "rack" if f["rack"] else "camera", "joints": f["joints"],
                "objects": [{"cls": d["cls"], "box": [round(v, 1) for v in d["box"]]} for d in f["dets"]]}
               for k, f in enumerate(frames)]
    return {"title": f"Procedure from {path.stem[:30]}", "category": "From demonstration video",
            "objects": {a: [c] for a, c in zip(aliases, chosen)}, "steps": steps, "source": "video-segmentation",
            "frames_analysed": len(frames), "object_counts": dict(counts.most_common(6)), "_samples": samples}


def _pose_from_ctx(ctx) -> dict[str, Any]:
    """BlazePose landmarks in the upright view, in the shape GenericTracker expects."""
    import math as _m

    obs = ctx.obs
    if obs is None:
        return {"detected": False}
    pv = ctx.vt.raw_to_view(obs.px)
    V, W = obs.vis, obs.world

    def ang(a, b, c):
        v1, v2 = W[a] - W[b], W[c] - W[b]
        n = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return None if n < 1e-9 else _m.degrees(_m.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / n)))))

    hands = [tuple(pv[[w, i, p]].mean(0)) for w, i, p in ((15, 19, 17), (16, 20, 18)) if V[w] >= 0.4]
    return {
        "detected": True, "hands": hands,
        "mouth": tuple(pv[[9, 10]].mean(0)) if min(V[9], V[10]) >= 0.3 else None,
        "nose": tuple(pv[0]) if V[0] >= 0.3 else None,
        "shoulder_px": max(float(np.linalg.norm(pv[11] - pv[12])), 40.0), "shoulder_y": float(pv[[11, 12], 1].mean()),
        "wrists": {"left": tuple(pv[15]) if V[15] >= 0.4 else None, "right": tuple(pv[16]) if V[16] >= 0.4 else None},
        "shoulders": {"left": tuple(pv[11]) if V[11] >= 0.4 else None, "right": tuple(pv[12]) if V[12] >= 0.4 else None},
        "elbow_deg": {"left": ang(11, 13, 15) if min(V[11], V[13], V[15]) >= 0.4 else None,
                      "right": ang(12, 14, 16) if min(V[12], V[14], V[16]) >= 0.4 else None},
    }


def save_video_dataset(exp_id: str, draft: dict[str, Any], datasets_dir: Path) -> str | None:
    samples = draft.get("_samples")
    if not samples:
        return None
    d = datasets_dir / exp_id / (time.strftime("%Y%m%d_%H%M%S") + "_from_video")
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "experiment_id": exp_id, "source": "demonstration video", "samples": len(samples),
        "steps": [s["name"] for s in draft.get("steps", [])],
        "label": "step_index = index of the step this frame belongs to (null = between steps)",
        "coordinate_frame": "rack when the AprilTag was visible, else camera (y-up)",
    }, indent=2), encoding="utf-8")
    with open(d / "samples.jsonl", "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, separators=(",", ":")) + "\n")
    return str(d)

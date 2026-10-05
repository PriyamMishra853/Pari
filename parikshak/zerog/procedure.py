"""Data-driven experiment procedures (configs/experiments/*.yaml).

Each step lists observable requirements ("predicates") that must hold for
hold_s seconds, optional forbidden conditions that raise an alert, and a hint.
The engine also:
  * detects a skipped step - the next step's requirements are met while the
    current step's are not - names it and says why,
  * detects a stall - no progress for stall_s - and emits an event so the
    flight copilot can convey guidance for the step the crew is stuck on.

Predicates are evaluated on per-frame facts measured from the upright view
(YOLO boxes, colour boxes, BlazePose landmarks) and the 3D body metrics.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from parikshak.zerog.config import ROOT

EXPERIMENTS_DIR = ROOT / "configs" / "experiments"
from parikshak.zerog.paths import CUSTOM_EXPERIMENTS as CUSTOM_DIR  # noqa: E402

OCCLUSION_GRACE_S = 0.7


# ------------------------------------------------------------------ registry
def load_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for d in (EXPERIMENTS_DIR, CUSTOM_DIR):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.yaml")):
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            exp = data.get("experiment", data)
            if not exp.get("id") or not exp.get("steps"):
                continue
            exp = dict(exp)
            exp["_path"] = str(p)
            exp["custom"] = d == CUSTOM_DIR
            specs[str(exp["id"]).upper()] = exp
    return specs


# --------------------------------------------------------------- object state
@dataclass
class ObjState:
    alias: str
    box: tuple[float, float, float, float] | None = None
    cls: str = ""
    conf: float = 0.0
    last_seen: float = -1e9
    rest_cy: float | None = None
    rest_h: float | None = None
    rest_aspect: float | None = None
    rest_since: float | None = None
    last_in_hand: float = -1e9
    hist: deque = field(default_factory=lambda: deque(maxlen=30))  # (t, cx, cy)

    def visible(self, t: float) -> bool:
        return self.box is not None and (t - self.last_seen) <= OCCLUSION_GRACE_S

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def ext(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1, 1.0)


def _box_gap(a, b) -> float:
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


def _expand(b, pad):
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def _inside(pt, b) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


class Facts:
    """Everything a predicate can ask about, for one frame."""

    def __init__(self, t: float, objects: dict[str, ObjState], pose: dict[str, Any], body: dict[str, Any],
                 rack_locked: bool, reps: dict[str, int]) -> None:
        self.t, self.objects, self.pose, self.body = t, objects, pose, body
        self.rack_locked, self.reps = rack_locked, reps

    # ---- helpers
    def obj(self, alias: str) -> ObjState | None:
        o = self.objects.get(alias)
        return o if o is not None and o.visible(self.t) else None

    def hands(self) -> list[tuple[float, float]]:
        return self.pose.get("hands", [])

    # ---- object predicates
    def visible(self, a):
        return self.obj(a) is not None

    def absent(self, a):
        return self.obj(a) is None

    def in_hand(self, a):
        o = self.obj(a)
        if o is None:
            return False
        # measured: hand gripping a bottle 0 px from its box, hand resting beside it 30 px
        pad = max(12.0, 0.10 * o.ext)
        eb = _expand(o.box, pad)
        return any(_inside(h, eb) for h in self.hands())

    def both_hands_on(self, a):
        o = self.obj(a)
        if o is None or len(self.hands()) < 2:
            return False
        eb = _expand(o.box, max(28.0, 0.40 * o.ext))
        return sum(1 for h in self.hands() if _inside(h, eb)) >= 2

    def lifted(self, a):
        o = self.obj(a)
        if o is None:
            return False
        if o.rest_cy is None:
            return self.in_hand(a)
        cy = o.centre[1]
        return (o.rest_cy - cy) > 0.35 * (o.rest_h or o.ext) and self.in_hand(a)

    def resting(self, a):
        o = self.obj(a)
        return o is not None and not self.in_hand(a) and self._speed(o) < 40.0

    def near_mouth(self, a):
        o, m = self.obj(a), self.pose.get("mouth")
        if m is None:
            return False
        if o is None:
            # at the lips the object is usually hidden by the hand and face: it
            # counts if it was held moments ago and a hand is now at the mouth
            st = self.objects.get(a)
            sw = self.pose.get("shoulder_px", 120.0)
            if st is None or self.t - st.last_in_hand > 1.5:
                return False
            return any(math.hypot(h[0] - m[0], h[1] - m[1]) < 0.7 * sw for h in self.hands())
        sw = self.pose.get("shoulder_px", 120.0)
        x1, y1, x2, y2 = o.box
        top = ((x1 + x2) / 2.0, y1)
        return _inside(m, _expand(o.box, 0.30 * o.ext)) or math.hypot(top[0] - m[0], top[1] - m[1]) < 0.45 * sw

    def near_face(self, a):
        o, n = self.obj(a), self.pose.get("nose")
        if o is None or n is None:
            return False
        sw = self.pose.get("shoulder_px", 120.0)
        cx, cy = o.centre
        sh_y = self.pose.get("shoulder_y", n[1] + sw)
        return math.hypot(cx - n[0], cy - n[1]) < 1.2 * sw and cy < sh_y + 0.2 * sw

    def above_head(self, a):
        o, n = self.obj(a), self.pose.get("nose")
        if o is None or n is None:
            return False
        sw = self.pose.get("shoulder_px", 120.0)
        return o.centre[1] < n[1] - 0.45 * sw

    def tilted(self, a):
        o = self.obj(a)
        if o is None:
            return False
        x1, y1, x2, y2 = o.box
        asp = (x2 - x1) / max(1.0, y2 - y1)
        base = o.rest_aspect or 0.5
        return asp > max(0.85, 1.6 * base)

    def shaken(self, a):
        o = self.obj(a)
        if o is None or not self.in_hand(a):
            return False
        pts = [(t, x, y) for t, x, y in o.hist if self.t - t <= 2.0]
        if len(pts) < 5:
            return False
        reversals = 0
        for axis in (1, 2):
            last_sign = 0
            for i in range(1, len(pts)):
                d = pts[i][axis] - pts[i - 1][axis]
                if abs(d) < 0.06 * o.ext:
                    continue
                s = 1 if d > 0 else -1
                if last_sign and s != last_sign:
                    reversals += 1
                last_sign = s
        return reversals >= 3

    def moving(self, a):
        o = self.obj(a)
        return o is not None and self._speed(o) > 60.0

    def near(self, pair):
        a, b = pair
        oa, ob = self.obj(a), self.obj(b)
        if oa is None or ob is None:
            return False
        return _box_gap(oa.box, ob.box) < 0.15 * min(oa.ext, ob.ext) + 8.0

    def inside(self, pair):
        a, b = pair
        oa, ob = self.obj(a), self.obj(b)
        return oa is not None and ob is not None and _inside(oa.centre, _expand(ob.box, 12.0))

    def _speed(self, o: ObjState) -> float:
        pts = [(t, x, y) for t, x, y in o.hist if self.t - t <= 0.6]
        if len(pts) < 2:
            return 0.0
        dt = max(1e-3, pts[-1][0] - pts[0][0])
        return math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / dt

    # ---- body predicates
    def person(self, _=True):
        return bool(self.pose.get("detected"))

    def hands_above_head(self, which="any"):
        n = self.pose.get("nose")
        if n is None:
            return False
        sw = self.pose.get("shoulder_px", 120.0)
        w = self.pose.get("wrists", {})
        up = {s: (p is not None and p[1] < n[1] - 0.25 * sw) for s, p in w.items()}
        return all(up.values()) and len(up) == 2 if which == "both" else any(up.values())

    def wrist_above_shoulder(self, which="any"):
        w, s = self.pose.get("wrists", {}), self.pose.get("shoulders", {})
        ok = {k: (w.get(k) is not None and s.get(k) is not None and w[k][1] < s[k][1]) for k in ("left", "right")}
        return all(ok.values()) if which == "both" else (ok.get(which, False) if which in ok else any(ok.values()))

    def elbow_flexed(self, which="any"):
        ang = self.pose.get("elbow_deg", {})
        ok = {k: (v is not None and v < 75.0) for k, v in ang.items()}
        return all(ok.values()) if which == "both" else (ok.get(which, False) if which in ok else any(ok.values()))

    def arms_extended(self, which="any"):
        ang = self.pose.get("elbow_deg", {})
        ok = {k: (v is not None and v > 145.0) for k, v in ang.items()}
        return all(ok.values()) if which == "both" else (ok.get(which, False) if which in ok else any(ok.values()))

    def curl_reps(self, n):
        return max(self.reps.values() or [0]) >= int(n)

    def upright(self, max_deg=20):
        inc = self.body.get("inclination_deg")
        return inc is not None and inc <= float(max_deg)

    def inclined(self, min_deg=25):
        inc = self.body.get("inclination_deg")
        return inc is not None and inc >= float(min_deg)

    def rack_frame(self, _=True):
        return self.rack_locked

    def hands_free(self, _=True):
        held = [a for a in self.objects if self.in_hand(a)]
        return not held


PREDICATES = {
    "visible", "absent", "in_hand", "both_hands_on", "lifted", "resting", "near_mouth", "near_face", "above_head",
    "tilted", "shaken", "moving", "near", "inside", "person", "hands_above_head",
    "wrist_above_shoulder", "elbow_flexed", "arms_extended", "curl_reps", "upright", "inclined",
    "rack_frame", "hands_free",
}

TEXT = {
    "visible": "{a} visible", "absent": "{a} out of view", "in_hand": "{a} held in hand",
    "both_hands_on": "{a} held with both hands",
    "lifted": "{a} lifted off the surface", "resting": "{a} resting (not held)",
    "near_mouth": "{a} at the mouth", "near_face": "{a} held at eye level",
    "above_head": "{a} raised above the head", "tilted": "{a} tilted to pour",
    "shaken": "{a} agitated (shake it)", "moving": "{a} moving", "near": "{a} touching {b}",
    "inside": "{a} inside {b}", "person": "crew member in view",
    "hands_above_head": "hands above the head ({a})", "wrist_above_shoulder": "wrist above shoulder ({a})",
    "elbow_flexed": "elbow bent ({a})", "arms_extended": "arms straight ({a})",
    "curl_reps": "{a} elbow curls", "upright": "body upright (within {a}°)",
    "inclined": "body leaning at least {a}°", "rack_frame": "rack frame locked (AprilTag visible)",
    "hands_free": "hands free",
}


def describe(pred: dict[str, Any]) -> str:
    (k, v), = pred.items()
    if k == "not":
        return "NOT " + describe(v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return TEXT.get(k, k).format(a=v[0], b=v[1])
    return TEXT.get(k, k).format(a=v, b="")


def evaluate(facts: Facts, pred: dict[str, Any]) -> bool:
    (k, v), = pred.items()
    if k == "not":
        return not evaluate(facts, v)
    if k == "any_of":
        return any(evaluate(facts, p) for p in v)
    fn = getattr(facts, k, None)
    if k not in PREDICATES or fn is None:
        return False
    return bool(fn(v))


# -------------------------------------------------------------------- engine
@dataclass
class StepRT:
    id: str
    name: str
    prompt: str
    require: list[dict[str, Any]]
    hold_s: float
    forbid: list[dict[str, Any]]
    hint: str
    voice: str
    expected_activity: str | None
    status: str = "pending"
    hold: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    elapsed_s: float = 0.0
    unmet: list[str] = field(default_factory=list)
    met: list[str] = field(default_factory=list)


@dataclass
class Alert:
    step_id: str
    severity: str
    kind: str
    message: str
    tts: str
    timestamp: float

    @property
    def spoken_tts(self) -> str:
        return self.tts


class ProcedureEngine:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.stall_s = float(spec.get("stall_s", 15.0))
        self.steps: list[StepRT] = []
        for i, s in enumerate(spec["steps"]):
            self.steps.append(StepRT(
                id=str(s.get("id", f"S{i + 1:02d}")), name=str(s.get("name", f"Step {i + 1}")),
                prompt=str(s.get("prompt", s.get("description", ""))), require=list(s.get("require", [])),
                hold_s=float(s.get("hold_s", 0.8)), forbid=list(s.get("forbid", [])),
                hint=str(s.get("hint", "")), voice=str(s.get("voice", s.get("prompt", ""))),
                expected_activity=s.get("expected_activity"),
            ))
        self.reset()

    def reset(self, t: float | None = None) -> None:
        t = time.time() if t is None else t
        self.idx = 0
        for s in self.steps:
            s.status, s.hold, s.started_at, s.completed_at, s.elapsed_s = "pending", 0.0, None, None, 0.0
            s.unmet, s.met = [], []
        if self.steps:
            self.steps[0].status, self.steps[0].started_at = "active", t
        self.alerts: list[Alert] = []
        self.events: list[dict[str, Any]] = []
        self._forbid_timers: dict[tuple[str, int], float] = {}
        self._last_alert_at: dict[tuple[str, str], float] = {}
        self._last_progress = t
        self._last_stall_event = -1e9
        self._lookahead = 0.0
        self.complete = False

    @property
    def active(self) -> StepRT | None:
        return self.steps[self.idx] if self.idx < len(self.steps) else None

    def _alert(self, step: StepRT, severity: str, kind: str, message: str, tts: str, t: float) -> None:
        key = (step.id, kind)
        if t - self._last_alert_at.get(key, -1e9) < 6.0:
            return
        self._last_alert_at[key] = t
        a = Alert(step.id, severity, kind, message, tts or message, t)
        self.alerts.append(a)
        self.events.append({"type": "alert", "step_id": step.id, "kind": kind, "severity": severity,
                            "message": message, "t": t})

    def _complete(self, step: StepRT, t: float, status: str = "completed") -> None:
        step.status, step.completed_at = status, t
        step.hold = step.hold_s if status == "completed" else step.hold
        self.events.append({"type": "step_" + status, "step_id": step.id, "name": step.name, "t": t})
        self.idx += 1
        self._last_progress = t
        nxt = self.active
        if nxt is not None:
            nxt.status, nxt.started_at = "active", t
            self.events.append({"type": "step_started", "step_id": nxt.id, "name": nxt.name, "prompt": nxt.prompt,
                                "voice": nxt.voice, "t": t})
        else:
            self.complete = True
            done = sum(1 for s in self.steps if s.status == "completed")
            self.events.append({"type": "procedure_complete", "t": t, "completed": done, "total": len(self.steps)})

    def update(self, facts: Facts, t: float, dt: float) -> None:
        step = self.active
        if step is None:
            return
        step.elapsed_s = t - (step.started_at or t)
        results = [(p, evaluate(facts, p)) for p in step.require]
        step.met = [describe(p) for p, ok in results if ok]
        step.unmet = [describe(p) for p, ok in results if not ok]
        ok_all = all(ok for _, ok in results) if results else True

        if ok_all:
            step.hold += dt
            self._last_progress = t
        else:
            step.hold = max(0.0, step.hold - 0.5 * dt)

        # forbidden conditions
        for i, f in enumerate(step.forbid):
            cond = f.get("when", {k: v for k, v in f.items() if k in PREDICATES or k == "not"})
            key = (step.id, i)
            if cond and evaluate(facts, cond):
                self._forbid_timers[key] = self._forbid_timers.get(key, 0.0) + dt
                if self._forbid_timers[key] >= float(f.get("min_s", 0.4)):
                    self._alert(step, f.get("severity", "caution"), f.get("kind", "deviation"),
                                f.get("message", "Deviation: " + describe(cond)),
                                f.get("tts", ""), t)
            else:
                self._forbid_timers[key] = 0.0

        if step.hold >= step.hold_s:
            self._complete(step, t)
            return

        # skipped step: the NEXT step's requirements hold while this one's don't
        nxt = self.steps[self.idx + 1] if self.idx + 1 < len(self.steps) else None
        if nxt is not None and not ok_all and nxt.require:
            if all(evaluate(facts, p) for p in nxt.require):
                self._lookahead += dt
                if self._lookahead >= max(0.8, nxt.hold_s):
                    missing = "; ".join(step.unmet) or "its checks"
                    self._alert(step, "caution", "skipped",
                                f"Step {step.id} '{step.name}' was skipped - {missing} was never observed. "
                                f"Step {nxt.id} '{nxt.name}' is already being performed.",
                                f"Warning. Step {step.id[1:].lstrip('0')}, {step.name}, was skipped.", t)
                    self._lookahead = 0.0
                    self._complete(step, t, status="skipped")
                    nxt.hold = nxt.hold_s
                    self._complete(nxt, t)
                    return
            else:
                self._lookahead = max(0.0, self._lookahead - dt)

        # stall: nothing has progressed for a while
        if t - self._last_progress >= self.stall_s and t - self._last_stall_event >= self.stall_s:
            self._last_stall_event = t
            self.events.append({"type": "stalled", "step_id": step.id, "name": step.name,
                                "unmet": list(step.unmet), "hint": step.hint, "t": t,
                                "idle_s": round(t - self._last_progress, 1)})

    def drain_events(self) -> list[dict[str, Any]]:
        ev, self.events = self.events, []
        return ev

    def telemetry(self, t: float) -> dict[str, Any]:
        step = self.active
        done = sum(1 for s in self.steps if s.status == "completed")
        recent = self.alerts[-1] if self.alerts and (t - self.alerts[-1].timestamp) < 5.0 else None
        return {
            "active_step_id": step.id if step else "DONE",
            "active_step_name": step.name if step else "Procedure complete",
            "prompt": step.prompt if step else "All steps finished.",
            "hint": step.hint if step else "",
            "unmet": list(step.unmet) if step else [],
            "met": list(step.met) if step else [],
            "compliance_score": int(100 * done / len(self.steps)) if self.steps else 100,
            "is_complete": self.complete,
            "steps": [{
                "id": s.id, "name": s.name, "prompt": s.prompt, "status": s.status,
                "elapsed_s": round(s.elapsed_s, 1),
                "is_verifying": s.status == "active" and s.hold > 0,
                "verification_pct": 100 if s.status == "completed" else int(min(100, 100 * s.hold / max(s.hold_s, 1e-3))),
                "expected_activity": s.expected_activity,
            } for s in self.steps],
            "recent_alert": None if recent is None else {
                "step_id": recent.step_id, "severity": recent.severity, "kind": recent.kind,
                "message": recent.message, "tts": recent.tts, "timestamp": recent.timestamp,
            },
            "alert_count": len(self.alerts),
            "alerts": [{"step_id": a.step_id, "severity": a.severity, "kind": a.kind, "message": a.message,
                        "timestamp": a.timestamp} for a in self.alerts[-20:]],
        }


# ------------------------------------------------------------------ tracker
class GenericTracker:
    """Runs a YAML experiment on the upright view: YOLO boxes (+ colour boxes),
    BlazePose landmarks, and the shared 3D body metrics."""

    def __init__(self, spec: dict[str, Any], detector) -> None:
        self.spec = spec
        self.experiment_id = str(spec["id"]).upper()
        self.title = spec.get("title", self.experiment_id)
        self.rack_id = spec.get("rack_id", "PAYLOAD-RACK")
        self.detector = detector
        self.aliases: dict[str, list[str]] = {}
        for alias, classes in (spec.get("objects") or {}).items():
            self.aliases[str(alias)] = [classes] if isinstance(classes, str) else [str(c) for c in classes]
        self.engine = ProcedureEngine(spec)
        self.reset()

    # report_generator compatibility (same attribute names as YoloExperimentTracker)
    @property
    def steps(self):
        return self.engine.steps

    @property
    def alerts(self):
        return self.engine.alerts

    @property
    def protocol_complete(self) -> bool:
        return self.engine.complete

    def reset(self) -> None:
        self.engine.reset()
        self.frame_count = 0
        self.start_wall_time = time.time()
        self.objs = {a: ObjState(a) for a in self.aliases}
        self.last_t: float | None = None
        self.reps = {"left": 0, "right": 0}
        self._arm_state = {"left": None, "right": None}
        self._rep_step: str | None = None
        self.overlay_objects: list[dict[str, Any]] = []
        self.detections: list[dict[str, Any]] = []
        self.facts: Facts | None = None

    def wanted_classes(self) -> set[str]:
        out: set[str] = set()
        for cl in self.aliases.values():
            out.update(cl)
        return out

    def _update_reps(self, elbow: dict[str, float | None]) -> None:
        step = self.engine.active
        if step is not None and step.id != self._rep_step:
            self._rep_step = step.id
            self.reps = {"left": 0, "right": 0}
        for side, ang in elbow.items():
            if ang is None:
                continue
            st = self._arm_state[side]
            if ang < 70.0:
                self._arm_state[side] = "flexed"
            elif ang > 140.0:
                if st == "flexed":
                    self.reps[side] += 1
                self._arm_state[side] = "extended"

    def process(self, view: np.ndarray, t: float, pose: dict[str, Any], body: dict[str, Any],
                rack_locked: bool, detections: list[dict[str, Any]]) -> dict[str, Any]:
        dt = 0.1 if self.last_t is None else min(max(t - self.last_t, 0.01), 0.4)
        self.last_t = t
        self.frame_count += 1
        self.detections = detections
        # assign best detection per alias
        for alias, classes in self.aliases.items():
            best = None
            for d in detections:
                if d["cls"] in classes and (best is None or d["conf"] > best["conf"]):
                    best = d
            o = self.objs[alias]
            if best is not None:
                o.box, o.cls, o.conf, o.last_seen = tuple(best["box"]), best["cls"], best["conf"], t
                cx, cy = o.centre
                o.hist.append((t, cx, cy))
        self._update_reps(pose.get("elbow_deg", {}))
        facts = Facts(t, self.objs, pose, body, rack_locked, self.reps)
        self.facts = facts
        # resting baselines (calibrated while the object sits untouched)
        for alias, o in self.objs.items():
            if o.visible(t) and not facts.in_hand(alias) and facts._speed(o) < 25.0:
                if o.rest_since is None:
                    o.rest_since = t
                if t - o.rest_since > 0.5:
                    x1, y1, x2, y2 = o.box
                    cy, h = (y1 + y2) / 2.0, y2 - y1
                    o.rest_cy = cy if o.rest_cy is None else 0.8 * o.rest_cy + 0.2 * cy
                    o.rest_h = h if o.rest_h is None else 0.8 * o.rest_h + 0.2 * h
                    asp = (x2 - x1) / max(1.0, h)
                    o.rest_aspect = asp if o.rest_aspect is None else 0.8 * o.rest_aspect + 0.2 * asp
            else:
                o.rest_since = None
        for alias in self.objs:
            if facts.in_hand(alias):
                self.objs[alias].last_in_hand = t
        self.engine.update(facts, t, dt)

        self.overlay_objects = []
        for alias, o in self.objs.items():
            if not o.visible(t):
                continue
            tags = [n for n, f in (("IN HAND", facts.in_hand), ("LIFTED", facts.lifted), ("AT MOUTH", facts.near_mouth),
                                   ("SHAKING", facts.shaken), ("TILTED", facts.tilted)) if f(alias)]
            self.overlay_objects.append({"label": f"{alias} ({o.cls})", "cls": o.cls, "alias": alias,
                                         "box": list(o.box), "conf": round(o.conf, 2), "state": tags,
                                         "color": "#22d3ee" if not tags else "#a3e635"})
        tel = self.engine.telemetry(t)
        tel.update({
            "experiment_id": self.experiment_id,
            "experiment_title": self.title,
            "rack_id": self.rack_id,
            "engine": "procedure-yaml",
            "reps": dict(self.reps),
        })
        return tel

"""Experiment service: one place where a camera frame becomes telemetry.

    frame (browser webcam / uploaded video)
      -> ZeroGPipeline.begin   rack frame, 3D pose, upright view
      -> procedure tracker     tuned (WBP-1, BCX-1, MOA-1) or YAML engine (others)
      -> ZeroGPipeline.finish  3D objects, interaction graph, HAR, mesh, overlay
      -> events -> flight copilot (spoken guidance, alerts, Groq reasoning)
      -> dataset recorder (optional)
"""

from __future__ import annotations

import base64
import math
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np
import yaml

from parikshak.perception.yolo_tracker import YoloExperimentTracker
from parikshak.zerog.body import body_metrics
from parikshak.zerog.colors import detect_color_boxes
from parikshak.zerog.dataset import DatasetRecorder
from parikshak.zerog.hardware import gpu_info, pose_backend_status, sam3d_status
from parikshak.zerog.pipeline import ZeroGPipeline
from parikshak.zerog.procedure import CUSTOM_DIR, PREDICATES, GenericTracker, describe, load_specs

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "runs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TUNED = {"WBP-1", "BCX-1", "MOA-1"}
STALL_S = 15.0


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1, v2 = a - b, c - b
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))))


def _tuned_checks(exp: str, step: str, tel: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Human-readable met / unmet checks for the hand-tuned trackers."""
    g = tel.get("geometry", {})
    st = tel.get("settle", {})
    checks: list[tuple[str, bool]] = []
    if exp == "WBP-1":
        checks = {
            "S01": [("water bottle visible", g.get("target_detected")), ("bottle resting on the table", g.get("in_table_zone"))],
            "S02": [("bottle held in hand", g.get("hand_contact")), ("bottle lifted off the table", g.get("is_lifted"))],
            "S03": [("bottle at the mouth", g.get("near_mouth")), ("drinking pose held", g.get("is_drinking_pose"))],
            "S04": [("bottle back on the table", st.get("is_returned")), ("hands released from bottle", st.get("hands_released"))],
        }.get(step, [])
    elif exp == "BCX-1":
        checks = {
            "S01": [("outer container box detected", g.get("container_locked") or g.get("outer_box_detected"))],
            "S02": [("red box visible", g.get("red_box_detected")), ("yellow box visible", g.get("yellow_box_detected"))],
            "S03": [("red box inside the container", g.get("red_inside"))],
            "S04": [("yellow box inside the container", g.get("yellow_inside")), ("boxes separated", not g.get("is_colliding"))],
            "S05": [("boxes in contact", g.get("is_colliding")), ("both boxes inside", g.get("red_inside") and g.get("yellow_inside"))],
            "S06": [("boxes separated", not g.get("is_colliding"))],
        }.get(step, [])
    else:
        checks = [("crew member visible", g.get("person_detected"))]
    return [n for n, ok in checks if ok], [n for n, ok in checks if not ok]


class CopilotHub:
    """Turns procedure events into spoken guidance; Groq calls run off the frame loop."""

    def __init__(self) -> None:
        from backend.app.groq.reasoning import CopilotReasoner

        self.reasoner = CopilotReasoner()
        self.feed: deque[dict[str, Any]] = deque(maxlen=80)
        self.next_id = 1
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="copilot")
        self.lock = threading.Lock()
        self.last_ai = 0.0
        self.auto_ai = True

    def post(self, kind: str, title: str, text: str, spoken: str | None = None, source: str = "local",
             severity: str | None = None, latency_ms: float | None = None, extra: dict | None = None) -> dict[str, Any]:
        with self.lock:
            msg = {"id": self.next_id, "t": time.time(), "type": kind, "title": title, "text": text,
                   "spoken": spoken, "source": source, "severity": severity, "latency_ms": latency_ms}
            if extra:
                msg.update(extra)
            self.next_id += 1
            self.feed.append(msg)
            return msg

    def since(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            return [m for m in self.feed if m["id"] > after_id]

    def clear(self) -> None:
        with self.lock:
            self.feed.clear()

    def _async(self, fn, *args) -> None:
        self.pool.submit(self._safe, fn, *args)

    def _safe(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # never let the copilot break tracking
            self.post("system", "Copilot error", f"{type(exc).__name__}: {exc}", source="local")

    # ----------------------------------------------------------- reactions
    def on_events(self, events: list[dict[str, Any]], tel: dict[str, Any], spec: dict[str, Any] | None) -> None:
        if not events:
            return
        from backend.app.groq.reasoning import experiment_state

        steps = {s["id"]: s for s in tel.get("steps", [])}
        spec_steps = {s.get("id"): s for s in (spec or {}).get("steps", [])}
        completed = [e for e in events if e["type"] == "step_completed"]
        started = [e for e in events if e["type"] == "step_started"]
        for e in events:
            if e["type"] == "alert":
                self.post("alert", f"{e.get('severity', 'caution').upper()} - Step {e['step_id']}", e["message"],
                          spoken=e.get("tts") or e["message"], severity=e.get("severity"), source="vision-rules")
                if self.auto_ai and e.get("severity") in ("caution", "critical") and time.time() - self.last_ai > 6:
                    self.last_ai = time.time()
                    st = experiment_state(tel, spec, e)
                    self._async(self._ai_alert, st, e)
            elif e["type"] == "stalled":
                st = experiment_state(tel, spec, e)
                if self.auto_ai:
                    self._async(self._ai_guidance, st)
                else:
                    g = self.reasoner._local_guidance(st)
                    self.post("guide", f"Guidance - Step {e['step_id']}", g["display"], spoken=g["spoken"], source="local-rules")
            elif e["type"] == "procedure_complete":
                st = experiment_state(tel, spec, e)
                self._async(self._ai_summary, st)
        if started:
            e = started[-1]
            sp = spec_steps.get(e["step_id"], {})
            prefix = f"Step {completed[-1]['step_id'][1:].lstrip('0')} verified. " if completed else ""
            voice = sp.get("voice") or f"Next, {steps.get(e['step_id'], {}).get('prompt', e.get('prompt', ''))}"
            self.post("guide", f"Next: {e['step_id']} {e.get('name', '')}",
                      steps.get(e["step_id"], {}).get("prompt", e.get("prompt", "")),
                      spoken=(prefix + voice).strip(), source="procedure")

    def _ai_alert(self, st, e) -> None:
        r = self.reasoner.explain_alert(st, e)
        self.post("ai", f"Recovery - Step {e['step_id']}", r.get("display", ""), spoken=r.get("spoken"),
                  source=r.get("source", "?"), severity=e.get("severity"), latency_ms=r.get("latency_ms"))

    def _ai_guidance(self, st) -> None:
        r = self.reasoner.guidance(st)
        s = st.get("active_step") or {}
        self.post("guide", f"Stuck on step {s.get('id', '')}? Guidance", r.get("display", ""), spoken=r.get("spoken"),
                  source=r.get("source", "?"), latency_ms=r.get("latency_ms"), extra={"checks": r.get("checks", [])})

    def _ai_summary(self, st) -> None:
        r = self.reasoner.summary(st)
        self.post("ai", "Procedure summary", r.get("display", ""), spoken=r.get("spoken"),
                  source=r.get("source", "?"), latency_ms=r.get("latency_ms"))


class TrackerService:
    def __init__(self, experiment_id: str = "WBP-1") -> None:
        self.lock = threading.RLock()
        self.pipeline = ZeroGPipeline()
        self.yolo = YoloExperimentTracker("WBP-1")
        self.yolo.render_hud = False
        self.yolo.external_pose_mode = True
        self.specs = load_specs()
        self.generic: GenericTracker | None = None
        self.copilot = CopilotHub()
        self.dataset = DatasetRecorder()
        self._det_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        self.toggles = {"show_mesh": True, "show_skeleton": True, "show_joints": True,
                        "show_rack_axes": True, "show_interactions": True}
        self.video_cap: Any | None = None
        self.active_video_path: str | None = None
        self.video_fps = 25.0
        self.video_t0: float | None = None
        self.video_pos0 = 0
        self.last_telemetry: dict[str, Any] = {}
        self._prev_steps: dict[str, str] = {}
        self._prev_alert_ts = 0.0
        self._prev_complete = False
        self._progress_sig: tuple | None = None
        self._last_progress_t = time.time()
        self._last_stall_t = 0.0
        self.experiment_id = "WBP-1"
        self.set_experiment(experiment_id)

    # --------------------------------------------------------------- basics
    @property
    def tracker(self):
        return self.generic if self.generic is not None else self.yolo

    @property
    def spec(self) -> dict[str, Any] | None:
        return self.specs.get(self.experiment_id)

    def set_toggles(self, new_toggles: dict[str, bool]) -> dict[str, bool]:
        with self.lock:
            self.toggles.update({k: bool(v) for k, v in new_toggles.items()})
            return dict(self.toggles)

    def _reset_event_state(self) -> None:
        self._prev_steps, self._prev_alert_ts, self._prev_complete = {}, 0.0, False
        self._progress_sig, self._last_progress_t, self._last_stall_t = None, time.time(), 0.0

    def set_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self.lock:
            self.specs = load_specs()
            exp = experiment_id.upper()
            if exp not in self.specs and exp not in TUNED:
                exp = "WBP-1"
            self.experiment_id = exp
            if exp in TUNED:
                self.generic = None
                self.yolo.init_procedure(exp)
            else:
                self.generic = GenericTracker(self.specs[exp], self.yolo.det_model)
            self.pipeline.reset()
            self.copilot.clear()
            self._reset_event_state()
            first = (self.spec or {}).get("steps", [{}])[0]
            self.copilot.post("guide", f"{exp} loaded - Step {first.get('id', 'S01')}", first.get("prompt", ""),
                              spoken=f"{(self.spec or {}).get('title', exp)} loaded. "
                                     + (first.get("voice") or f"Step one. {first.get('prompt', '')}"),
                              source="procedure")
            self.last_telemetry = self._idle_telemetry()
            return self.last_telemetry

    def reset(self) -> dict[str, Any]:
        with self.lock:
            if self.generic is not None:
                self.generic.reset()
            else:
                self.yolo.reset()
            self.pipeline.reset()
            self._reset_event_state()
            if self.video_cap is not None:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_t0, self.video_pos0 = None, 0
            self.copilot.post("system", "Procedure restarted", "All steps reset to pending.", source="procedure")
            self.last_telemetry = self._idle_telemetry()
            return self.last_telemetry

    reset_tracker = reset

    def _idle_telemetry(self) -> dict[str, Any]:
        spec = self.spec or {}
        steps = [{"id": s.get("id"), "name": s.get("name"), "prompt": s.get("prompt"),
                  "status": "active" if i == 0 else "pending", "elapsed_s": 0.0, "is_verifying": False,
                  "verification_pct": 0, "expected_activity": s.get("expected_activity")}
                 for i, s in enumerate(spec.get("steps", []))]
        first = steps[0] if steps else {}
        return {
            "experiment_id": self.experiment_id, "experiment_title": spec.get("title", self.experiment_id),
            "rack_id": spec.get("rack_id", ""), "active_step_id": first.get("id", "S01"),
            "active_step_name": first.get("name", ""), "prompt": first.get("prompt", ""),
            "hint": (spec.get("steps") or [{}])[0].get("hint", ""), "unmet": [], "met": [],
            "compliance_score": 0, "is_complete": False, "steps": steps, "recent_alert": None,
            "alert_count": 0, "alerts": [], "feed": self.copilot.since(0), "zerog": None, "events": [],
            "dataset": self.dataset.status(),
        }

    def get_telemetry(self) -> dict[str, Any]:
        with self.lock:
            tel = dict(self.last_telemetry or self._idle_telemetry())
            tel["feed"] = self.copilot.since(0)
            return tel

    # ------------------------------------------------------------ experiments
    def get_experiments_list(self) -> dict[str, Any]:
        self.specs = load_specs()
        out = []
        for eid, s in self.specs.items():
            out.append({
                "id": eid, "name": s.get("title", eid), "title": s.get("title", eid),
                "category": s.get("category", "Custom" if s.get("custom") else ""),
                "inspired_by": s.get("inspired_by", ""), "props": s.get("props", ""),
                "rack": s.get("rack_id", ""), "steps_count": len(s.get("steps", [])),
                "engine": "tuned" if eid in TUNED else "procedure-yaml", "custom": bool(s.get("custom")),
                "steps": [{"id": st.get("id"), "name": st.get("name"), "prompt": st.get("prompt")} for st in s.get("steps", [])],
            })
        return {"experiments": out, "active_experiment_id": self.experiment_id}

    def create_custom_experiment(self, data: dict[str, Any]) -> dict[str, Any]:
        title = str(data.get("title", "")).strip() or "Custom procedure"
        slug = re.sub(r"[^A-Z0-9]+", "", title.upper())[:6] or "EXP"
        exp_id = f"CUS-{slug}{time.strftime('%H%M%S')[-4:]}"
        objects = {}
        for alias, classes in (data.get("objects") or {}).items():
            a = re.sub(r"[^a-z0-9_]+", "_", str(alias).lower()).strip("_")
            cl = [classes] if isinstance(classes, str) else list(classes)
            if a and cl:
                objects[a] = [str(c) for c in cl]
        known = set(self.yolo.det_model.names.values()) | {"red_box", "yellow_box"}
        bad_cls = [c for cl in objects.values() for c in cl if c not in known]
        if bad_cls:
            return {"error": f"Unknown object classes: {bad_cls}. Use YOLO/COCO class names or red_box / yellow_box."}
        steps = []
        for i, s in enumerate(data.get("steps") or []):
            req = s.get("require") or []
            for p in req:
                k = next(iter(p))
                inner = p[k] if k == "not" else p
                kk = next(iter(inner))
                if kk not in PREDICATES:
                    return {"error": f"Step {i + 1}: unknown check '{kk}'."}
                arg = inner[kk]
                for a in (arg if isinstance(arg, list) else [arg]):
                    if isinstance(a, str) and kk not in ("hands_above_head", "wrist_above_shoulder", "elbow_flexed",
                                                         "arms_extended") and a not in objects:
                        return {"error": f"Step {i + 1}: '{a}' is not one of the experiment's objects {list(objects)}."}
            name = str(s.get("name", f"Step {i + 1}")).strip() or f"Step {i + 1}"
            steps.append({
                "id": f"S{i + 1:02d}", "name": name, "prompt": str(s.get("prompt") or name),
                "voice": str(s.get("voice") or f"Step {i + 1}. {s.get('prompt') or name}"),
                "require": req, "hold_s": float(s.get("hold_s", 0.8)),
                "hint": str(s.get("hint") or ""), "expected_activity": s.get("expected_activity") or None,
                "forbid": s.get("forbid") or [],
            })
        if not steps:
            return {"error": "Add at least one step."}
        spec = {"experiment": {
            "id": exp_id, "title": title, "category": str(data.get("category") or "Custom"),
            "inspired_by": str(data.get("inspired_by") or "User-defined procedure"),
            "props": str(data.get("props") or ", ".join(sorted({c for cl in objects.values() for c in cl}))),
            "rack_id": "PAYLOAD-RACK", "objects": objects, "stall_s": 15, "steps": steps,
            "author": str(data.get("author") or "crew"), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }}
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        path = CUSTOM_DIR / f"{exp_id}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self.specs = load_specs()
        return {"status": "CREATED", "experiment_id": exp_id, "path": str(path),
                "steps": [{"id": s["id"], "name": s["name"], "checks": [describe(p) for p in s["require"]]} for s in steps]}

    def object_classes(self) -> list[str]:
        return sorted(set(self.yolo.det_model.names.values()) | {"red_box", "yellow_box"})

    # --------------------------------------------------------------- frames
    def process_b64_frame(self, b64_data: str, include_frame: bool = False) -> dict[str, Any]:
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        try:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            return {"error": f"Failed to decode image: {exc}"}
        if frame is None:
            return {"error": "Invalid image data"}
        tel = self.process_frame(frame, time.time())
        return {"telemetry": tel}

    process_client_frame = process_b64_frame

    @staticmethod
    def _detections(res, names, wanted: set[str] | None) -> list[dict[str, Any]]:
        out = []
        if res is None:
            return out
        for b in res.boxes:
            cls = names[int(b.cls[0])]
            conf = float(b.conf[0])
            if conf < 0.25 or (wanted is not None and cls not in wanted):
                continue
            out.append({"cls": cls, "conf": round(conf, 3), "box": [float(v) for v in b.xyxy[0].tolist()]})
        return out

    def _pose_dict(self, ctx) -> dict[str, Any]:
        obs, vt = ctx.obs, ctx.vt
        if obs is None:
            # BlazePose found nobody: fall back to YOLO-pose 2D keypoints in the view
            sk = self.yolo._extract_full_body_skeleton(ctx.view, ctx.t, 0.1, False) if self.generic is not None else None
            if not sk or not sk.get("detected"):
                return {"detected": False}
            kp = {k["name"]: (k["x"], k["y"], k["conf"]) for k in sk["keypoints"]}
            ok = lambda n: kp.get(n) and kp[n][2] >= 0.3  # noqa: E731
            sw = math.hypot(kp["left_shoulder"][0] - kp["right_shoulder"][0], kp["left_shoulder"][1] - kp["right_shoulder"][1]) \
                if ok("left_shoulder") and ok("right_shoulder") else 120.0
            pd = {"detected": True, "source": "yolo-pose", "shoulder_px": max(sw, 40.0),
                  "nose": kp["nose"][:2] if ok("nose") else None,
                  "mouth": (kp["nose"][0], kp["nose"][1] + 0.25 * sw) if ok("nose") else None,
                  "hands": [kp[n][:2] for n in ("left_wrist", "right_wrist") if ok(n)],
                  "wrists": {s: (kp[f"{s}_wrist"][:2] if ok(f"{s}_wrist") else None) for s in ("left", "right")},
                  "shoulders": {s: (kp[f"{s}_shoulder"][:2] if ok(f"{s}_shoulder") else None) for s in ("left", "right")},
                  "elbow_deg": {}}
            for s in ("left", "right"):
                if ok(f"{s}_shoulder") and ok(f"{s}_elbow") and ok(f"{s}_wrist"):
                    pd["elbow_deg"][s] = _angle(np.array(kp[f"{s}_shoulder"][:2]), np.array(kp[f"{s}_elbow"][:2]),
                                                np.array(kp[f"{s}_wrist"][:2]))
                else:
                    pd["elbow_deg"][s] = None
            sh = [p for p in pd["shoulders"].values() if p is not None]
            pd["shoulder_y"] = float(np.mean([p[1] for p in sh])) if sh else None
            return pd
        pv = vt.raw_to_view(obs.px)
        V, W = obs.vis, obs.world
        hands = []
        for w, i, p in ((15, 19, 17), (16, 20, 18)):
            if V[w] >= 0.4:
                hands.append(tuple(pv[[w, i, p]].mean(0)))
        sw = float(np.linalg.norm(pv[11] - pv[12]))
        pd = {
            "detected": True, "source": "blazepose", "hands": hands,
            "mouth": tuple(pv[[9, 10]].mean(0)) if min(V[9], V[10]) >= 0.3 else None,
            "nose": tuple(pv[0]) if V[0] >= 0.3 else None,
            "shoulder_px": max(sw, 40.0), "shoulder_y": float(pv[[11, 12], 1].mean()),
            "wrists": {"left": tuple(pv[15]) if V[15] >= 0.4 else None, "right": tuple(pv[16]) if V[16] >= 0.4 else None},
            "shoulders": {"left": tuple(pv[11]) if V[11] >= 0.4 else None, "right": tuple(pv[12]) if V[12] >= 0.4 else None},
            "elbow_deg": {
                "left": _angle(W[11], W[13], W[15]) if min(V[11], V[13], V[15]) >= 0.4 else None,
                "right": _angle(W[12], W[14], W[16]) if min(V[12], V[14], V[16]) >= 0.4 else None,
            },
        }
        return pd

    def _run_det(self, view: np.ndarray, _k: int = 0):
        t0 = time.perf_counter()
        res = self.yolo.det_model(view, imgsz=320, verbose=False)[0]
        return res, (time.perf_counter() - t0) * 1000.0

    def _detect_view(self, ctx) -> tuple[Any, float]:
        """Result of the detection started on the predicted view, or a fresh run
        when the pose model chose a different orientation."""
        if ctx.early is not None:
            k_pred, _, fut = ctx.early
            res, ms = fut.result()
            if k_pred == ctx.vt.k:
                return res, ms
        return self._run_det(ctx.view)

    def process_frame(self, raw: np.ndarray, t: float) -> dict[str, Any]:
        with self.lock:
            t_start = time.perf_counter()
            if raw.shape[1] != 640 or raw.shape[0] != 480:
                raw = cv2.resize(raw, (640, 480))
            pl = self.pipeline
            spec = self.spec or {}
            if self.generic is None:
                need_det = self.experiment_id != "BCX-1"  # BCX-1 is colour-based; YOLO is not needed
            else:
                need_det = bool(self.generic.wanted_classes() - {"red_box", "yellow_box"})
            launch = (lambda v, k: self._det_pool.submit(self._run_det, v, k)) if need_det else None
            ctx = pl.begin(raw, t, on_predicted_view=launch)
            view = ctx.view
            expected = sorted({c for cl in (spec.get("objects") or {}).values() for c in (cl if isinstance(cl, list) else [cl])})

            if self.generic is None:
                tr = self.yolo
                tr.external_pose = pl.coco17_for_view(ctx)
                det_ms = 0.0
                if need_det:
                    tr.injected_det, det_ms = self._detect_view(ctx)
                _, tel = tr.process_frame(view, t)
                tr.injected_det = None
                overlay_objs = list(tr.overlay_objects)
                skel_view = tel.get("geometry", {}).get("skeleton") if ctx.obs is None else None
                met, unmet = _tuned_checks(self.experiment_id, tel.get("active_step_id", ""), tel)
                tel["met"], tel["unmet"] = met, unmet
                sp = next((s for s in spec.get("steps", []) if s.get("id") == tel.get("active_step_id")), {})
                tel["hint"] = sp.get("hint", "")
                tel["alerts"] = [{"step_id": a.step_id, "severity": a.severity, "kind": a.kind, "message": a.message,
                                  "timestamp": a.timestamp} for a in tr.alerts[-20:]]
                tel["engine"] = "tuned"
            else:
                g = self.generic
                wanted = g.wanted_classes()
                yolo_wanted = wanted - {"red_box", "yellow_box"}
                det_ms = 0.0
                dets: list[dict[str, Any]] = []
                if yolo_wanted:
                    res, det_ms = self._detect_view(ctx)
                    dets = self._detections(res, self.yolo.det_model.names, yolo_wanted)
                if wanted & {"red_box", "yellow_box"}:
                    dets += detect_color_boxes(view, wanted)
                pose = self._pose_dict(ctx)
                body = body_metrics(ctx.obs, ctx.rack) if (ctx.obs is not None and ctx.obs.cam is not None) else {}
                tel = g.process(view, t, pose, body, ctx.rack.valid, dets)
                overlay_objs = list(g.overlay_objects)
                skel_view = None

            z = pl.finish(ctx, skel_view, overlay_objs, expected, self.toggles)
            z["perf"]["det_ms"] = round(det_ms, 1)
            z["perf"]["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            tel["zerog"] = z

            events = self.generic.engine.drain_events() if self.generic is not None else self._tuned_events(tel, t)
            events += self._stall_events(tel, t) if self.generic is None else []
            self.copilot.on_events(events, tel, spec)
            tel["events"] = events
            tel["feed"] = self.copilot.since(0)
            if self.dataset.active:
                self.dataset.add(view, tel)
            tel["dataset"] = self.dataset.status()
            tel["toggles"] = dict(self.toggles)
            self.last_telemetry = tel
            return tel

    def _tuned_events(self, tel: dict[str, Any], t: float) -> list[dict[str, Any]]:
        ev: list[dict[str, Any]] = []
        cur = {s["id"]: s["status"] for s in tel.get("steps", [])}
        if self._prev_steps:
            for s in tel.get("steps", []):
                before = self._prev_steps.get(s["id"])
                if before != s["status"]:
                    if s["status"] in ("completed", "skipped"):
                        ev.append({"type": "step_" + s["status"], "step_id": s["id"], "name": s["name"], "t": t})
                    elif s["status"] == "active":
                        ev.append({"type": "step_started", "step_id": s["id"], "name": s["name"], "prompt": s["prompt"], "t": t})
        self._prev_steps = cur
        ra = tel.get("recent_alert")
        if ra and ra.get("timestamp", 0) > self._prev_alert_ts:
            self._prev_alert_ts = ra["timestamp"]
            if ra.get("severity") not in ("info",):
                ev.append({"type": "alert", "step_id": ra["step_id"], "kind": ra["kind"], "severity": ra["severity"],
                           "message": ra["message"], "tts": ra.get("tts"), "t": t})
        if tel.get("is_complete") and not self._prev_complete:
            ev.append({"type": "procedure_complete", "t": t})
        self._prev_complete = bool(tel.get("is_complete"))
        return ev

    def _stall_events(self, tel: dict[str, Any], t: float) -> list[dict[str, Any]]:
        active = next((s for s in tel.get("steps", []) if s["status"] == "active"), None)
        if active is None:
            return []
        sig = (active["id"], active.get("verification_pct", 0))
        if sig != self._progress_sig:
            if self._progress_sig is None or sig[0] != self._progress_sig[0] or sig[1] > self._progress_sig[1]:
                self._last_progress_t = t
            self._progress_sig = sig
        if t - self._last_progress_t >= STALL_S and t - self._last_stall_t >= STALL_S:
            self._last_stall_t = t
            return [{"type": "stalled", "step_id": active["id"], "name": active["name"], "unmet": tel.get("unmet", []),
                     "hint": tel.get("hint", ""), "idle_s": round(t - self._last_progress_t, 1), "t": t}]
        return []

    def simulate_event(self, event_name: str) -> dict[str, Any]:
        """Inject nominal steps or deviations into the tuned trackers (tests and
        offline rehearsal only - never called by the live pipeline)."""
        with self.lock:
            now = time.time()
            if event_name == "nominal_step":
                if self.tracker.step_idx < len(self.tracker.steps):
                    cur = self.tracker.steps[self.tracker.step_idx]
                    cur.status = "completed"
                    cur.completed_at = now
                    if self.tracker.experiment_id in ["BCX-1", "BOX-COL-1"]:
                        if cur.id == "S01":
                            self.tracker.bcx1_s01_hold_duration = 0.8
                            self.tracker.bcx1_container_locked = True
                        elif cur.id == "S02":
                            self.tracker.bcx1_s02_hold_duration = 0.8
                            self.tracker.bcx1_colors_verified = True
                        elif cur.id == "S03":
                            self.tracker.bcx1_s03_hold_duration = 0.8
                            self.tracker.bcx1_red_placed = True
                            self.tracker.bcx1_red_inside = True
                        elif cur.id == "S04":
                            self.tracker.bcx1_s04_hold_duration = 0.8
                            self.tracker.bcx1_yellow_placed = True
                            self.tracker.bcx1_yellow_inside = True
                            self.tracker.bcx1_distance_px = 60.0
                        elif cur.id == "S05":
                            self.tracker.bcx1_collision_hold_duration = 0.8
                            self.tracker.bcx1_collision = True
                            self.tracker.bcx1_distance_px = 0.0
                        elif cur.id == "S06" or self.tracker.step_idx >= len(self.tracker.steps) - 1:
                            self.tracker.bcx1_s06_hold_duration = 0.7
                            self.tracker.bcx1_collision = False
                            self.tracker.bcx1_distance_px = 40.0
                            self.tracker.protocol_complete = True
                    else:
                        if cur.id == "S01":
                            self.tracker.s01_stable_duration = self.tracker.S01_TARGET_S
                        elif cur.id == "S02":
                            self.tracker.target_lifted = True
                            self.tracker.s02_lift_duration = self.tracker.S02_TARGET_S
                        elif cur.id == "S03":
                            self.tracker.water_consumed = True
                            self.tracker.drink_hold_duration = self.tracker.S03_TARGET_S
                        elif cur.id == "S04" or self.tracker.step_idx >= len(self.tracker.steps) - 1:
                            self.tracker.protocol_complete = True
                            self.tracker.s04_settle_duration = self.tracker.S04_TARGET_S
                    self.tracker._advance_step(now)
            elif event_name == "inject_skip":
                if self.tracker.experiment_id in ["MOA-1", "MULTI-OBJ-1"]:
                    for s in self.tracker.steps:
                        if s.id == "S06":
                            s.status = "skipped"
                            break
                    self.tracker.moa1_water_consumed = False
                    self.tracker._trigger_alert(
                        step_id="S06",
                        severity="critical",
                        kind="skipped",
                        message="Step S06 skipped! Water bottle returned to table without drinking.",
                        tts="Warning: Step skipped. Drink water from the bottle before returning it.",
                        t=now,
                    )
                elif self.tracker.experiment_id in ["BCX-1", "BOX-COL-1"]:
                    for s in self.tracker.steps:
                        if s.id == "S03":
                            s.status = "skipped"
                            break
                    self.tracker.bcx1_red_inside = False
                    self.tracker.bcx1_yellow_inside = True
                    self.tracker._trigger_alert(
                        step_id="S03",
                        severity="critical",
                        kind="skipped",
                        message="Step S03 skipped! Red box was skipped and yellow box placed instead.",
                        tts="Warning: Step skipped. Place the red box before proceeding.",
                        t=now,
                    )
                else:
                    # Mark step S03 as skipped
                    for s in self.tracker.steps:
                        if s.id == "S03":
                            s.status = "skipped"
                            break
                    self.tracker._trigger_alert(
                        step_id="S03",
                        severity="caution",
                        kind="skipped",
                        message="Step S03 Skipped! Bottle returned to table without drinking.",
                        tts="Warning. Step three skipped. Drink water before returning the bottle.",
                        t=now,
                    )
            elif event_name in ["inject_out_of_order", "moa1_out_of_order"]:
                if self.tracker.experiment_id in ["MOA-1", "MULTI-OBJ-1"]:
                    self.tracker.moa1_phone_picked = True
                    self.tracker.moa1_is_seated = False
                    self.tracker.moa1_chair_pulled = False
                    cur_sid = self.tracker.steps[self.tracker.step_idx].id if self.tracker.step_idx < len(self.tracker.steps) else "S01"
                    self.tracker._trigger_alert(
                        step_id=cur_sid,
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order step detected! Pull chair and sit before picking up smartphone.",
                        tts="Warning: Step out of order. Pull chair and sit down before picking up smartphone.",
                        t=now,
                    )
                elif self.tracker.experiment_id in ["BCX-1", "BOX-COL-1"]:
                    for s in self.tracker.steps:
                        if s.id == "S03":
                            s.status = "skipped"
                            break
                    self.tracker.bcx1_red_inside = False
                    self.tracker.bcx1_yellow_inside = True
                    self.tracker._trigger_alert(
                        step_id="S03",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S03 requires placing Red Box first. Yellow box detected inside.",
                        tts="Warning: Step out of order. Place the Red Box inside the container first.",
                        t=now,
                    )
                else:
                    self.tracker._trigger_alert(
                        step_id="S01",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S01 requires resting bottle on table first to calibrate baseline.",
                        tts="Warning: Step out of order. Place bottle on table to calibrate baseline first.",
                        t=now,
                    )
            elif event_name == "inject_wrong_object":
                self.tracker._trigger_alert(
                    step_id="S02",
                    severity="caution",
                    kind="wrong_object",
                    message="Wrong object grasped! Confusable cup/mug detected instead of target bottle.",
                    tts="Wrong object grasped. That is the cup, not the water bottle.",
                    t=now,
                )
            elif event_name == "bcx1_collision":
                # Simulate BCX-1 collision event (Step S05)
                self.tracker.bcx1_container_locked = True
                self.tracker.bcx1_colors_verified = True
                self.tracker.bcx1_red_placed = True
                self.tracker.bcx1_yellow_placed = True
                self.tracker.bcx1_red_inside = True
                self.tracker.bcx1_yellow_inside = True
                self.tracker.bcx1_collision = True
                self.tracker.bcx1_distance_px = 0.0
                self.tracker.bcx1_collision_hold_duration = 0.8
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04", "S05"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S05",
                    severity="info",
                    kind="collision_confirmed",
                    message="Box Collision Confirmed! Red and Yellow boxes in active collision inside container.",
                    tts="Notice: Box collision confirmed. Contact verified inside container.",
                    t=now,
                )
            elif event_name == "bcx1_skip_red":
                # Simulate BCX-1 skipped red box (placing yellow first in S03)
                for s in self.tracker.steps:
                    if s.id == "S03":
                        s.status = "skipped"
                        break
                self.tracker.bcx1_red_inside = False
                self.tracker.bcx1_yellow_inside = True
                self.tracker._trigger_alert(
                    step_id="S03",
                    severity="critical",
                    kind="out_of_order",
                    message="Out of order! Step S03 requires placing Red Box first. Yellow box detected inside.",
                    tts="Warning: Step out of order. Place the Red Box inside the container first.",
                    t=now,
                )
            elif event_name == "bcx1_hazard_oob":
                # Simulate collision out of bounds hazard
                self.tracker.bcx1_collision = True
                self.tracker.bcx1_distance_px = 0.0
                self.tracker.bcx1_red_inside = False
                self.tracker.bcx1_yellow_inside = False
                self.tracker._trigger_alert(
                    step_id="S05",
                    severity="critical",
                    kind="hazard",
                    message="Out-of-Bounds Hazard! Collision must take place inside the container.",
                    tts="Hazard warning: Collision must occur inside the container.",
                    t=now,
                )
            elif event_name == "moa1_pull_chair":
                self.tracker.moa1_chair_pulled = True
                self.tracker.moa1_s01_hold_duration = 0.8
                for s in self.tracker.steps:
                    if s.id == "S01":
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S01",
                    severity="info",
                    kind="step_complete",
                    message="Chair pulled into position. Now sit down on the chair.",
                    tts="Chair positioned. Please sit down on the chair.",
                    t=now,
                )
            elif event_name == "moa1_sit":
                self.tracker.moa1_chair_pulled = True
                self.tracker.moa1_is_seated = True
                self.tracker.moa1_knee_angle_deg = 92.5
                self.tracker.moa1_s02_hold_duration = 0.8
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S02",
                    severity="info",
                    kind="step_complete",
                    message="Seated posture confirmed (knee angle 92.5°). Pick up the smartphone.",
                    tts="Seated posture confirmed. Pick up the smartphone.",
                    t=now,
                )
            elif event_name == "moa1_phone_pickup":
                self.tracker.moa1_is_seated = True
                self.tracker.moa1_phone_picked = True
                self.tracker.moa1_s03_hold_duration = 0.6
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S03",
                    severity="info",
                    kind="step_complete",
                    message="Smartphone picked up. Place it back down onto the desk surface.",
                    tts="Phone picked up. Return the smartphone to the desk.",
                    t=now,
                )
            elif event_name == "moa1_phone_stow":
                self.tracker.moa1_phone_picked = True
                self.tracker.moa1_phone_stowed = True
                self.tracker.moa1_s04_hold_duration = 0.6
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S04",
                    severity="info",
                    kind="step_complete",
                    message="Smartphone stowed on desk. Next, grasp and lift the water bottle.",
                    tts="Smartphone stowed. Grasp and lift the water bottle.",
                    t=now,
                )
            elif event_name == "moa1_bottle_lift":
                self.tracker.moa1_bottle_lifted = True
                self.tracker.moa1_s05_hold_duration = 0.6
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04", "S05"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S05",
                    severity="info",
                    kind="step_complete",
                    message="Water bottle lifted. Bring to mouth and drink water (hold >= 1.5s).",
                    tts="Bottle lifted. Bring to mouth and drink water.",
                    t=now,
                )
            elif event_name == "moa1_drink":
                self.tracker.moa1_bottle_lifted = True
                self.tracker.moa1_drinking_hold_duration = 1.5
                self.tracker.moa1_water_consumed = True
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S06",
                    severity="info",
                    kind="step_complete",
                    message="Drinking verified (held >= 1.5s). Return bottle to table and release hands.",
                    tts="Water consumed. Return bottle to table and release hands.",
                    t=now,
                )
            elif event_name == "moa1_bottle_return":
                self.tracker.moa1_bottle_returned = True
                self.tracker.protocol_complete = True
                self.tracker.moa1_s07_hold_duration = 0.6
                for s in self.tracker.steps:
                    s.status = "completed"
                    s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S07",
                    severity="info",
                    kind="nominal_completion",
                    message="Multi-Object Experiment Complete! All seven activities verified nominal.",
                    tts="Multi-object experiment complete. All seven steps verified nominal.",
                    t=now,
                )
            elif event_name == "moa1_skip_drink":
                for s in self.tracker.steps:
                    if s.id == "S06":
                        s.status = "skipped"
                        break
                self.tracker.moa1_water_consumed = False
                self.tracker._trigger_alert(
                    step_id="S06",
                    severity="critical",
                    kind="skipped",
                    message="Step S06 skipped! Water bottle returned to table without drinking.",
                    tts="Warning: Step skipped. Drink water from the bottle before returning it.",
                    t=now,
                )
            elif event_name == "reset":
                self.tracker.reset()

            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _, telem = self.tracker.process_frame(dummy)
            self.last_telemetry = telem
            return telem

    # --------------------------------------------------------------- copilot
    def copilot_action(self, kind: str) -> dict[str, Any]:
        from backend.app.groq.reasoning import experiment_state

        with self.lock:
            tel = dict(self.last_telemetry)
            spec = self.spec
        st = experiment_state(tel, spec, {"type": "crew_request", "request": kind})
        r = self.copilot.reasoner.review(st) if kind == "review" else self.copilot.reasoner.next_step(st)
        title = "Step review" if kind == "review" else "What to do now"
        msg = self.copilot.post("review" if kind == "review" else "guide", title, r.get("display", ""),
                                spoken=r.get("spoken"), source=r.get("source", "?"), latency_ms=r.get("latency_ms"),
                                severity=None if r.get("status") != "IMPROVEMENT_NEEDED" else "caution",
                                extra={"status": r.get("status"), "improvement": r.get("improvement"),
                                       "safety": r.get("safety")})
        return {**r, "message": msg}

    def status_line(self) -> dict[str, Any]:
        tel = self.last_telemetry or {}
        active = next((s for s in tel.get("steps", []) if s["status"] == "active"), None)
        done = sum(1 for s in tel.get("steps", []) if s["status"] == "completed")
        n = len(tel.get("steps", []))
        txt = (f"Step {active['id'][1:].lstrip('0')} of {n}: {active['prompt']}. {done} steps verified."
               if active else f"Procedure complete. {done} of {n} steps verified.")
        return {"text": txt}

    # ----------------------------------------------------------------- video
    def load_video_file(self, video_path: str | Path) -> dict[str, Any]:
        with self.lock:
            p = str(video_path)
            if not Path(p).exists():
                return {"error": f"Video file not found: {p}"}
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap = cv2.VideoCapture(p)
            if not self.video_cap.isOpened():
                return {"error": "OpenCV could not open this video (try MP4/H.264 or WebM)."}
            self.active_video_path = p
            total = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = float(self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if not (1.0 <= self.video_fps <= 240.0):
                self.video_fps = 25.0
            self.video_t0, self.video_pos0 = None, 0
            self.reset()
            return {"status": "loaded", "video_path": p, "total_frames": total, "fps": self.video_fps,
                    "duration_s": round(total / self.video_fps, 1) if self.video_fps else 0}

    def get_next_video_frame(self) -> dict[str, Any]:
        with self.lock:
            if self.video_cap is None or not self.video_cap.isOpened():
                return {"error": "No video opened"}
            now = time.time()
            if self.video_t0 is None:
                self.video_t0, self.video_pos0 = now, int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            # play in real time: skip frames the CPU could not process
            target = self.video_pos0 + int((now - self.video_t0) * self.video_fps)
            pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            while pos < target - 1:
                if not self.video_cap.grab():
                    break
                pos += 1
            ok, frame = self.video_cap.read()
            total = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not ok or frame is None:
                return {"is_eof": True, "total_frames": total}
            pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            vt = self.video_t0 + (pos - self.video_pos0) / self.video_fps
            small = cv2.resize(frame, (640, 480))
            tel = self.process_frame(small, vt)
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
            return {"frame": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"),
                    "telemetry": tel, "frame_pos": pos, "total_frames": total, "is_eof": False}

    def stop_video(self) -> dict[str, Any]:
        with self.lock:
            if self.video_cap is not None:
                self.video_cap.release()
            self.video_cap, self.active_video_path = None, None
            return {"status": "stopped"}

    # ------------------------------------------------------------ hardware
    def system_status(self) -> dict[str, Any]:
        from backend.app.groq.client import get_client

        return {
            "hardware": gpu_info(),
            "mesh_backends": {
                "pose3d": pose_backend_status(self.pipeline.pose.available, self.pipeline.pose.error),
                "sam3d": sam3d_status(),
            },
            "active_mesh_backend": self.pipeline.mesh_backend,
            "groq": get_client().status(),
            "detector": "YOLOv8n (ONNX Runtime, CPU)" if str(getattr(self.yolo.det_model, "ckpt_path", "") or "").endswith(".onnx")
            or "onnx" in str(getattr(self.yolo.det_model, "model", "")) else "YOLOv8n (PyTorch, CPU)",
        }

    def set_mesh_backend(self, backend: str) -> dict[str, Any]:
        if backend == "sam3d":
            st = sam3d_status()
            if not st["available"]:
                return {"error": "MODEL UNAVAILABLE", **st}
            return {"error": "SAM 3D Body adapter is installed but not wired on this build", **st}
        self.pipeline.mesh_backend = "pose3d"
        return {"active_mesh_backend": "pose3d"}


_service: TrackerService | None = None


def get_tracker_service() -> TrackerService:
    global _service
    if _service is None:
        _service = TrackerService("WBP-1")
    return _service

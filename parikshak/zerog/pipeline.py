"""Per-frame zero-g pipeline.

    raw camera frame
      -> AprilTag rack localization            (rack.py)   T_rack_camera, camera roll
      -> 3D pose + orientation search          (pose3d.py) 33 joints, camera frame
      -> upright "view" for procedure trackers (orient.py) rack-aligned if tags are seen
      -> [procedure tracker runs on the view: YOLO objects, steps, alerts]
      -> objects in 3D, interaction graph, temporal HAR (har.py)
      -> body inclination + pose-driven mesh   (body.py)
      -> overlay primitives in RAW camera pixels + a 3D rack-world payload

Everything is reported in the rack frame when the rack is LOCKED/HOLD, otherwise
in the camera frame - and the payload says which.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from parikshak.zerog.body import body_metrics, build_mesh, hand_centres, lift_coco_2d, mesh_overlay, project
from parikshak.zerog.config import CameraModel, camera_model
from parikshak.zerog.har import InteractionGraph, TemporalHAR, object_cam_position
from parikshak.zerog.orient import ViewTransform, quarter_turns_to_upright
from parikshak.zerog.pose3d import BONES, COCO_FROM_BLAZE, PoseEstimator, PoseObservation
from parikshak.zerog.rack import RackLocalizer, RackState


def _r(v, nd=3) -> list[float]:
    return [round(float(x), nd) for x in np.asarray(v).ravel()]


@dataclass
class FrameCtx:
    t: float
    raw: np.ndarray
    cam: CameraModel
    rack: RackState
    vt: ViewTransform | None = None
    view: np.ndarray | None = None
    view_source: str = "none"
    obs: PoseObservation | None = None
    pose_source: str = "none"
    timings: dict[str, float] = field(default_factory=dict)
    early: Any = None


class ZeroGPipeline:
    def __init__(self) -> None:
        self.pose = PoseEstimator()
        self.rack = RackLocalizer()
        self.graph = InteractionGraph()
        self.har = TemporalHAR()
        self.body_k = 0
        self.mesh_backend = "pose3d"
        self.rotation_test: dict[str, Any] = {"active": False, "samples": [], "started_at": None}
        self._frame_times: deque[float] = deque(maxlen=20)
        self.last: dict[str, Any] = {}

    def reset(self) -> None:
        self.har.reset()
        self.graph = InteractionGraph()
        self.pose.reset()

    # ------------------------------------------------------------- stage 1
    def begin(self, raw: np.ndarray, t: float, on_predicted_view=None) -> FrameCtx:
        """on_predicted_view(view, k) lets the caller start object detection on
        the most likely upright view while the pose model runs (both release the
        GIL); ctx.early holds (k, whatever it returned)."""
        h, w = raw.shape[:2]
        cam = camera_model(w, h)
        t0 = time.perf_counter()
        rack = self.rack.update(raw, cam, t)
        ctx = FrameCtx(t=t, raw=raw, cam=cam, rack=rack)
        ctx.timings["rack_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        prefer = None
        if rack.valid and rack.camera_roll_deg is not None:
            prefer = quarter_turns_to_upright(rack.camera_roll_deg)
        ctx.early = None
        if on_predicted_view is not None:
            k_pred = prefer if prefer is not None else self.body_k
            vt_pred = ViewTransform(k_pred, w, h)
            ctx.early = (k_pred, vt_pred, on_predicted_view(vt_pred.apply(raw), k_pred))
        t0 = time.perf_counter()
        obs = self.pose.estimate(raw, cam, t, prefer_k=prefer if prefer is not None else self.body_k)
        ctx.timings["pose_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ctx.obs = obs
        if obs is not None:
            ctx.pose_source = "blazepose"
            self.body_k = obs.k

        if rack.valid and prefer is not None:
            k, ctx.view_source = prefer, "rack"
        elif obs is not None:
            k, ctx.view_source = obs.k, "body"
        else:
            k, ctx.view_source = self.body_k, "body (last)"
        if ctx.early is not None and ctx.early[0] == k:
            ctx.vt = ctx.early[1]
        else:
            ctx.vt = ViewTransform(k, w, h)
        ctx.view = ctx.vt.apply(raw)
        return ctx

    def coco17_for_view(self, ctx: FrameCtx) -> np.ndarray | None:
        if ctx.obs is None:
            return None
        pv = ctx.vt.raw_to_view(ctx.obs.px)
        return PoseEstimator.to_coco17(ctx.obs, pv)

    # ------------------------------------------------------------- stage 2
    def finish(self, ctx: FrameCtx, tracker_skeleton_view: list[dict[str, Any]] | None,
               overlay_objects_view: list[dict[str, Any]], expected_classes: list[str] | None,
               toggles: dict[str, bool]) -> dict[str, Any]:
        t, cam, rack, vt = ctx.t, ctx.cam, ctx.rack, ctx.vt
        t0 = time.perf_counter()

        obs = ctx.obs
        flat = False
        if obs is None and tracker_skeleton_view:
            kp = np.array([[k.get("x", 0.0), k.get("y", 0.0), k.get("conf", 0.0)] for k in tracker_skeleton_view])
            if kp.shape[0] >= 17:
                raw_xy = vt.view_to_raw(kp[:, :2])
                obs = lift_coco_2d(np.column_stack([raw_xy, kp[:, 2]]), cam)
                if obs is not None:
                    flat = True
                    ctx.pose_source = "yolo-pose (2D, flat depth)"
        person = obs is not None

        body = body_metrics(obs, rack) if person else {"available": False}
        parts = build_mesh(obs) if (person and obs.cam is not None) else []
        mesh_px = mesh_overlay(parts, cam) if parts else []

        rack_ok = rack.valid
        frame_name = "rack" if rack_ok else "camera"

        def to_world(p_cam: np.ndarray) -> np.ndarray:
            """Display frame for 3D: rack (x right, y up, z out) or camera converted to y-up."""
            p = np.asarray(p_cam, dtype=np.float64)
            if rack_ok:
                return rack.cam_to_rack(p)
            return p * np.array([1.0, -1.0, -1.0]) if p.ndim == 1 else p * np.array([[1.0, -1.0, -1.0]])

        # ---- objects in 3D
        hands_px: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        body_nodes: dict[str, np.ndarray] = {}
        if person and obs.cam is not None:
            hc = hand_centres(obs.cam)
            hp = {"left_hand": obs.px[[15, 17, 19]].mean(0), "right_hand": obs.px[[16, 18, 20]].mean(0)}
            for side in ("left_hand", "right_hand"):
                if obs.vis[15 if side == "left_hand" else 16] >= 0.4:
                    hands_px[side] = (hp[side], hc[side])
                    body_nodes[side] = to_world(hc[side])
            body_nodes["head"] = to_world(obs.cam[[7, 8]].mean(0) if min(obs.vis[7], obs.vis[8]) > 0.3 else obs.cam[0])
            body_nodes["mouth"] = to_world(obs.cam[[9, 10]].mean(0))
            body_nodes["torso"] = to_world(obs.cam[[11, 12]].mean(0))
            for side, idx in (("left_foot", 31), ("right_foot", 32)):
                if obs.vis[idx] >= 0.5:
                    body_nodes[side] = to_world(obs.cam[idx])

        objects = []
        objects_px = []
        for i, o in enumerate(overlay_objects_view):
            box_raw = vt.view_box_to_raw(tuple(o["box"]))
            p_cam, how = object_cam_position(box_raw, o.get("cls", ""), cam, hands_px)
            # top-centre of the object in the upright view (the bottle's mouth end)
            vx1, vy1, vx2, _ = o["box"]
            top_raw = vt.view_to_raw(np.array([[(vx1 + vx2) / 2.0, vy1]]))[0]
            top_cam = np.array([(top_raw[0] - cam.cx) * p_cam[2] / cam.fx, (top_raw[1] - cam.cy) * p_cam[2] / cam.fy, p_cam[2]])
            oid = f'{o.get("cls", "obj")}#{i}'
            objects.append({"id": oid, "cls": o.get("cls", ""), "label": o.get("label", ""),
                            "pos": to_world(p_cam), "top": to_world(top_cam), "placement": how, "state": o.get("state", []),
                            "color": o.get("color", "#22d3ee"),
                            "size_m": round(float(max(box_raw[2] - box_raw[0], box_raw[3] - box_raw[1]) * p_cam[2] / cam.fx), 3)})
            objects_px.append({"id": oid, "label": o.get("label", ""), "box": _r(box_raw, 1), "state": o.get("state", []),
                               "color": o.get("color", "#22d3ee"), "conf": o.get("conf")})

        graph = self.graph.build(body_nodes, objects, t)
        har = self.har.update(t, person, graph, expected_classes)

        # ---- overlay primitives (raw pixels)
        overlay: dict[str, Any] = {"width": cam.width, "height": cam.height}
        if person:
            overlay["skeleton"] = {"points": [[round(float(x), 1), round(float(y), 1), round(float(v), 2)]
                                              for (x, y), v in zip(obs.px, obs.vis)], "bones": BONES}
        overlay["mesh"] = mesh_px
        overlay["objects"] = objects_px
        if rack.tag_corners_px:
            overlay["tags"] = rack.tag_corners_px
        if rack_ok:
            axes = {}
            o3 = rack.rack_to_cam(np.zeros(3))
            for name, v in (("x", [0.15, 0, 0]), ("y", [0, 0.15, 0]), ("z", [0, 0, 0.15])):
                uv = project(cam, np.vstack([o3, rack.rack_to_cam(np.array(v, dtype=float))]))
                if uv is not None:
                    axes[name] = _r(uv.ravel(), 1)
            overlay["rack_axes"] = axes
        inter = []
        for e in graph["edges"]:
            if e["from"] not in hands_px:
                continue
            obj = next((op for op in objects_px if op["id"] == e["to"]), None)
            if obj is None or e["distance_m"] > 0.6:
                continue
            b = obj["box"]
            inter.append({"from": _r(hands_px[e["from"]][0], 1), "to": [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)],
                          "label": f'{e["distance_m"] * 100:.0f} cm', "contact": e["contact_prob"]})
        overlay["interactions"] = inter

        # ---- 3D world payload
        world: dict[str, Any] = {"frame": frame_name, "objects": [], "joints": None, "segments": [], "edges": []}
        if person and obs.cam is not None:
            J = to_world(obs.cam)
            world["joints"] = [_r(p) for p in J]
            world["vis"] = [round(float(v), 2) for v in obs.vis]
            world["bones"] = BONES
            world["segments"] = [{"name": p["name"], "kind": p["kind"], "a": _r(to_world(p["a"])), "b": _r(to_world(p["b"])),
                                  "r": round(float(p["r"]), 3), "inferred": bool(p["vis"] < 0.5)} for p in parts]
        world["objects"] = [{"id": o["id"], "cls": o["cls"], "label": o["label"], "pos": _r(o["pos"]),
                             "size": o["size_m"], "color": o["color"], "state": o["state"]} for o in objects]
        world["edges"] = [e for e in graph["edges"] if e["distance_m"] < 0.8]
        world["nodes"] = graph["nodes"]
        if rack_ok:
            world["tags"] = [{"id": i, "pos": _r(c), "size": self.rack.tag_size_m}
                             for i, c in self.rack.layout.tags.items()]
            world["camera"] = rack.camera_pose_in_rack()
        else:
            world["camera"] = {"position": [0, 0, 0], "forward": [0, 0, -1], "up": [0, 1, 0], "right": [1, 0, 0]}

        # ---- status (never fabricated)
        hands_ok = person and not flat and (obs.vis[15] >= 0.5 or obs.vis[16] >= 0.5)
        status = {
            "astronaut": "DETECTED" if person else "NOT DETECTED",
            "pose_source": ctx.pose_source,
            "mesh": ("VALID" if (parts and not flat) else "FLAT (2D-lifted)" if parts else "LOST"),
            "mesh_backend": "pose3d",
            "joints_visible": int((obs.vis >= 0.5).sum()) if person else 0,
            "joints_total": 33,
            "hands": "DETECTED" if hands_ok else "NOT DETECTED",
            "rack_frame": rack.status,
            "frame": frame_name,
            "pnp_reproj_px": None if (not person or obs.pnp_reproj_px is None) else round(obs.pnp_reproj_px, 1),
            "uncertain": (not person) or har["status"] == "UNCERTAIN",
        }
        ctx.timings["har_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        self._frame_times.append(t)
        fps = 0.0
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0

        out = {
            "status": status,
            "rack": self.rack.to_dict(rack),
            "view": {"k": vt.k, "source": ctx.view_source, "rotation_deg": vt.k * 90,
                     "camera_roll_deg": rack.camera_roll_deg},
            "body": body,
            "har": har,
            "interaction_graph": graph,
            "overlay": overlay,
            "world": world,
            "perf": {**ctx.timings, "fps": round(fps, 2), "pose_tries": obs.tries if (person and not flat) else 0},
        }
        if self.rotation_test["active"]:
            self._log_rotation(out, t)
        out["rotation_test"] = {"active": self.rotation_test["active"],
                                "n": len(self.rotation_test["samples"])}
        self.last = out
        return out

    # --------------------------------------------------------- rotation test
    def start_rotation_test(self) -> dict[str, Any]:
        self.rotation_test = {"active": True, "samples": [], "started_at": time.time()}
        return {"status": "recording"}

    def stop_rotation_test(self) -> dict[str, Any]:
        self.rotation_test["active"] = False
        return self.rotation_report()

    def _log_rotation(self, out: dict[str, Any], t: float) -> None:
        b, r = out["body"], out["rack"]
        self.rotation_test["samples"].append({
            "t": round(t - (self.rotation_test["started_at"] or t), 2),
            "rack": r["status"],
            "camera_roll_deg": r["camera_roll_deg"],
            "view_k": out["view"]["k"],
            "inclination_rack_deg": b.get("inclination_deg") if b.get("frame") == "rack" else None,
            "inclination_camera_deg": b.get("camera_inclination_deg"),
            "image_lean_deg": b.get("image_lean_deg"),
            "pelvis_rack_m": b.get("pelvis_rack_m"),
            "pelvis_camera_m": b.get("pelvis_camera_m"),
            "activity": out["har"]["activity"],
            "confidence": out["har"]["confidence"],
            "astronaut": out["status"]["astronaut"],
        })
        if len(self.rotation_test["samples"]) > 3000:
            self.rotation_test["samples"] = self.rotation_test["samples"][-3000:]

    def rotation_report(self) -> dict[str, Any]:
        s = self.rotation_test["samples"]
        rolls = [x["camera_roll_deg"] for x in s if x["camera_roll_deg"] is not None]
        inc_r = [x["inclination_rack_deg"] for x in s if x["inclination_rack_deg"] is not None]
        inc_c = [x["inclination_camera_deg"] for x in s if x["inclination_camera_deg"] is not None]
        acts = [x["activity"] for x in s]

        def spread(v, circular: bool = False):
            if len(v) < 2:
                return None
            a = np.asarray(v, dtype=np.float64)
            if circular:  # -180/+180 are the same direction: unwrap before measuring the sweep
                a = np.degrees(np.unwrap(np.radians(a)))
            return round(float(np.percentile(a, 95) - np.percentile(a, 5)), 1)

        lean = [x["image_lean_deg"] for x in s if x["image_lean_deg"] is not None]
        return {
            "active": self.rotation_test["active"],
            "samples": s[-600:],
            "summary": {
                "n": len(s),
                "rack_locked_pct": round(100 * sum(1 for x in s if x["rack"] in ("LOCKED", "HOLD")) / len(s), 1) if s else 0,
                "camera_roll_range_deg": spread(rolls, circular=True),
                "inclination_rack_spread_deg": spread(inc_r),
                "inclination_camera_spread_deg": spread(inc_c),
                "image_lean_spread_deg": spread(lean, circular=True),
                "activity_changes": sum(1 for a, b in zip(acts, acts[1:]) if a != b),
                "note": "All values are measured; with the rack locked, a stable rack inclination while the "
                        "camera roll sweeps shows the rack-relative representation is invariant to camera rotation.",
            },
        }

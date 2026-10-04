"""Body metrics (inclination, head direction, hands) and the pose-driven body mesh.

The mesh is built on the measured 3D joints: capsules for limbs, an elliptical
torso, an ellipsoid head. It is projected back through the camera intrinsics so
the overlay lands on the person, and the same segments are sent to the 3D rack
world. It is labelled for what it is - a pose-driven proxy mesh from BlazePose
GHUM joints - not SAM 3D Body output.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np

from parikshak.zerog.config import CameraModel
from parikshak.zerog.orient import image_up_angle_deg
from parikshak.zerog.pose3d import PoseObservation

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
NOSE, L_EAR, R_EAR, M_L, M_R = 0, 7, 8, 9, 10
L_IDX, R_IDX, L_PNK, R_PNK = 19, 20, 17, 18
L_HEEL, R_HEEL, L_FT, R_FT = 29, 30, 31, 32


def _n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _mid(P: np.ndarray, a: int, b: int) -> np.ndarray:
    return (P[a] + P[b]) / 2.0


def posture_class(incl_deg: float) -> str:
    if incl_deg < 20:
        return "UPRIGHT"
    if incl_deg < 60:
        return "LEANING"
    if incl_deg < 120:
        return "HORIZONTAL"
    return "INVERTED"


def _head_direction(v: np.ndarray, axes: dict[str, np.ndarray]) -> str:
    best, name = -2.0, "?"
    for k, a in axes.items():
        d = float(np.dot(v, a))
        if d > best:
            best, name = d, k
    return name


RACK_AXES = {
    "UP (+Y rack)": np.array([0, 1.0, 0]), "DOWN (-Y rack)": np.array([0, -1.0, 0]),
    "RIGHT (+X rack)": np.array([1.0, 0, 0]), "LEFT (-X rack)": np.array([-1.0, 0, 0]),
    "TOWARD CREW (+Z)": np.array([0, 0, 1.0]), "INTO RACK (-Z)": np.array([0, 0, -1.0]),
}
CAM_AXES = {
    "UP (image)": np.array([0, -1.0, 0]), "DOWN (image)": np.array([0, 1.0, 0]),
    "RIGHT (image)": np.array([1.0, 0, 0]), "LEFT (image)": np.array([-1.0, 0, 0]),
    "TOWARD CAMERA": np.array([0, 0, -1.0]), "AWAY FROM CAMERA": np.array([0, 0, 1.0]),
}


def body_metrics(obs: PoseObservation, rack) -> dict[str, Any]:
    """Inclination of the body's head-up axis in the rack frame (and camera frame)."""
    P, V, px = obs.cam, obs.vis, obs.px
    out: dict[str, Any] = {"available": P is not None}
    if P is None:
        return out
    hips_seen = V[L_HIP] >= 0.5 and V[R_HIP] >= 0.5
    # BlazePose regresses hip positions from its GHUM body prior even when the
    # hips are hidden by a desk (it is hip-centred). That torso axis is far
    # steadier than the neck, which leans forward ~20 deg on a seated person.
    hips_inferred = (not hips_seen) and V[L_HIP] >= 0.05 and V[R_HIP] >= 0.05 and not getattr(obs, "flat", False)
    sh = _mid(P, L_SH, R_SH)
    if hips_seen or hips_inferred:
        up_cam = _n(sh - _mid(P, L_HIP, R_HIP))
        up_px = px[[L_SH, R_SH]].mean(0) - px[[L_HIP, R_HIP]].mean(0)
        basis = "hip->shoulder (torso)" if hips_seen else "hip->shoulder (hips estimated by BlazePose)"
    else:
        head = _mid(P, L_EAR, R_EAR) if (V[L_EAR] >= 0.4 and V[R_EAR] >= 0.4) else P[NOSE]
        up_cam = _n(head - sh)
        up_px = (px[[L_EAR, R_EAR]].mean(0) if V[L_EAR] >= 0.4 and V[R_EAR] >= 0.4 else px[NOSE]) - px[[L_SH, R_SH]].mean(0)
        basis = "shoulder->head (hips not visible)"
    cam_incl = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(up_cam, [0, -1.0, 0]))))))
    out.update({
        "axis_basis": basis,
        "image_lean_deg": round(image_up_angle_deg(up_px), 1),
        "camera_inclination_deg": round(cam_incl, 1),
        "camera_head_direction": _head_direction(up_cam, CAM_AXES),
        "up_vector_camera": [round(float(x), 3) for x in up_cam],
        "shoulder_width_m": round(float(np.linalg.norm(P[L_SH] - P[R_SH])), 3),
        "pelvis_camera_m": [round(float(x), 3) for x in _mid(P, L_HIP, R_HIP)],
    })
    # "inclination_deg" is measured in the frontal plane (rack XY, or the image
    # plane without a rack): it is what a single camera measures reliably. The
    # full 3D angle includes forward/back lean, whose depth component is noisy.
    if rack is not None and rack.valid:
        up_r = _n(rack.dir_cam_to_rack(up_cam))
        frontal = math.degrees(math.atan2(float(up_r[0]), float(up_r[1])))
        out.update({
            "frame": "rack",
            "inclination_deg": round(abs(frontal), 1),
            "inclination_3d_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(up_r[1]))))), 1),
            "lateral_lean_deg": round(frontal, 1),
            "sagittal_lean_deg": round(math.degrees(math.atan2(float(up_r[2]), float(up_r[1]))), 1),
            "head_direction": _head_direction(up_r, RACK_AXES),
            "up_vector_rack": [round(float(x), 3) for x in up_r],
            "pelvis_rack_m": [round(float(x), 3) for x in rack.cam_to_rack(_mid(P, L_HIP, R_HIP))],
        })
    else:
        lean = image_up_angle_deg(up_px)
        out.update({
            "frame": "camera",
            "inclination_deg": round(abs(lean), 1),
            "inclination_3d_deg": round(cam_incl, 1),
            "lateral_lean_deg": round(lean, 1),
            "sagittal_lean_deg": None,
            "head_direction": _head_direction(up_cam, CAM_AXES),
            "up_vector_rack": None,
            "pelvis_rack_m": None,
        })
    out["posture"] = posture_class(float(out["inclination_deg"]))
    out["hands_visible"] = {
        "left": bool(V[L_WR] >= 0.5), "right": bool(V[R_WR] >= 0.5),
    }
    return out


def hand_centres(P: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "left_hand": (P[L_WR] + P[L_IDX] + P[L_PNK]) / 3.0,
        "right_hand": (P[R_WR] + P[R_IDX] + P[R_PNK]) / 3.0,
    }


# ----------------------------------------------------------------------- mesh
def _basis(axis: np.ndarray, hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    a = _n(axis)
    h = hint if hint is not None and abs(float(np.dot(_n(hint), a))) < 0.95 else (
        np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0]))
    u = _n(h - np.dot(h, a) * a)
    v = np.cross(a, u)
    return u, v


_TH = np.linspace(0, 2 * np.pi, 12, endpoint=False)


def _capsule(A: np.ndarray, B: np.ndarray, r: float) -> list[np.ndarray]:
    d = B - A
    if np.linalg.norm(d) < 1e-6:
        d = np.array([0, 1e-3, 0])
    u, v = _basis(d)
    an = _n(d)
    rings = []
    for s, rr, off in ((0.0, 0.7, -0.6), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.7, 0.6)):
        c = A + s * d + off * r * an
        rings.append(np.array([c + rr * r * (math.cos(t) * u + math.sin(t) * v) for t in _TH]))
    return rings


def _ellipsoid(C: np.ndarray, up: np.ndarray, side: np.ndarray, rx: float, ry: float, rz: float) -> list[np.ndarray]:
    a = _n(up)
    u = _n(side - np.dot(side, a) * a)
    v = np.cross(a, u)
    rings = []
    for lat in (-0.75, -0.4, 0.0, 0.4, 0.75):
        c = C + a * ry * lat
        k = math.sqrt(max(0.0, 1 - lat * lat))
        rings.append(np.array([c + k * (rx * math.cos(t) * u + rz * math.sin(t) * v) for t in _TH]))
    return rings


def _torso(P: np.ndarray) -> list[np.ndarray]:
    sh, hp = _mid(P, L_SH, R_SH), _mid(P, L_HIP, R_HIP)
    up = _n(sh - hp)
    side_sh = P[L_SH] - P[R_SH]
    side_hp = P[L_HIP] - P[R_HIP]
    sw, hw = float(np.linalg.norm(side_sh)), float(np.linalg.norm(side_hp))
    u = _n(side_sh - np.dot(side_sh, up) * up)
    v = np.cross(up, u)
    rings = []
    for s, half_w, half_d in ((1.05, 0.40 * sw, 0.16 * sw), (1.0, 0.52 * sw, 0.24 * sw),
                              (0.5, 0.45 * max(sw, hw), 0.22 * sw), (0.0, 0.58 * hw, 0.25 * sw),
                              (-0.05, 0.50 * hw, 0.20 * sw)):
        c = hp + s * (sh - hp)
        rings.append(np.array([c + half_w * math.cos(t) * u + half_d * math.sin(t) * v for t in _TH]))
    return rings


SEGMENTS = [
    # name, joint a, joint b, radius as fraction of shoulder width, kind
    ("upper_arm_l", L_SH, L_EL, 0.15, "limb"), ("forearm_l", L_EL, L_WR, 0.11, "limb"),
    ("upper_arm_r", R_SH, R_EL, 0.15, "limb"), ("forearm_r", R_EL, R_WR, 0.11, "limb"),
    ("thigh_l", L_HIP, L_KN, 0.20, "limb"), ("shin_l", L_KN, L_AN, 0.14, "limb"),
    ("thigh_r", R_HIP, R_KN, 0.20, "limb"), ("shin_r", R_KN, R_AN, 0.14, "limb"),
    ("foot_l", L_HEEL, L_FT, 0.07, "limb"), ("foot_r", R_HEEL, R_FT, 0.07, "limb"),
]


def build_mesh(obs: PoseObservation) -> list[dict[str, Any]]:
    """3D mesh rings per body part, in camera coordinates."""
    P, V = obs.cam, obs.vis
    if P is None:
        return []
    sw = max(0.25, min(0.55, float(np.linalg.norm(P[L_SH] - P[R_SH]))))
    parts: list[dict[str, Any]] = []

    def vis(*idx: int) -> float:
        return float(min(V[i] for i in idx))

    parts.append({"name": "torso", "rings": _torso(P), "vis": vis(L_SH, R_SH), "a": _mid(P, L_HIP, R_HIP),
                  "b": _mid(P, L_SH, R_SH), "r": 0.5 * sw, "kind": "torso"})
    ears_ok = V[L_EAR] >= 0.3 and V[R_EAR] >= 0.3
    head_c = _mid(P, L_EAR, R_EAR) if ears_ok else P[NOSE] + 0.04 * _n(_mid(P, L_SH, R_SH) - P[NOSE])
    up = _n(head_c - _mid(P, L_SH, R_SH))
    side = P[L_EAR] - P[R_EAR] if ears_ok else P[L_SH] - P[R_SH]
    head_c = head_c + up * 0.03
    rh = max(0.075, min(0.11, 0.30 * sw))
    parts.append({"name": "head", "rings": _ellipsoid(head_c, up, side, rh * 0.9, rh * 1.2, rh),
                  "vis": vis(NOSE), "a": head_c - up * rh, "b": head_c + up * rh, "r": rh, "kind": "head"})
    neck_a = _mid(P, L_SH, R_SH)
    parts.append({"name": "neck", "rings": _capsule(neck_a, head_c - up * rh * 0.9, 0.12 * sw),
                  "vis": vis(L_SH, R_SH, NOSE), "a": neck_a, "b": head_c - up * rh * 0.9, "r": 0.12 * sw, "kind": "limb"})
    for name, a, b, rf, kind in SEGMENTS:
        r = rf * sw
        parts.append({"name": name, "rings": _capsule(P[a], P[b], r), "vis": vis(a, b),
                      "a": P[a], "b": P[b], "r": r, "kind": kind})
    for side_name, w, i, p in (("hand_l", L_WR, L_IDX, L_PNK), ("hand_r", R_WR, R_IDX, R_PNK)):
        tip = (P[i] + P[p]) / 2.0
        parts.append({"name": side_name, "rings": _capsule(P[w], tip, 0.045), "vis": vis(w, i),
                      "a": P[w], "b": tip, "r": 0.045, "kind": "hand"})
    return parts


def project(cam: CameraModel, pts: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if np.any(pts[:, 2] <= 0.05):
        return None
    uv = (cam.K @ (pts / pts[:, 2:3]).T).T[:, :2]
    return uv


def mesh_overlay(parts: list[dict[str, Any]], cam: CameraModel) -> list[dict[str, Any]]:
    """Projected silhouettes + wireframe rings in raw-camera pixels, far-to-near."""
    out = []
    for p in parts:
        allpts = np.vstack(p["rings"])
        uv = project(cam, allpts)
        if uv is None:
            continue
        hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
        ring_px = []
        n = len(_TH)
        for i in range(len(p["rings"])):
            ring_px.append([[round(float(x), 1), round(float(y), 1)] for x, y in uv[i * n:(i + 1) * n]])
        # longitudinal lines through ring points 0, 3, 6, 9
        longs = [[[round(float(uv[i * n + j][0]), 1), round(float(uv[i * n + j][1]), 1)]
                  for i in range(len(p["rings"]))] for j in (0, 3, 6, 9)]
        out.append({
            "name": p["name"],
            "kind": p["kind"],
            "depth": round(float(allpts[:, 2].mean()), 3),
            "inferred": bool(p["vis"] < 0.5),
            "hull": [[round(float(x), 1), round(float(y), 1)] for x, y in hull],
            "rings": ring_px,
            "longs": longs,
        })
    out.sort(key=lambda s: -s["depth"])
    return out


def lift_coco_2d(kp17: np.ndarray, cam: CameraModel) -> PoseObservation | None:
    """Fallback 3D from 2D COCO keypoints when BlazePose finds nobody: all joints
    at one depth estimated from shoulder width. Flat - and labelled as such."""
    from parikshak.zerog.pose3d import COCO_FROM_BLAZE

    if kp17 is None or kp17.shape[0] < 17:
        return None
    if kp17[5, 2] < 0.3 or kp17[6, 2] < 0.3:
        return None
    sw_px = float(np.linalg.norm(kp17[5, :2] - kp17[6, :2]))
    if sw_px < 20:
        return None
    z = cam.fx * 0.38 / sw_px
    px = np.zeros((33, 2))
    vis = np.zeros(33)
    for j, b in enumerate(COCO_FROM_BLAZE):
        px[b] = kp17[j, :2]
        vis[b] = kp17[j, 2]
    # hands/feet sub-points borrow the wrist/ankle
    for w, subs in ((15, (17, 19, 21)), (16, (18, 20, 22)), (27, (29, 31)), (28, (30, 32))):
        for s in subs:
            px[s], vis[s] = px[w], vis[w] * 0.8
    for s, src in ((9, 0), (10, 0), (1, 2), (3, 2), (4, 5), (6, 5)):
        px[s], vis[s] = px[src], vis[src] * 0.8
    cam_pts = np.column_stack([(px[:, 0] - cam.cx) * z / cam.fx, (px[:, 1] - cam.cy) * z / cam.fy, np.full(33, z)])
    obs = PoseObservation(k=0, px=px, vis=vis, world=cam_pts - cam_pts[[23, 24]].mean(0), cam=cam_pts)
    obs.flat = True  # type: ignore[attr-defined]
    return obs

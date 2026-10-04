"""AprilTag rack localization: tag36h11 detection -> PnP -> T_rack_camera.

The rack is the reference frame ("rack-as-gravity"):
    +X across the rack face, +Y up the rack face, +Z out of the face toward the crew.

Tag corner positions come from configs/rack.yaml. One visible tag is enough to
recover the full 6-DoF pose; more tags make it steadier. If no tag is seen the
last pose is held for a short grace period and then reported LOST - it is never
invented.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np

from parikshak.zerog.config import CameraModel, RackLayout, rack_layout
from parikshak.zerog.orient import image_up_angle_deg

HOLD_S = 0.6  # keep the last pose this long when tags drop out (hand occlusion)


@dataclass
class RackState:
    status: str = "LOST"  # LOCKED | HOLD | LOST
    R: np.ndarray | None = None  # rack -> camera rotation
    t: np.ndarray | None = None  # rack -> camera translation (m)
    tags_visible: list[int] = field(default_factory=list)
    tag_corners_px: dict[int, list[list[float]]] = field(default_factory=dict)
    reproj_px: float | None = None
    camera_roll_deg: float | None = None  # rack +Y in the image, clockwise from image-up
    latency_ms: float = 0.0
    locked_since: float | None = None

    @property
    def valid(self) -> bool:
        return self.status in ("LOCKED", "HOLD") and self.R is not None

    def cam_to_rack(self, p_cam: np.ndarray) -> np.ndarray:
        """P_rack = R^T (P_cam - t), for one point or an Nx3 array."""
        p = np.asarray(p_cam, dtype=np.float64)
        return (p - self.t.reshape(1, 3)) @ self.R if p.ndim == 2 else self.R.T @ (p - self.t)

    def dir_cam_to_rack(self, v_cam: np.ndarray) -> np.ndarray:
        return self.R.T @ np.asarray(v_cam, dtype=np.float64)

    def rack_to_cam(self, p_rack: np.ndarray) -> np.ndarray:
        p = np.asarray(p_rack, dtype=np.float64)
        return p @ self.R.T + self.t.reshape(1, 3) if p.ndim == 2 else self.R @ p + self.t

    def camera_pose_in_rack(self) -> dict[str, list[float]] | None:
        if not self.valid:
            return None
        pos = -self.R.T @ self.t
        forward = self.R.T @ np.array([0.0, 0.0, 1.0])
        up = self.R.T @ np.array([0.0, -1.0, 0.0])
        right = self.R.T @ np.array([1.0, 0.0, 0.0])
        return {
            "position": _r(pos),
            "forward": _r(forward),
            "up": _r(up),
            "right": _r(right),
        }


def _r(v: np.ndarray, nd: int = 3) -> list[float]:
    return [round(float(x), nd) for x in np.asarray(v).ravel()]


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class RackLocalizer:
    def __init__(self, layout: RackLayout | None = None) -> None:
        self.layout = layout or rack_layout()
        self.tag_size_m = self.layout.tag_size_m
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.adaptiveThreshWinSizeMax = 33
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.state = RackState()
        self._last_seen = 0.0

    def set_tag_size(self, size_m: float) -> None:
        if 0.01 <= size_m <= 1.0:
            self.tag_size_m = float(size_m)

    def update(self, frame_bgr: np.ndarray, cam: CameraModel, t: float | None = None) -> RackState:
        t = time.time() if t is None else t
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        obs: list[tuple[int, np.ndarray]] = []
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if int(i) in self.layout.tags:
                    obs.append((int(i), c.reshape(4, 2).astype(np.float64)))

        prev = self.state
        st = RackState(locked_since=prev.locked_since)
        st.tags_visible = sorted(i for i, _ in obs)
        st.tag_corners_px = {i: [[round(float(x), 1), round(float(y), 1)] for x, y in c] for i, c in obs}

        pose = self._solve(obs, cam, prev) if obs else None
        if pose is not None:
            R, tvec, err = pose
            st.status, st.R, st.t, st.reproj_px = "LOCKED", R, tvec, err
            if prev.status != "LOCKED" or st.locked_since is None:
                st.locked_since = t
            self._last_seen = t
        elif prev.R is not None and (t - self._last_seen) <= HOLD_S:
            st.status, st.R, st.t, st.reproj_px = "HOLD", prev.R, prev.t, prev.reproj_px
        else:
            st.status, st.locked_since = "LOST", None

        if st.valid:
            o = st.rack_to_cam(np.zeros(3))
            y = st.rack_to_cam(np.array([0.0, 0.1, 0.0]))
            if o[2] > 0.01 and y[2] > 0.01:
                po = cam.K @ (o / o[2])
                py = cam.K @ (y / y[2])
                st.camera_roll_deg = round(image_up_angle_deg(py[:2] - po[:2]), 1)
        st.latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self.state = st
        return st

    @staticmethod
    def _reproj(obj, img, rv, tv, cam: CameraModel) -> float:
        proj, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
        return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

    def _solve(self, obs, cam: CameraModel, prev: RackState):
        size = self.tag_size_m
        if len(obs) == 1:
            # NOTE: OpenCV 5.0.0's SOLVEPNP_IPPE / IPPE_SQUARE return identity or
            # NaN poses for valid input (verified in tests/test_zerog_rack.py), so
            # the single-tag case uses SQPnP plus a refinement seeded from the
            # previous pose, which also resolves the planar flip ambiguity.
            tag_id, img = obs[0]
            c = self.layout.tags[tag_id]
            h = size / 2.0
            obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
            cands = []
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    cands.append((rv, tv))
                if prev.R is not None:
                    rv0, _ = cv2.Rodrigues(prev.R)
                    tv0 = (prev.t + prev.R @ c).reshape(3, 1)
                    ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, rv0.copy(), tv0.copy(),
                                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                    if ok:
                        cands.append((rv, tv))
            except cv2.error:
                return None
            scored = []
            for rv, tv in cands:
                if not np.all(np.isfinite(rv)) or not np.all(np.isfinite(tv)) or float(tv.ravel()[2]) <= 0:
                    continue
                R, _ = cv2.Rodrigues(rv)
                scored.append((R, tv.ravel(), self._reproj(obj, img, rv, tv, cam)))
            if not scored:
                return None
            scored.sort(key=lambda s: s[2])
            best = scored[0]
            # A lone small tag has a two-fold flip ambiguity; when both solutions
            # fit about equally well, prefer the one closest to the last pose.
            if len(scored) > 1 and prev.R is not None and scored[1][2] < 1.5 * max(scored[0][2], 0.5):
                best = min(scored[:2], key=lambda s: _rot_angle_deg(s[0], prev.R))
            R, t_tag, err = best
            # Tag axes are the rack axes, displaced by the tag centre:
            #   P_cam = R (P_rack - c) + t_tag
            return R, t_tag - R @ c, err

        obj = np.vstack([self.layout.corners(i, size) for i, _ in obs])
        img = np.vstack([c for _, c in obs])
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return None
            rv, tv = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rv, tv)
        except cv2.error:
            return None
        proj, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(rv)
        return R, tv.ravel(), err

    def to_dict(self, st: RackState | None = None) -> dict[str, Any]:
        st = st or self.state
        return {
            "status": st.status,
            "tags_visible": st.tags_visible,
            "tag_corners_px": st.tag_corners_px,
            "reprojection_px": None if st.reproj_px is None else round(st.reproj_px, 2),
            "camera_roll_deg": st.camera_roll_deg,
            "camera_pose": st.camera_pose_in_rack(),
            "latency_ms": st.latency_ms,
            "tag_size_m": round(self.tag_size_m, 4),
            "rack_id": self.layout.rack_id,
            "rack_name": self.layout.name,
            "known_tags": sorted(self.layout.tags),
        }

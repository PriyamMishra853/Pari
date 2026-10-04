"""3D human pose from a single RGB camera on CPU (MediaPipe BlazePose GHUM).

BlazePose returns 33 landmarks: image positions plus metric "world" positions
(hip-centred, metres) regressed from the GHUM body model. PnP between the two
places the body in the camera frame, so every joint gets a real camera-frame
3D position that the rack transform can then carry into rack coordinates.

Orientation search: BlazePose expects a roughly upright person. When the camera
is rolled (laptop on its side or upside down) the frame is rotated by quarter
turns before detection - first the turn suggested by the rack (if locked), then
the previously successful one, then the rest - and the landmarks are mapped back
to raw camera pixels exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np

from parikshak.zerog.config import ROOT, CameraModel
from parikshak.zerog.orient import ViewTransform, image_up_angle_deg, quarter_turns_to_upright

MODEL_PATH = ROOT / "models" / "mediapipe" / "pose_landmarker_lite.task"

LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]
IDX = {n: i for i, n in enumerate(LANDMARKS)}

BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32),
    (28, 32), (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
    (0, 2), (2, 7), (0, 5), (5, 8), (9, 10),
]

# BlazePose index for each of the 17 COCO keypoints the procedure trackers use.
COCO_FROM_BLAZE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

SEARCH_INTERVAL_S = 0.4  # full 4-way search at most this often while nobody is found


@dataclass
class PoseObservation:
    k: int                 # quarter-turns (clockwise) applied before detection
    px: np.ndarray         # 33x2 raw-camera pixels
    vis: np.ndarray        # 33 visibility in [0, 1]
    world: np.ndarray      # 33x3 BlazePose world landmarks (m, hip-centred)
    cam: np.ndarray | None = None  # 33x3 camera frame (m)
    pnp_reproj_px: float | None = None
    detect_ms: float = 0.0
    tries: int = 1

    def visible(self, i: int, thr: float = 0.5) -> bool:
        return bool(self.vis[i] >= thr)


class OneEuro:
    """One-Euro low-pass filter for arrays (Casiez et al., 2012)."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.6, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.x.shape != x.shape or self.t is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(1e-3, t - self.t)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class PoseEstimator:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.available = self.model_path.exists()
        self.error: str | None = None if self.available else f"model file missing: {self.model_path}"
        self._lm: dict[int, object] = {}
        self._ts: dict[int, int] = {}
        self.k = 0
        self._last_search = -1e9
        self._f_cam = OneEuro(min_cutoff=1.0, beta=0.8)
        self._f_px = OneEuro(min_cutoff=1.5, beta=0.05)
        try:
            import mediapipe  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on install
            self.available = False
            self.error = f"mediapipe not importable: {exc}"

    # ------------------------------------------------------------------ model
    def _landmarker(self):
        # IMAGE mode on purpose: VIDEO mode's ROI tracking keeps fitting a stale
        # (e.g. sideways) body for several frames after the camera is turned,
        # which defeats instant re-orientation. Detection every frame is ~40 ms.
        if 0 not in self._lm:
            from mediapipe.tasks.python import BaseOptions, vision

            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._lm[0] = vision.PoseLandmarker.create_from_options(opts)
        return self._lm[0]

    def _detect(self, raw: np.ndarray, k: int, t: float) -> PoseObservation | None:
        import mediapipe as mp

        h, w = raw.shape[:2]
        vt = ViewTransform(k, w, h, out_w=w if k % 2 == 0 else h, out_h=h if k % 2 == 0 else w)
        img = vt.apply(raw)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        res = self._landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        ms = (time.perf_counter() - t0) * 1000.0
        if not res.pose_landmarks or not res.pose_world_landmarks:
            return None
        lms = res.pose_landmarks[0]
        rw, rh = img.shape[1], img.shape[0]
        px_rot = np.array([[l.x * rw, l.y * rh] for l in lms], dtype=np.float64)
        vis = np.array([
            min(float(l.visibility if l.visibility is not None else 1.0),
                float(l.presence if l.presence is not None else 1.0)) for l in lms
        ])
        world = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks[0]], dtype=np.float64)
        return PoseObservation(k=k, px=vt.view_to_raw(px_rot), vis=vis, world=world, detect_ms=ms)

    # --------------------------------------------------------------- public
    def estimate(self, raw: np.ndarray, cam: CameraModel, t: float, prefer_k: int | None = None) -> PoseObservation | None:
        if not self.available:
            return None
        first = self.k if prefer_k is None else prefer_k
        tries = 1
        obs = self._detect(raw, first, t)
        if obs is not None:
            # Detected but lying sideways in the detection image: re-run upright,
            # which BlazePose localises far better.
            ang = self._body_angle(obs, in_detection_image=True)
            if ang is not None and abs(ang) > 60.0:
                k2 = (first + quarter_turns_to_upright(ang)) % 4
                if k2 != first:
                    tries += 1
                    obs2 = self._detect(raw, k2, t)
                    if obs2 is not None:
                        obs = obs2
        elif t - self._last_search >= SEARCH_INTERVAL_S:
            self._last_search = t
            for k in [(first + d) % 4 for d in (1, 3, 2)]:
                tries += 1
                obs = self._detect(raw, k, t)
                if obs is not None:
                    break
        if obs is None:
            return None
        obs.tries = tries
        self.k = obs.k
        self._lift(obs, cam)
        # temporal smoothing in raw-pixel and camera space (both continuous
        # across orientation changes, since they are not rotated frames)
        if obs.cam is not None:
            if self._f_cam.x is not None and np.linalg.norm(self._f_cam.x[23:25].mean(0) - obs.cam[23:25].mean(0)) > 0.6:
                self._f_cam.reset()
            obs.cam = self._f_cam(obs.cam, t)
        obs.px = self._f_px(obs.px, t)
        return obs

    def reset(self) -> None:
        self._f_cam.reset()
        self._f_px.reset()

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _body_angle(obs: PoseObservation, in_detection_image: bool = False) -> float | None:
        """Clockwise angle of the body's head-up direction from image-up."""
        px = obs.px
        sh = px[[11, 12]].mean(0)
        if obs.vis[23] > 0.5 and obs.vis[24] > 0.5:
            v = sh - px[[23, 24]].mean(0)
        elif obs.vis[0] > 0.5 and (obs.vis[11] > 0.5 or obs.vis[12] > 0.5):
            v = px[0] - sh
        else:
            return None
        ang = image_up_angle_deg(v)
        if in_detection_image:
            ang = ((ang + 90.0 * obs.k + 180.0) % 360.0) - 180.0
        return ang

    @staticmethod
    def _lift(obs: PoseObservation, cam: CameraModel) -> None:
        """Place the metric world skeleton in the camera frame with PnP."""
        sel = obs.vis >= 0.6
        if int(sel.sum()) < 6:
            sel = obs.vis >= 0.4
        if int(sel.sum()) < 6:
            return
        obj = obs.world[sel].astype(np.float64)
        img = obs.px[sel].astype(np.float64)
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return
            rv, tv = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rv, tv)
        except cv2.error:
            return
        if not (np.all(np.isfinite(rv)) and np.all(np.isfinite(tv))):
            return
        z = float(tv.ravel()[2])
        if not (0.2 <= z <= 8.0):
            return
        R, _ = cv2.Rodrigues(rv)
        obs.cam = obs.world @ R.T + tv.reshape(1, 3)
        proj, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
        obs.pnp_reproj_px = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

    @staticmethod
    def to_coco17(obs: PoseObservation, pts: np.ndarray) -> np.ndarray:
        """17x3 (x, y, conf) COCO keypoints from BlazePose points in some image."""
        out = np.zeros((17, 3), dtype=np.float64)
        for j, b in enumerate(COCO_FROM_BLAZE):
            out[j, 0], out[j, 1], out[j, 2] = pts[b, 0], pts[b, 1], obs.vis[b]
        return out

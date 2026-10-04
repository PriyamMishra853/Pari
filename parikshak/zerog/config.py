"""Loads camera intrinsics and the rack fiducial layout from configs/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    p = CONFIGS / name
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    status: str
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def calibrated(self) -> bool:
        return self.status.upper().startswith("CALIBRATED_CHECKERBOARD")


def camera_model(width: int, height: int) -> CameraModel:
    """Intrinsics scaled to the frame size actually being processed.

    Distortion is only applied when the camera was calibrated against a
    checkerboard; an estimated distortion vector is worse than none.
    """
    cam = load_yaml("camera.yaml").get("camera", {})
    res = cam.get("resolution", [640, 480])
    intr = cam.get("intrinsics", {})
    sx = width / float(res[0] or 640)
    sy = height / float(res[1] or 480)
    K = np.array(
        [
            [float(intr.get("fx", 615.0)) * sx, 0.0, float(intr.get("cx", res[0] / 2)) * sx],
            [0.0, float(intr.get("fy", 615.0)) * sy, float(intr.get("cy", res[1] / 2)) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    status = str(cam.get("calibration_status", "UNCALIBRATED_ESTIMATE"))
    model = CameraModel(K=K, dist=np.zeros(5), status=status, width=width, height=height)
    if model.calibrated:
        model.dist = np.array(cam.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64)
    return model


@dataclass
class RackLayout:
    rack_id: str
    name: str
    tag_size_m: float
    tags: dict[int, np.ndarray]  # tag id -> centre in rack frame (metres)
    labels: dict[int, str] = field(default_factory=dict)
    zones: dict[str, Any] = field(default_factory=dict)

    def corners(self, tag_id: int, size_m: float | None = None) -> np.ndarray:
        """The four corners of a tag in rack coordinates, in ArUco order
        (top-left, top-right, bottom-right, bottom-left as seen from +Z)."""
        h = (size_m or self.tag_size_m) / 2.0
        c = self.tags[tag_id]
        return np.array(
            [c + (-h, h, 0.0), c + (h, h, 0.0), c + (h, -h, 0.0), c + (-h, -h, 0.0)],
            dtype=np.float64,
        )


def rack_layout() -> RackLayout:
    rack = load_yaml("rack.yaml").get("rack", {})
    fid = rack.get("fiducials", {})
    tags: dict[int, np.ndarray] = {}
    labels: dict[int, str] = {}
    for t in fid.get("tags", []):
        tags[int(t["id"])] = np.array(t["pos"], dtype=np.float64)
        labels[int(t["id"])] = str(t.get("label", ""))
    if not tags:
        tags = {10: np.zeros(3)}
    return RackLayout(
        rack_id=str(rack.get("id", "RACK")),
        name=str(rack.get("name", "Payload rack")),
        tag_size_m=float(fid.get("tag_size_m", 0.08)),
        tags=tags,
        labels=labels,
        zones=rack.get("payload_zones", {}),
    )

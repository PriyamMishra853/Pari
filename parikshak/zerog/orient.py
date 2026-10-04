"""Quarter-turn view transforms between the raw camera image and the upright
"view" image the procedure trackers reason in.

When the laptop is turned on its side, the raw image is rotated. The trackers
(lift = up, mouth above bottle, ...) assume an upright image, so the pipeline
rotates the frame by k quarter turns - chosen from the rack's measured +Y axis
when tags are visible, or from the body when they are not - and letterboxes it
back to 640x480. Every point can be mapped exactly between the two images.
"""

from __future__ import annotations

import math

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np

_ROTATE = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE} if cv2 is not None else {}


class ViewTransform:
    """k quarter-turns clockwise, then uniform scale + letterbox to out size."""

    def __init__(self, k: int, raw_w: int, raw_h: int, out_w: int = 640, out_h: int = 480) -> None:
        self.k = int(k) % 4
        self.raw_w, self.raw_h = int(raw_w), int(raw_h)
        self.out_w, self.out_h = int(out_w), int(out_h)
        rw, rh = (raw_w, raw_h) if self.k % 2 == 0 else (raw_h, raw_w)
        self.rot_w, self.rot_h = rw, rh
        self.s = min(out_w / rw, out_h / rh)
        self.new_w = int(round(rw * self.s))
        self.new_h = int(round(rh * self.s))
        self.ox = (out_w - self.new_w) // 2
        self.oy = (out_h - self.new_h) // 2

    @property
    def identity(self) -> bool:
        return self.k == 0 and self.new_w == self.out_w and self.new_h == self.out_h

    def apply(self, img: np.ndarray) -> np.ndarray:
        rot = img if self.k == 0 else cv2.rotate(img, _ROTATE[self.k])
        if rot.shape[1] == self.out_w and rot.shape[0] == self.out_h:
            return rot
        resized = cv2.resize(rot, (self.new_w, self.new_h), interpolation=cv2.INTER_AREA)
        if self.new_w == self.out_w and self.new_h == self.out_h:
            return resized
        canvas = np.zeros((self.out_h, self.out_w, 3), dtype=img.dtype)
        canvas[self.oy:self.oy + self.new_h, self.ox:self.ox + self.new_w] = resized
        return canvas

    def raw_to_view(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x, y = p[:, 0], p[:, 1]
        W, H = self.raw_w, self.raw_h
        if self.k == 0:
            xr, yr = x, y
        elif self.k == 1:
            xr, yr = (H - 1) - y, x
        elif self.k == 2:
            xr, yr = (W - 1) - x, (H - 1) - y
        else:
            xr, yr = y, (W - 1) - x
        return np.stack([xr * self.s + self.ox, yr * self.s + self.oy], axis=1)

    def view_to_raw(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        xr = (p[:, 0] - self.ox) / self.s
        yr = (p[:, 1] - self.oy) / self.s
        W, H = self.raw_w, self.raw_h
        if self.k == 0:
            x, y = xr, yr
        elif self.k == 1:
            x, y = yr, (H - 1) - xr
        elif self.k == 2:
            x, y = (W - 1) - xr, (H - 1) - yr
        else:
            x, y = (W - 1) - yr, xr
        return np.stack([x, y], axis=1)

    def view_box_to_raw(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        corners = self.view_to_raw(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
        return (float(corners[:, 0].min()), float(corners[:, 1].min()),
                float(corners[:, 0].max()), float(corners[:, 1].max()))


def image_up_angle_deg(vec_xy: np.ndarray) -> float:
    """Angle of an image-plane direction, clockwise from image-up (0 = up,
    90 = right, 180 = down, -90 = left)."""
    dx, dy = float(vec_xy[0]), float(vec_xy[1])
    return math.degrees(math.atan2(dx, -dy))


def quarter_turns_to_upright(angle_deg: float) -> int:
    """Clockwise quarter-turns that bring a direction at angle_deg to image-up."""
    return int(round(-angle_deg / 90.0)) % 4

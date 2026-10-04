"""Saturated colour-box detection (red_box / yellow_box) that rejects skin."""

from __future__ import annotations

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np

_K = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) if cv2 is not None else None


def _masks(frame: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    not_skin = cv2.bitwise_not(cv2.inRange(ycc, np.array([0, 133, 77]), np.array([255, 173, 127])))
    red = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 110, 60]), np.array([10, 255, 255])),
                         cv2.inRange(hsv, np.array([165, 110, 60]), np.array([180, 255, 255])))
    yellow = cv2.inRange(hsv, np.array([18, 110, 90]), np.array([38, 255, 255]))
    out = {}
    for name, m in (("red_box", red), ("yellow_box", yellow)):
        m = cv2.bitwise_and(m, not_skin)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K)
        out[name] = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K)
    return out


def detect_color_boxes(frame: np.ndarray, wanted: set[str], min_area: float = 250.0) -> list[dict]:
    dets = []
    for name, m in _masks(frame).items():
        if name not in wanted:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        fill = area / max(1.0, w * h)
        dets.append({"cls": name, "conf": round(float(min(0.99, 0.5 + fill / 2)), 2),
                     "box": [float(x), float(y), float(x + w), float(y + h)]})
    return dets

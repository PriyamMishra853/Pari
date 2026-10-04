"""Builds a camera-roll test clip: a still scene (person + AprilTag 10 on the wall)
viewed by a camera that rolls 0 -> 90 -> 180 -> 0 degrees about its optical axis.

Rolling the camera about the optical axis is exactly an in-image rotation about
the principal point, so this exercises the real pipeline (tag detection, PnP,
orientation search, rack transform) with a known ground truth: the person never
moves relative to the tag, so rack-frame inclination must stay constant while
the measured camera roll sweeps.

    python tools/make_rotation_demo.py [--person tests/data/synthetic_person_desk.jpg]
Writes recordings/ROTATION_TEST_synthetic.mp4 (runs it from Reports & recordings).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paste_tag(img: np.ndarray, tag_id: int, x: int, y: int, size: int) -> None:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    m = cv2.aruco.generateImageMarker(d, tag_id, size)
    q = size // 6
    img[y - q:y + size + q, x - q:x + size + q] = 255
    img[y:y + size, x:x + size] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default=str(ROOT / "tests" / "data" / "synthetic_person_desk.jpg"))
    ap.add_argument("--out", default=str(ROOT / "recordings" / "ROTATION_TEST_synthetic.mp4"))
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    scene = cv2.resize(cv2.imread(args.person), (640, 480))
    paste_tag(scene, 10, 440, 70, 64)  # tag 10 = rack origin, upright on the wall (kept near the centre so it stays in view at 90 deg)
    # keyframes (seconds, roll degrees)
    keys = [(0, 0), (2, 0), (5, 90), (7, 90), (10, 180), (12, 180), (15, 0), (16, 0)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 480))
    total = keys[-1][0] * args.fps
    for f in range(total):
        t = f / args.fps
        for (t0, a0), (t1, a1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                a = a0 + (a1 - a0) * (t - t0) / max(1e-6, t1 - t0)
                break
        # camera rolls by +a  ->  image content rotates by -a about the principal point
        M = cv2.getRotationMatrix2D((320, 240), a, 1.0)
        frame = cv2.warpAffine(scene, M, (640, 480), borderValue=(18, 18, 22))
        w.write(frame)
    w.release()
    print(f"wrote {out} ({total} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

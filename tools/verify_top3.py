"""Step-by-step verification of the top experiments on synthetic webcam frames.

    python tools/verify_top3.py

WBP-1 uses three generated stills of a person (bottle on the table, bottle lifted,
drinking); BCX-1 uses procedurally drawn container/red/yellow boxes composited on
the desk of a real-looking frame. Each pose is held for a few seconds at 4 fps
through the real service (BlazePose, YOLO ONNX, trackers), and the script prints
how the procedure advanced. It checks the logic end to end; real-world accuracy
still has to be confirmed with real props, lighting and camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parikshak.perception.tracker_service import TrackerService  # noqa: E402

DATA = ROOT / "tests" / "data"


def run(svc: TrackerService, exp: str, seq: list[tuple[str, np.ndarray, float]], fps: float = 4.0) -> dict:
    svc.set_experiment(exp)
    t = 1000.0
    timeline = []
    for label, img, secs in seq:
        for _ in range(int(secs * fps)):
            tel = svc.process_frame(img, t)
            t += 1.0 / fps
        z = tel["zerog"]
        timeline.append((label, tel["active_step_id"], [s["status"][0].upper() for s in tel["steps"]],
                         z["har"]["activity"], [o["label"] + ("[" + ",".join(o["state"]) + "]" if o["state"] else "")
                                                 for o in z["overlay"]["objects"]]))
    for label, step, st, act, objs in timeline:
        print(f"  after {label:<28} active={step:<5} steps={''.join(st)} activity={act:<16} objects={objs}")
    final = [s["status"] for s in tel["steps"]]
    alerts = [(a["step_id"], a["kind"]) for a in tel.get("alerts", [])]
    print(f"  RESULT {exp}: {final}  complete={tel['is_complete']}  alerts={alerts}")
    return tel


def bcx_scene(base: np.ndarray, red: tuple | None, yellow: tuple | None, container: bool) -> np.ndarray:
    img = base.copy()
    if container:  # open cardboard box on the desk, front face visible
        cv2.rectangle(img, (150, 300), (500, 470), (60, 105, 150), -1)
        cv2.rectangle(img, (150, 300), (500, 470), (20, 40, 60), 4)
        cv2.rectangle(img, (170, 318), (480, 452), (45, 80, 120), -1)
    for c, col in ((red, (30, 30, 210)), (yellow, (30, 210, 235))):
        if c is not None:
            x, y = c
            cv2.rectangle(img, (x, y), (x + 60, y + 60), col, -1)
            cv2.rectangle(img, (x, y), (x + 60, y + 60), (10, 10, 10), 2)
    return img


def main() -> int:
    on_table = cv2.imread(str(DATA / "wbp_on_table.jpg"))
    lifted = cv2.imread(str(DATA / "wbp_lifted.jpg"))
    drinking = cv2.imread(str(DATA / "wbp_drinking.jpg"))
    svc = TrackerService("WBP-1")

    print("WBP-1 nominal run")
    run(svc, "WBP-1", [("bottle on table", on_table, 3), ("bottle lifted", lifted, 3),
                       ("drinking", drinking, 4), ("bottle back, hands off", on_table, 4)])
    print("WBP-1 deviation: drinking skipped")
    run(svc, "WBP-1", [("bottle on table", on_table, 3), ("bottle lifted", lifted, 3),
                       ("bottle back, hands off", on_table, 7)])

    base = cv2.imread(str(DATA / "synthetic_person_desk.jpg"))
    print("BCX-1 nominal run")
    run(svc, "BCX-1", [
        ("container in view", bcx_scene(base, None, None, True), 3),
        ("red + yellow shown", bcx_scene(base, (40, 120), (540, 120), True), 3),
        ("red inside", bcx_scene(base, (200, 360), (540, 120), True), 3),
        ("yellow inside, apart", bcx_scene(base, (200, 360), (380, 360), True), 3),
        ("boxes colliding", bcx_scene(base, (260, 360), (318, 360), True), 3),
        ("boxes separated", bcx_scene(base, (200, 360), (380, 360), True), 3),
    ])
    print("BCX-1 deviation: yellow first")
    run(svc, "BCX-1", [
        ("container in view", bcx_scene(base, None, None, True), 3),
        ("red + yellow shown", bcx_scene(base, (40, 120), (540, 120), True), 3),
        ("yellow inside first", bcx_scene(base, (40, 120), (380, 360), True), 3),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

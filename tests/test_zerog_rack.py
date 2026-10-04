"""Rack-as-gravity: the solved rack pose must follow a real camera roll, and a
point fixed to the rack must keep its rack coordinates whatever the roll."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from parikshak.zerog.config import CameraModel, rack_layout
from parikshak.zerog.orient import ViewTransform, quarter_turns_to_upright
from parikshak.zerog.rack import RackLocalizer


def _rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def _render(layout, cam: CameraModel, R: np.ndarray, t: np.ndarray, tag_ids, size_m: float) -> np.ndarray:
    frame = np.full((cam.height, cam.width, 3), 200, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tid in tag_ids:
        marker = cv2.aruco.generateImageMarker(dictionary, tid, 200)
        # tag36h11 has a 1-module white border outside the black frame; pad so
        # the projected quad covers exactly the black outer square.
        tag = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        corners_rack = layout.corners(tid, size_m)
        pc = corners_rack @ R.T + t
        px = (cam.K @ (pc / pc[:, 2:3]).T).T[:, :2].astype(np.float32)
        src = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], np.float32)
        H = cv2.getPerspectiveTransform(src, px)
        warped = cv2.warpPerspective(tag, H, (cam.width, cam.height), borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((200, 200), 255, np.uint8), H, (cam.width, cam.height))
        # white quiet zone around the tag
        quiet = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        frame[quiet > 0] = 255
        frame[mask > 0] = warped[mask > 0]
    return frame


def _camera_facing_rack(dist: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    # Camera looks along -Z_rack (into the rack face), image-up = rack +Y.
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)  # rack -> cam
    centre = np.array([0.0, 0.0, dist])  # camera position in rack frame
    t = -R @ centre
    return R, t


@pytest.mark.parametrize("roll", [0, 90, 180, 270, 30])
def test_rack_pose_follows_camera_roll(roll):
    layout = rack_layout()
    cam = CameraModel(K=np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1]]), dist=np.zeros(5),
                      status="TEST", width=640, height=480)
    R0, t0 = _camera_facing_rack()
    Rr = _rz(roll)  # rolling the camera about its optical axis
    R, t = Rr @ R0, Rr @ t0
    tid = sorted(layout.tags)[0]
    # put the camera in front of the first tag
    c = layout.tags[tid]
    t = t - R @ c
    frame = _render(layout, cam, R, t, [tid], 0.12)

    loc = RackLocalizer(layout)
    loc.set_tag_size(0.12)
    st = loc.update(frame, cam, t=0.0)
    assert st.status == "LOCKED", st
    assert st.reproj_px is not None and st.reproj_px < 2.0

    # rotation error < 3 deg, translation error < 2 cm
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(st.R.T @ R) - 1) / 2))))
    assert ang < 3.0, ang
    assert np.linalg.norm(st.t - t) < 0.02

    # camera roll reported = direction of rack +Y in the image
    expected = ((roll + 180) % 360) - 180
    got = st.camera_roll_deg
    assert min(abs(got - expected), 360 - abs(got - expected)) < 4.0, (got, expected)

    # a point fixed in the rack keeps its rack coordinates
    p_rack = np.array([0.25, -0.10, 0.40]) + c
    p_cam = R @ p_rack + t
    assert np.allclose(st.cam_to_rack(p_cam), p_rack, atol=0.02)


def test_view_transform_round_trip():
    pts = np.array([[0, 0], [639, 0], [639, 479], [100, 300]], dtype=np.float64)
    for k in range(4):
        vt = ViewTransform(k, 640, 480)
        back = vt.view_to_raw(vt.raw_to_view(pts))
        assert np.allclose(back, pts, atol=1e-6), k


def test_quarter_turns():
    assert quarter_turns_to_upright(0) == 0
    assert quarter_turns_to_upright(90) == 3   # up points right -> turn 90 ccw
    assert quarter_turns_to_upright(-90) == 1
    assert quarter_turns_to_upright(180) == 2
    assert quarter_turns_to_upright(-170) == 2

"""Service wrapper for YOLO & OpenCV experiment tracking.

Manages video input sources (live client webcam, uploaded video files, and
simulated runs) and coordinates frame processing and telemetry serialization.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    import importlib
    cv2 = importlib.import_module("cv2")
except ImportError:
    cv2 = None  # type: ignore
import numpy as np

from parikshak.perception.yolo_tracker import YoloExperimentTracker

UPLOAD_DIR = Path("runs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TrackerService:
    """Singleton-style coordinator for live and replay experiment tracking."""

    def __init__(self, experiment_id: str = "WBP-1") -> None:
        self.lock = threading.RLock()
        self.experiment_id = experiment_id
        self.tracker = YoloExperimentTracker(self.experiment_id)
        self.video_cap: Any | None = None
        self.active_video_path: str | None = None
        self.is_video_playing = False
        self.last_telemetry: dict[str, Any] = {}

    def set_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self.lock:
            self.experiment_id = experiment_id.upper()
            self.tracker = YoloExperimentTracker(self.experiment_id)
            # Process empty frame to prime telemetry
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _, self.last_telemetry = self.tracker.process_frame(dummy)
            return self.last_telemetry

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.tracker.reset()
            if self.video_cap is not None:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _, self.last_telemetry = self.tracker.process_frame(dummy)
            return self.last_telemetry

    def process_b64_frame(self, b64_data: str) -> dict[str, Any]:
        """Accepts base64 encoded JPEG/PNG frame from browser webcam, processes it

        with YOLO, and returns the annotated frame and telemetry.
        """
        # Strip header if present: 'data:image/jpeg;base64,...'
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"error": "Invalid image data"}
        except Exception as exc:
            return {"error": f"Failed to decode image: {exc}"}

        with self.lock:
            annotated, telemetry = self.tracker.process_frame(frame)
            self.last_telemetry = telemetry

            # Re-encode annotated frame to JPEG base64
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

            return {
                "annotated_frame": out_b64,
                "telemetry": telemetry,
            }

    def load_video_file(self, video_path: str | Path) -> dict[str, Any]:
        with self.lock:
            path_str = str(video_path)
            if not os.path.exists(path_str):
                return {"error": f"Video file not found: {path_str}"}

            if self.video_cap is not None:
                self.video_cap.release()

            self.video_cap = cv2.VideoCapture(path_str)
            self.active_video_path = path_str
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.tracker.reset()

            return {
                "status": "loaded",
                "video_path": path_str,
                "total_frames": total_frames,
                "fps": fps,
                "duration_s": round(total_frames / fps, 1) if fps > 0 else 0,
            }

    def get_next_video_frame(self) -> dict[str, Any]:
        with self.lock:
            if self.video_cap is None or not self.video_cap.isOpened():
                return {"error": "No video opened"}

            ret, frame = self.video_cap.read()
            if not ret:
                # Loop back or signal EOF
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
                if not ret:
                    return {"is_eof": True}

            current_pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            annotated, telemetry = self.tracker.process_frame(frame)
            self.last_telemetry = telemetry

            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

            return {
                "annotated_frame": out_b64,
                "telemetry": telemetry,
                "frame_pos": current_pos,
                "total_frames": total_frames,
                "is_eof": False,
            }

    def list_available_cameras(self) -> list[dict[str, Any]]:
        cameras = []
        for idx in [1, 0]:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    mean_val = float(np.mean(frame)) if ret and frame is not None else 0.0
                    is_physical = (idx == 1 or mean_val > 15.0)
                    name = f"HP HD Camera (Physical Webcam - Index {idx})" if is_physical else f"Virtual / Aux Camera (Index {idx})"
                    cameras.append({
                        "index": idx,
                        "name": name,
                        "is_physical": is_physical,
                        "mean_brightness": round(mean_val, 1),
                    })
                    cap.release()
            except Exception:
                pass
        return cameras

    def start_local_camera(self, camera_idx: int = -1) -> dict[str, Any]:
        with self.lock:
            if self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
                self.video_cap = None

            # Prioritize candidate indices:
            # If camera_idx specified and >= 0, check that requested index first.
            # If camera_idx < 0 (auto), prioritize index 1 (HP HD Camera) before index 0 (EShare Virtual Camera)
            if camera_idx is not None and camera_idx >= 0:
                candidates = [camera_idx, 1, 0]
            else:
                candidates = [1, 0]

            selected_cap = None
            selected_idx = 1

            for idx in candidates:
                try:
                    temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if temp_cap.isOpened():
                        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        # Warm up 2 frames
                        for _ in range(2):
                            temp_cap.read()
                        ret, test_frame = temp_cap.read()
                        mean_val = float(np.mean(test_frame)) if ret and test_frame is not None else 0.0

                        # If this camera has a real non-black image (brightness > 8.0), it is our physical camera!
                        if mean_val > 8.0:
                            selected_cap = temp_cap
                            selected_idx = idx
                            break

                        # Fallback if no bright camera found yet
                        if selected_cap is None:
                            selected_cap = temp_cap
                            selected_idx = idx
                        else:
                            temp_cap.release()
                except Exception:
                    continue

            if selected_cap is None or not selected_cap.isOpened():
                return {"error": "Failed to open hardware camera. Please check webcam connection or Windows camera privacy settings."}

            self.video_cap = selected_cap
            self.is_video_playing = True
            return {"status": "camera_started", "camera_index": selected_idx}

    def stop_local_camera(self) -> dict[str, Any]:
        with self.lock:
            if self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
                self.video_cap = None
            self.is_video_playing = False
            return {"status": "camera_stopped"}

    def get_camera_frame_mjpeg(self) -> bytes | None:
        with self.lock:
            if self.video_cap is None or not self.video_cap.isOpened():
                res = self.start_local_camera(-1)
                if "error" in res or self.video_cap is None:
                    return None
            ret = False
            frame = None
            for _ in range(3):
                ret, temp = self.video_cap.read()
                if ret and temp is not None and temp.size > 0:
                    frame = temp
                    break
                time.sleep(0.01)
            if not ret or frame is None:
                return None
            h, w = frame.shape[:2]
            if w != 640 or h != 480:
                frame = cv2.resize(frame, (640, 480))
            annotated, telemetry = self.tracker.process_frame(frame)
            self.last_telemetry = telemetry
            ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                return None
            return jpeg.tobytes()

    def get_camera_frame_b64(self) -> dict[str, Any]:
        with self.lock:
            if self.video_cap is None or not self.video_cap.isOpened():
                res = self.start_local_camera(0)
                if "error" in res or self.video_cap is None:
                    return {"error": res.get("error", "Hardware camera not accessible")}
            ret = False
            frame = None
            for _ in range(4):
                ret, temp = self.video_cap.read()
                if ret and temp is not None and temp.size > 0:
                    frame = temp
                    break
                time.sleep(0.015)

            if not ret or frame is None:
                return {"error": "Failed to read frame from hardware camera. Please ensure webcam is not in use by another app."}

            h, w = frame.shape[:2]
            if w != 640 or h != 480:
                frame = cv2.resize(frame, (640, 480))

            annotated, telemetry = self.tracker.process_frame(frame)
            self.last_telemetry = telemetry
            ret, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                return {"error": "Failed to encode frame"}
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
            return {
                "annotated_frame": out_b64,
                "telemetry": telemetry,
                "status": "camera_live",
            }

    def generate_demo_video(self, output_path: str = "runs/uploads/demo_bottle_run.mp4") -> str:
        """Generates a synthetic realistic demonstration video for instant test


        and verification if the user doesn't have an MP4 file handy.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        w, h = 640, 480
        fps = 20
        total_frames = fps * 15  # 15 seconds

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

        # Bottle starting position
        table_y = int(h * 0.75)
        bottle_w, bottle_h = 50, 140

        for f in range(total_frames):
            frame = np.full((h, w, 3), (25, 30, 40), dtype=np.uint8)

            # Draw lab desk surface
            cv2.rectangle(frame, (0, table_y), (w, h), (40, 50, 65), -1)
            cv2.line(frame, (0, table_y), (w, table_y), (80, 100, 130), 2)

            # Astronaut / Person head & body outline
            person_x = int(w * 0.5)
            person_y = int(h * 0.3)
            # Head
            cv2.circle(frame, (person_x, person_y), 50, (190, 170, 160), -1)
            # Eyes & mouth
            cv2.circle(frame, (person_x - 15, person_y - 5), 4, (40, 40, 40), -1)
            cv2.circle(frame, (person_x + 15, person_y - 5), 4, (40, 40, 40), -1)
            cv2.line(frame, (person_x - 12, person_y + 22), (person_x + 12, person_y + 22), (40, 40, 40), 2)
            # Body suit
            cv2.ellipse(frame, (person_x, person_y + 160), (100, 120), 0, 0, 360, (210, 220, 230), -1)

            # Motion sequence over 15 seconds:
            # 0-3s: S01 bottle on table, hand approaches
            # 3-6s: S02 hand grasps and lifts bottle upward
            # 6-10s: S03 bottle brought to mouth, drinking action held
            # 10-14s: S04 bottle returned to table and released
            t = f / fps

            if t < 3.0:
                # Idle on table
                bx = int(w * 0.6)
                by = table_y - bottle_h
                hand_x = int(w * 0.8 - (t / 3.0) * 80)
                hand_y = table_y - 30
            elif t < 6.0:
                # Lifting
                p = (t - 3.0) / 3.0
                bx = int(w * 0.6 - p * 40)
                by = int((table_y - bottle_h) - p * 120)
                hand_x = bx + 20
                hand_y = by + 60
            elif t < 10.0:
                # Drinking at mouth
                bx = person_x + 20
                by = person_y + 10
                hand_x = bx + 20
                hand_y = by + 50
            else:
                # Returning to table
                p = (t - 10.0) / 4.0
                p = min(1.0, p)
                bx = int((person_x + 20) + p * 80)
                by = int((person_y + 10) + p * (table_y - bottle_h - (person_y + 10)))
                hand_x = int(bx + 20 + p * 80)
                hand_y = int(by + 50)

            # Draw Hand / Wrist
            cv2.circle(frame, (hand_x, hand_y), 18, (180, 160, 150), -1)

            # Draw Water Bottle (Blue body + cap)
            cv2.rectangle(frame, (bx - bottle_w // 2, by), (bx + bottle_w // 2, by + bottle_h), (220, 140, 50), -1)
            cv2.rectangle(frame, (bx - bottle_w // 2, by), (bx + bottle_w // 2, by + bottle_h), (255, 200, 120), 2)
            # Water level inside
            cv2.rectangle(frame, (bx - bottle_w // 2 + 3, by + 40), (bx + bottle_w // 2 - 3, by + bottle_h - 3), (240, 180, 40), -1)
            # Cap
            cv2.rectangle(frame, (bx - 12, by - 16), (bx + 12, by), (200, 200, 200), -1)

            writer.write(frame)

        writer.release()
        return str(out_p)

    def simulate_event(self, event_name: str) -> dict[str, Any]:
        """Manually inject nominal steps or deviations for instant testing."""
        with self.lock:
            now = time.time()
            if event_name == "nominal_step":
                if self.tracker.step_idx < len(self.tracker.steps):
                    cur = self.tracker.steps[self.tracker.step_idx]
                    cur.status = "completed"
                    cur.completed_at = now
                    if self.tracker.experiment_id in ["BCX-1", "BOX-COL-1"]:
                        if cur.id == "S01":
                            self.tracker.bcx1_s01_hold_duration = 0.8
                            self.tracker.bcx1_container_locked = True
                        elif cur.id == "S02":
                            self.tracker.bcx1_s02_hold_duration = 0.8
                            self.tracker.bcx1_colors_verified = True
                        elif cur.id == "S03":
                            self.tracker.bcx1_s03_hold_duration = 0.8
                            self.tracker.bcx1_red_placed = True
                            self.tracker.bcx1_red_inside = True
                        elif cur.id == "S04":
                            self.tracker.bcx1_s04_hold_duration = 0.8
                            self.tracker.bcx1_yellow_placed = True
                            self.tracker.bcx1_yellow_inside = True
                            self.tracker.bcx1_distance_px = 60.0
                        elif cur.id == "S05":
                            self.tracker.bcx1_collision_hold_duration = 0.8
                            self.tracker.bcx1_collision = True
                            self.tracker.bcx1_distance_px = 0.0
                        elif cur.id == "S06" or self.tracker.step_idx >= len(self.tracker.steps) - 1:
                            self.tracker.bcx1_s06_hold_duration = 0.7
                            self.tracker.bcx1_collision = False
                            self.tracker.bcx1_distance_px = 40.0
                            self.tracker.protocol_complete = True
                    else:
                        if cur.id == "S01":
                            self.tracker.s01_stable_duration = self.tracker.S01_TARGET_S
                        elif cur.id == "S02":
                            self.tracker.target_lifted = True
                            self.tracker.s02_lift_duration = self.tracker.S02_TARGET_S
                        elif cur.id == "S03":
                            self.tracker.water_consumed = True
                            self.tracker.drink_hold_duration = self.tracker.S03_TARGET_S
                        elif cur.id == "S04" or self.tracker.step_idx >= len(self.tracker.steps) - 1:
                            self.tracker.protocol_complete = True
                            self.tracker.s04_settle_duration = self.tracker.S04_TARGET_S
                    self.tracker._advance_step(now)
            elif event_name == "inject_skip":
                if self.tracker.experiment_id in ["MOA-1", "MULTI-OBJ-1"]:
                    for s in self.tracker.steps:
                        if s.id == "S06":
                            s.status = "skipped"
                            break
                    self.tracker.moa1_water_consumed = False
                    self.tracker._trigger_alert(
                        step_id="S06",
                        severity="critical",
                        kind="skipped",
                        message="Step S06 skipped! Water bottle returned to table without drinking.",
                        tts="Warning: Step skipped. Drink water from the bottle before returning it.",
                        t=now,
                    )
                elif self.tracker.experiment_id in ["BCX-1", "BOX-COL-1"]:
                    for s in self.tracker.steps:
                        if s.id == "S03":
                            s.status = "skipped"
                            break
                    self.tracker.bcx1_red_inside = False
                    self.tracker.bcx1_yellow_inside = True
                    self.tracker._trigger_alert(
                        step_id="S03",
                        severity="critical",
                        kind="skipped",
                        message="Step S03 skipped! Red box was skipped and yellow box placed instead.",
                        tts="Warning: Step skipped. Place the red box before proceeding.",
                        t=now,
                    )
                else:
                    # Mark step S03 as skipped
                    for s in self.tracker.steps:
                        if s.id == "S03":
                            s.status = "skipped"
                            break
                    self.tracker._trigger_alert(
                        step_id="S03",
                        severity="caution",
                        kind="skipped",
                        message="Step S03 Skipped! Bottle returned to table without drinking.",
                        tts="Warning. Step three skipped. Drink water before returning the bottle.",
                        t=now,
                    )
            elif event_name in ["inject_out_of_order", "moa1_out_of_order"]:
                if self.tracker.experiment_id in ["MOA-1", "MULTI-OBJ-1"]:
                    self.tracker.moa1_phone_picked = True
                    self.tracker.moa1_is_seated = False
                    self.tracker.moa1_chair_pulled = False
                    cur_sid = self.tracker.steps[self.tracker.step_idx].id if self.tracker.step_idx < len(self.tracker.steps) else "S01"
                    self.tracker._trigger_alert(
                        step_id=cur_sid,
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order step detected! Pull chair and sit before picking up smartphone.",
                        tts="Warning: Step out of order. Pull chair and sit down before picking up smartphone.",
                        t=now,
                    )
                elif self.tracker.experiment_id in ["BCX-1", "BOX-COL-1"]:
                    for s in self.tracker.steps:
                        if s.id == "S03":
                            s.status = "skipped"
                            break
                    self.tracker.bcx1_red_inside = False
                    self.tracker.bcx1_yellow_inside = True
                    self.tracker._trigger_alert(
                        step_id="S03",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S03 requires placing Red Box first. Yellow box detected inside.",
                        tts="Warning: Step out of order. Place the Red Box inside the container first.",
                        t=now,
                    )
                else:
                    self.tracker._trigger_alert(
                        step_id="S01",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S01 requires resting bottle on table first to calibrate baseline.",
                        tts="Warning: Step out of order. Place bottle on table to calibrate baseline first.",
                        t=now,
                    )
            elif event_name == "inject_wrong_object":
                self.tracker._trigger_alert(
                    step_id="S02",
                    severity="caution",
                    kind="wrong_object",
                    message="Wrong object grasped! Confusable cup/mug detected instead of target bottle.",
                    tts="Wrong object grasped. That is the cup, not the water bottle.",
                    t=now,
                )
            elif event_name == "bcx1_collision":
                # Simulate BCX-1 collision event (Step S05)
                self.tracker.bcx1_container_locked = True
                self.tracker.bcx1_colors_verified = True
                self.tracker.bcx1_red_placed = True
                self.tracker.bcx1_yellow_placed = True
                self.tracker.bcx1_red_inside = True
                self.tracker.bcx1_yellow_inside = True
                self.tracker.bcx1_collision = True
                self.tracker.bcx1_distance_px = 0.0
                self.tracker.bcx1_collision_hold_duration = 0.8
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04", "S05"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S05",
                    severity="info",
                    kind="collision_confirmed",
                    message="Box Collision Confirmed! Red and Yellow boxes in active collision inside container.",
                    tts="Notice: Box collision confirmed. Contact verified inside container.",
                    t=now,
                )
            elif event_name == "bcx1_skip_red":
                # Simulate BCX-1 skipped red box (placing yellow first in S03)
                for s in self.tracker.steps:
                    if s.id == "S03":
                        s.status = "skipped"
                        break
                self.tracker.bcx1_red_inside = False
                self.tracker.bcx1_yellow_inside = True
                self.tracker._trigger_alert(
                    step_id="S03",
                    severity="critical",
                    kind="out_of_order",
                    message="Out of order! Step S03 requires placing Red Box first. Yellow box detected inside.",
                    tts="Warning: Step out of order. Place the Red Box inside the container first.",
                    t=now,
                )
            elif event_name == "bcx1_hazard_oob":
                # Simulate collision out of bounds hazard
                self.tracker.bcx1_collision = True
                self.tracker.bcx1_distance_px = 0.0
                self.tracker.bcx1_red_inside = False
                self.tracker.bcx1_yellow_inside = False
                self.tracker._trigger_alert(
                    step_id="S05",
                    severity="critical",
                    kind="hazard",
                    message="Out-of-Bounds Hazard! Collision must take place inside the container.",
                    tts="Hazard warning: Collision must occur inside the container.",
                    t=now,
                )
            elif event_name == "moa1_pull_chair":
                self.tracker.moa1_chair_pulled = True
                self.tracker.moa1_s01_hold_duration = 0.8
                for s in self.tracker.steps:
                    if s.id == "S01":
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S01",
                    severity="info",
                    kind="step_complete",
                    message="Chair pulled into position. Now sit down on the chair.",
                    tts="Chair positioned. Please sit down on the chair.",
                    t=now,
                )
            elif event_name == "moa1_sit":
                self.tracker.moa1_chair_pulled = True
                self.tracker.moa1_is_seated = True
                self.tracker.moa1_knee_angle_deg = 92.5
                self.tracker.moa1_s02_hold_duration = 0.8
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S02",
                    severity="info",
                    kind="step_complete",
                    message="Seated posture confirmed (knee angle 92.5°). Pick up the smartphone.",
                    tts="Seated posture confirmed. Pick up the smartphone.",
                    t=now,
                )
            elif event_name == "moa1_phone_pickup":
                self.tracker.moa1_is_seated = True
                self.tracker.moa1_phone_picked = True
                self.tracker.moa1_s03_hold_duration = 0.6
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S03",
                    severity="info",
                    kind="step_complete",
                    message="Smartphone picked up. Place it back down onto the desk surface.",
                    tts="Phone picked up. Return the smartphone to the desk.",
                    t=now,
                )
            elif event_name == "moa1_phone_stow":
                self.tracker.moa1_phone_picked = True
                self.tracker.moa1_phone_stowed = True
                self.tracker.moa1_s04_hold_duration = 0.6
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S04",
                    severity="info",
                    kind="step_complete",
                    message="Smartphone stowed on desk. Next, grasp and lift the water bottle.",
                    tts="Smartphone stowed. Grasp and lift the water bottle.",
                    t=now,
                )
            elif event_name == "moa1_bottle_lift":
                self.tracker.moa1_bottle_lifted = True
                self.tracker.moa1_s05_hold_duration = 0.6
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04", "S05"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S05",
                    severity="info",
                    kind="step_complete",
                    message="Water bottle lifted. Bring to mouth and drink water (hold >= 1.5s).",
                    tts="Bottle lifted. Bring to mouth and drink water.",
                    t=now,
                )
            elif event_name == "moa1_drink":
                self.tracker.moa1_bottle_lifted = True
                self.tracker.moa1_drinking_hold_duration = 1.5
                self.tracker.moa1_water_consumed = True
                for s in self.tracker.steps:
                    if s.id in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                        s.status = "completed"
                        s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S06",
                    severity="info",
                    kind="step_complete",
                    message="Drinking verified (held >= 1.5s). Return bottle to table and release hands.",
                    tts="Water consumed. Return bottle to table and release hands.",
                    t=now,
                )
            elif event_name == "moa1_bottle_return":
                self.tracker.moa1_bottle_returned = True
                self.tracker.protocol_complete = True
                self.tracker.moa1_s07_hold_duration = 0.6
                for s in self.tracker.steps:
                    s.status = "completed"
                    s.completed_at = now
                self.tracker._advance_step(now)
                self.tracker._trigger_alert(
                    step_id="S07",
                    severity="info",
                    kind="nominal_completion",
                    message="Multi-Object Experiment Complete! All seven activities verified nominal.",
                    tts="Multi-object experiment complete. All seven steps verified nominal.",
                    t=now,
                )
            elif event_name == "moa1_skip_drink":
                for s in self.tracker.steps:
                    if s.id == "S06":
                        s.status = "skipped"
                        break
                self.tracker.moa1_water_consumed = False
                self.tracker._trigger_alert(
                    step_id="S06",
                    severity="critical",
                    kind="skipped",
                    message="Step S06 skipped! Water bottle returned to table without drinking.",
                    tts="Warning: Step skipped. Drink water from the bottle before returning it.",
                    t=now,
                )
            elif event_name == "reset":
                self.tracker.reset()

            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _, telem = self.tracker.process_frame(dummy)
            self.last_telemetry = telem
            return telem


# Global tracker service instance
_service: TrackerService | None = None


def get_tracker_service() -> TrackerService:
    global _service
    if _service is None:
        _service = TrackerService("WBP-1")
    return _service

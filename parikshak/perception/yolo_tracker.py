"""YOLO and OpenCV-based human activity and procedure compliance tracker.

Supports both:
  1. The interactive benchmark: WBP-1 Water Bottle Protocol
  2. Space experiment tracking (vials, tools, latches, gloves, crew actions)

Tracks object bounding boxes, hand positions, mouth/face keypoints, spatial
contact, and sequence order to detect nominal execution and compliance deviations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import importlib
    cv2 = importlib.import_module("cv2")
except ImportError:
    cv2 = None  # type: ignore
import numpy as np

# 17 COCO Keypoints for Astronaut Skeletal Tracking
COCO_KEYPOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Skeletal limb segments (bones) connecting joints with cybernetic colors (BGR)
SKELETON_BONES = [
    # Head / Visor links (Electric Cyan)
    (0, 1, (255, 230, 0)),
    (0, 2, (255, 230, 0)),
    (1, 3, (255, 230, 0)),
    (2, 4, (255, 230, 0)),
    # Shoulders / Neck (Neon Cyan)
    (5, 6, (0, 240, 255)),
    # Left Arm (Tech Emerald)
    (5, 7, (60, 240, 120)),
    (7, 9, (60, 240, 120)),
    # Right Arm (Tech Emerald)
    (6, 8, (60, 240, 120)),
    (8, 10, (60, 240, 120)),
    # Torso / Spine Core (Amber Gold)
    (5, 11, (0, 180, 255)),
    (6, 12, (0, 180, 255)),
    (11, 12, (0, 180, 255)),
    # Left Leg / Lower Body (Neon Magenta)
    (11, 13, (230, 80, 200)),
    (13, 15, (230, 80, 200)),
    # Right Leg / Lower Body (Neon Magenta)
    (12, 14, (230, 80, 200)),
    (14, 16, (230, 80, 200)),
]


@dataclass
class StepState:
    id: str
    name: str
    prompt: str
    status: str = "pending"  # pending, active, completed, skipped, failed
    started_at: float | None = None
    completed_at: float | None = None
    hold_start: float | None = None
    elapsed_s: float = 0.0


@dataclass
class DeviationAlert:
    step_id: str
    severity: str  # advisory, caution, critical
    kind: str      # out_of_order, skipped, wrong_object, premature, timeout
    message: str
    timestamp: float
    spoken_tts: str


class YoloExperimentTracker:
    """End-to-end vision tracker and procedure validator using YOLO and OpenCV."""

    # Calibration and verification hold targets (in seconds)
    S01_TARGET_S = 0.7  # Stable on table
    S02_TARGET_S = 0.5  # Grasp and lift
    S03_TARGET_S = 0.7  # Drinking hold (responsive natural sip duration)
    S04_TARGET_S = 0.5  # Return to table & release

    def __init__(self, experiment_id: str = "WBP-1") -> None:
        self.load_models()
        self.init_procedure(experiment_id)

    def load_models(self) -> None:
        from pathlib import Path

        from ultralytics import YOLO

        root = Path(__file__).resolve().parents[2]
        # ONNX Runtime is ~1.5x faster than PyTorch on this class of CPU.
        det_onnx = root / "yolov8n.onnx"
        self.det_model = YOLO(str(det_onnx), task="detect") if det_onnx.exists() else YOLO(str(root / "yolov8n.pt"))
        self._pose_onnx = root / "yolov8n-pose.onnx"
        self._pose_pt = root / "yolov8n-pose.pt"
        self._pose_model = None
        # Per-frame injections from the zero-g pipeline (parikshak/zerog/pipeline.py):
        #   external_pose_mode: BlazePose supplies the skeleton; YOLO-pose only runs as a fallback
        #   external_pose:      17x3 COCO keypoints (x, y, conf) in this frame, or None
        #   injected_det:       shared YOLO detection result for this frame
        self.external_pose_mode = False
        self.external_pose = None
        self.injected_det = None
        self.render_hud = True
        self.last_pose_source = "none"
        self.overlay_objects: list[dict[str, Any]] = []
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.det_model(dummy, imgsz=320, verbose=False)

    @property
    def pose_model(self):
        if self._pose_model is None:
            from ultralytics import YOLO

            if self._pose_onnx.exists():
                self._pose_model = YOLO(str(self._pose_onnx), task="pose")
            else:
                self._pose_model = YOLO(str(self._pose_pt))
        return self._pose_model

    def _detect(self, frame: np.ndarray):
        if self.injected_det is not None:
            return self.injected_det
        return self.det_model(frame, imgsz=320, verbose=False)[0]

    def init_procedure(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id.upper()
        if self.experiment_id == "WBP-1":
            self.title = "Water Bottle Protocol (Activity Benchmark)"
            self.rack_id = "BENCH-1"
            self.steps = [
                StepState("S01", "Locate water bottle on table",
                          "Ensure the water bottle is resting on the table surface."),
                StepState("S02", "Grasp and lift bottle",
                          "Grasp the water bottle and lift it off the table."),
                StepState("S03", "Drink water from bottle",
                          "Bring the bottle to your mouth and drink water."),
                StepState("S04", "Return bottle to table surface",
                          "Place the water bottle back onto the table surface and release it."),
            ]
            self.expected_target = "bottle"
            self.confusables = ["cup", "wine glass", "mug"]
        elif self.experiment_id == "CRX-2":
            self.title = "CRX-2 : Colloid Resuspension and Cold Return"
            self.rack_id = "MSG-A (Space Station Rack)"
            self.steps = [
                StepState("S01", "Secure in foot restraint",
                          "Secure yourself in the foot restraint."),
                StepState("S02", "Retrieve vial B from cold locker",
                          "Retrieve vial B, the orange banded vial, from cold locker L one."),
                StepState("S03", "Confirm processing unit idle",
                          "Check processing unit is idle and latch is closed."),
                StepState("S04", "Resuspend vial B by agitation",
                          "Agitate vial B ten times to resuspend the colloid."),
                StepState("S05", "Place vial B in tray T1",
                          "Place vial B in tray T one and let it settle."),
                StepState("S06", "Confirm uniform suspension",
                          "Inspect vial B and confirm suspension is uniform."),
                StepState("S07", "Return vial B to cold locker",
                          "Return vial B to cold locker L one."),
                StepState("S08", "Verify glovebox latch closed",
                          "Verify glovebox latch is closed and locked."),
            ]
            self.expected_target = "bottle"  # or vial
            self.confusables = ["cup"]
        elif self.experiment_id in ["BCX-1", "BOX-COL-1"]:
            self.title = "BCX-1: Two-Box Collision in Container (Red & Yellow)"
            self.rack_id = "BENCH-1 (Desktop / Glovebox)"
            self.steps = [
                StepState("S01", "Identify Bigger Container Box",
                          "Position and verify the outer container box in workspace view."),
                StepState("S02", "Detect & Verify Box Colors",
                          "Present Red Box and Yellow Box to camera to verify color detection."),
                StepState("S03", "Place Red Box into Container",
                          "Place the Red Box inside the container box."),
                StepState("S04", "Place Yellow Box into Container",
                          "Place the Yellow Box inside the container box, separated from the red box."),
                StepState("S05", "Collide Red and Yellow Boxes",
                          "Bring the Red and Yellow boxes together until collision occurs."),
                StepState("S06", "Separate Boxes & Confirm",
                          "Separate the boxes inside the container to conclude trial."),
            ]
            self.expected_target = "red_box"
            self.confusables = []
        elif self.experiment_id in ["MOA-1", "MULTI-OBJ", "MOA"]:
            self.title = "MOA-1 : Multi-Object Experiment (Chair, Phone & Bottle)"
            self.rack_id = "BENCH-1 (Desktop / Ergonomics Lab)"
            self.steps = [
                StepState("S01", "Pull Chair into Position",
                          "Grasp the chair and pull it into position at your desk."),
                StepState("S02", "Sit Down on Chair",
                          "Sit down on the chair in an ergonomic posture."),
                StepState("S03", "Pick Up Smartphone",
                          "Pick up the smartphone from the desk."),
                StepState("S04", "Return Smartphone to Desk",
                          "Place the smartphone back onto the desk surface."),
                StepState("S05", "Grasp and Lift Water Bottle",
                          "Grasp the water bottle and lift it off the desk."),
                StepState("S06", "Drink Water from Bottle",
                          "Bring the water bottle to your mouth and drink water."),
                StepState("S07", "Return Bottle to Table & Release Hands",
                          "Return the bottle to the table surface and release your hands."),
            ]
            self.expected_target = "multi_object"
            self.confusables = []
        else:
            # Generic / custom space procedure
            self.title = f"Procedure {self.experiment_id}"
            self.rack_id = "PAYLOAD-1"
            self.steps = [
                StepState("S01", "Identify payload sample", "Identify and inspect the payload sample."),
                StepState("S02", "Retrieve sample from rack", "Grasp and retrieve sample from rack."),
                StepState("S03", "Execute procedure protocol", "Perform protocol operation on sample."),
                StepState("S04", "Restow and secure", "Return sample and secure in restraint."),
            ]
            self.expected_target = "bottle"
            self.confusables = ["cup"]

        self.reset()

    def reset(self) -> None:
        self.step_idx = 0
        for s in self.steps:
            s.status = "pending"
            s.started_at = None
            s.completed_at = None
            s.hold_start = None
            s.elapsed_s = 0.0

        if self.steps:
            self.steps[0].status = "active"
            self.steps[0].started_at = time.time()

        self.alerts: list[DeviationAlert] = []
        self.recent_alerts: list[DeviationAlert] = []
        self.frame_count = 0
        self.start_wall_time = time.time()
        self.last_nominal_seen = time.time()
        self.last_frame_time: float | None = None
        self.baseline_table_y: float | None = None
        self.initial_bottle_y: float | None = None

        # Step 1: Stability on table calibration (target: 1.0s)
        self.s01_stable_start: float | None = None
        self.s01_stable_duration: float = 0.0

        # Step 2: Grasp and lift verification (target: 1.0s)
        self.s02_lift_start: float | None = None
        self.s02_lift_duration: float = 0.0
        self.target_lifted = False

        # Step 3: Sustained drinking verification (target: 0.8s)
        self.drink_hold_start: float | None = None
        self.drink_hold_duration: float = 0.0
        self.last_drinking_seen_time: float = 0.0
        self.water_consumed = False
        self.abandon_table_start: float | None = None

        # Step 4: Return to table and hands released (target: 0.6s)
        self.s04_settle_start: float | None = None
        self.s04_settle_duration: float = 0.0
        self.last_s04_seen_time: float = 0.0
        self.protocol_complete = False

        # Cached mouth position for when bottle occludes face during drinking
        self.last_known_mouth: tuple[float, float] | None = None
        self.last_known_mouth_time: float = 0.0

        # Target bottle tracking memory (smoothing during hand occlusion)
        self.last_target_box: tuple[float, float, float, float] | None = None
        self.last_target_time: float = 0.0
        self.last_target_conf: float = 0.0

        # Full-Body 17-Point Skeletal Tracking & Biometric Posture State
        self.last_skeleton_data: dict[str, Any] | None = None
        self.last_body_centroid: tuple[float, float] | None = None
        self.last_centroid_time: float = 0.0
        self.body_motion_velocity_px: float = 0.0
        self.posture_status: str = "NO_ASTRONAUT_DETECTED"
        self.posture_stability: str = "IDLE"
        self.torso_angle_deg: float = 0.0
        self.body_points_count: int = 0

        # BCX-1 Two-Box Collision state variables
        self.bcx1_outer_box: tuple[float, float, float, float] | None = None
        self.bcx1_locked_container: tuple[float, float, float, float] | None = None
        self.bcx1_container_locked: bool = False
        self.bcx1_red_box: tuple[float, float, float, float] | None = None
        self.bcx1_yellow_box: tuple[float, float, float, float] | None = None
        self.bcx1_last_red_box: tuple[float, float, float, float] | None = None
        self.bcx1_last_yellow_box: tuple[float, float, float, float] | None = None
        self.bcx1_last_seen_red_time: float = 0.0
        self.bcx1_last_seen_yellow_time: float = 0.0
        self.bcx1_red_inside: bool = False
        self.bcx1_yellow_inside: bool = False
        self.bcx1_collision: bool = False
        self.bcx1_distance_px: float = 999.0
        self.bcx1_colors_verified: bool = False
        self.bcx1_red_placed: bool = False
        self.bcx1_yellow_placed: bool = False

        # BCX-1 Hold Timers for 6 steps
        self.bcx1_s01_hold_start: float | None = None
        self.bcx1_s01_hold_duration: float = 0.0
        self.bcx1_s02_hold_start: float | None = None
        self.bcx1_s02_hold_duration: float = 0.0
        self.bcx1_s03_hold_start: float | None = None
        self.bcx1_s03_hold_duration: float = 0.0
        self.bcx1_s04_hold_start: float | None = None
        self.bcx1_s04_hold_duration: float = 0.0
        self.bcx1_collision_hold_start: float | None = None
        self.bcx1_collision_hold_duration: float = 0.0
        self.bcx1_s06_hold_start: float | None = None
        self.bcx1_s06_hold_duration: float = 0.0
        self.bcx1_last_collision_time: float = 0.0
        self.bcx1_last_separation_time: float = 0.0
        self.bcx1_abandon_start: float | None = None

        # MOA-1 Multi-Object Activity tracking state
        self.moa1_chair_box: tuple[float, float, float, float] | None = None
        self.moa1_last_chair_box: tuple[float, float, float, float] | None = None
        self.moa1_initial_chair_y: float | None = None
        self.moa1_chair_pulled: bool = False
        self.moa1_s01_hold_start: float | None = None
        self.moa1_s01_hold_duration: float = 0.0

        self.moa1_is_seated: bool = False
        self.moa1_knee_angle_deg: float | None = None
        self.moa1_s02_hold_start: float | None = None
        self.moa1_s02_hold_duration: float = 0.0

        self.moa1_phone_box: tuple[float, float, float, float] | None = None
        self.moa1_last_phone_box: tuple[float, float, float, float] | None = None
        self.moa1_initial_phone_y: float | None = None
        self.moa1_phone_picked: bool = False
        self.moa1_s03_hold_start: float | None = None
        self.moa1_s03_hold_duration: float = 0.0

        self.moa1_phone_stowed: bool = False
        self.moa1_s04_hold_start: float | None = None
        self.moa1_s04_hold_duration: float = 0.0

        self.moa1_bottle_box: tuple[float, float, float, float] | None = None
        self.moa1_last_bottle_box: tuple[float, float, float, float] | None = None
        self.moa1_initial_bottle_y: float | None = None
        self.moa1_bottle_lifted: bool = False
        self.moa1_s05_hold_start: float | None = None
        self.moa1_s05_hold_duration: float = 0.0

        self.moa1_drinking_hold_start: float | None = None
        self.moa1_drinking_hold_duration: float = 0.0
        self.moa1_last_drinking_seen_time: float = 0.0
        self.moa1_water_consumed: bool = False

        self.moa1_bottle_returned: bool = False
        self.moa1_s07_hold_start: float | None = None
        self.moa1_s07_hold_duration: float = 0.0

    def process_frame(self, frame: np.ndarray, current_time: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Processes one video frame: runs YOLO detection + pose, checks procedure step logic,
        draws Mission Control HUD, and returns telemetry.
        """
        if current_time is None:
            current_time = time.time()

        self.frame_count += 1
        h, w = frame.shape[:2]
        if w != 640 or h != 480:
            frame = cv2.resize(frame, (640, 480))
            h, w = 480, 640

        # Check if dummy blank frame (used during reset / initial handshake)
        is_dummy_frame = (np.mean(frame) < 1.0)

        # Route to dedicated BCX-1 Two-Box Collision pipeline
        if self.experiment_id in ["BCX-1", "BOX-COL-1"]:
            return self._process_bcx1_frame(frame, current_time, is_dummy_frame)

        # Route to dedicated MOA-1 Multi-Object Activity pipeline
        if self.experiment_id in ["MOA-1", "MULTI-OBJ", "MOA"]:
            return self._process_moa1_frame(frame, current_time, is_dummy_frame)

        # 1. Run YOLO Object Detection with imgsz=320 for real-time high-FPS CPU inference
        det_results = self._detect(frame)
        detected_objects = []
        target_box = None
        target_conf = 0.0
        confusable_box = None

        for box in det_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.det_model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            if conf < 0.15:
                continue

            detected_objects.append({
                "class": cls_name,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })

            # Check if this matches our target (e.g. bottle or drinking container)
            if cls_name == "bottle" or (self.expected_target == "bottle" and cls_name in ["bottle", "cup", "wine glass", "vase", "bowl"]):
                rank_score = conf + (1.0 if cls_name == "bottle" else 0.0)
                if rank_score > target_conf:
                    target_box = (x1, y1, x2, y2)
                    target_conf = rank_score

            # Check if secondary confusable object (e.g. cup/mug distinct from target)
            elif cls_name in self.confusables:
                confusable_box = (x1, y1, x2, y2)

        # Temporal smoothing for target bottle: retain for up to 0.7s during hand occlusion
        if target_box is not None:
            self.last_target_box = target_box
            self.last_target_time = current_time
            self.last_target_conf = target_conf
        elif self.last_target_box is not None and (current_time - self.last_target_time) < 0.7:
            target_box = self.last_target_box
            target_conf = self.last_target_conf

        # 2. Run Full-Body 17-Point Pose Estimation & Spaceflight Biometrics
        dt = (current_time - self.last_frame_time) if self.last_frame_time else 0.05
        dt = min(max(dt, 0.01), 0.35)
        self.last_frame_time = current_time

        skeleton_data = self._extract_full_body_skeleton(frame, current_time, dt, is_dummy_frame)
        person_detected = skeleton_data["detected"]
        wrists = skeleton_data["wrists"]
        mouth_region = skeleton_data["mouth_region"]

        # 3. Geometric Reasoning & Spatial Relationships
        hand_contact_target = False
        hand_contact_confusable = False
        closest_wrist_dist = 9999.0

        if target_box is not None:
            bx1, by1, bx2, by2 = target_box
            b_width = bx2 - bx1
            b_height = by2 - by1

            for w_info in wrists:
                wx, wy = w_info["point"]
                dist_x = max(bx1 - wx, 0.0, wx - bx2)
                dist_y = max(by1 - wy, 0.0, wy - by2)
                dist = math.hypot(dist_x, dist_y)
                closest_wrist_dist = min(closest_wrist_dist, dist)

                # Wrist inside or in realistic proximity (< 60px or bottle width)
                if dist < max(60.0, b_width * 1.0):
                    hand_contact_target = True

        if confusable_box is not None:
            cx1, cy1, cx2, cy2 = confusable_box
            for w_info in wrists:
                wx, wy = w_info["point"]
                dist_x = max(cx1 - wx, 0.0, wx - cx2)
                dist_y = max(cy1 - wy, 0.0, wy - cy2)
                dist = math.hypot(dist_x, dist_y)
                if dist < 85.0:
                    hand_contact_confusable = True

        # Distance to mouth/face & Drinking Pose Check
        dist_to_mouth = 9999.0
        near_mouth = False
        is_drinking_pose = False
        wrist_dist_to_mouth = 9999.0

        if mouth_region is not None and len(wrists) > 0:
            mx, my = mouth_region
            wrist_dist_to_mouth = min(math.hypot(w["point"][0] - mx, w["point"][1] - my) for w in wrists)

        # Table surface baseline (calibrated during S01 when resting on table)
        table_zone_y = h * 0.65
        ref_table = self.baseline_table_y or table_zone_y
        in_table_zone = False
        if target_box is not None:
            b_bottom = target_box[3]
            # In table zone if resting on or below the table reference line or lower frame
            if b_bottom >= (ref_table - 40.0) or (b_bottom > h * 0.45):
                in_table_zone = True

        # Vertical Lift Calculation
        is_lifted = False
        lift_delta = 0.0
        if target_box is not None:
            b_bottom = target_box[3]
            b_center_y = (target_box[1] + target_box[3]) / 2.0
            if self.initial_bottle_y is not None:
                lift_delta = self.initial_bottle_y - b_center_y

            # True physical lift: bottle bottom lifted off baseline table surface
            # or center is lifted above initial position, or bottle top is raised to drinking height
            if self.baseline_table_y is not None:
                if (self.baseline_table_y - b_bottom) > 22.0 or lift_delta > 22.0 or target_box[1] < (h * 0.45):
                    is_lifted = True
            elif self.initial_bottle_y is not None and lift_delta > 25.0:
                is_lifted = True
            elif target_box[1] < (h * 0.35) or (mouth_region is not None and target_box[1] <= (mouth_region[1] + 35.0)):
                is_lifted = True

        if target_box is not None and mouth_region is not None:
            bx1, by1, bx2, by2 = target_box
            b_height = by2 - by1
            b_width = bx2 - bx1
            top_center = ((bx1 + bx2) / 2.0, by1)  # top drinking opening of bottle
            top_left = (bx1, by1)
            top_right = (bx2, by1)
            mid_left = (bx1, (by1 + by2) / 2.0)
            mid_right = (bx2, (by1 + by2) / 2.0)
            center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            mx, my = mouth_region

            # Multi-angle distance from mouth to bottle key points (upright, horizontal, tilted bottles)
            pts = [top_center, top_left, top_right, mid_left, mid_right, center]
            dist_to_mouth = min(math.hypot(p[0] - mx, p[1] - my) for p in pts)

            # Distance from mouth to bottle bounding box edge
            dist_x = max(bx1 - mx, 0.0, mx - bx2)
            dist_y = max(by1 - my, 0.0, my - by2)
            box_dist_to_mouth = math.hypot(dist_x, dist_y)

            # Check if mouth coordinates overlap bottle region
            mouth_overlap = (bx1 - 50.0 <= mx <= bx2 + 50.0) and (by1 - 40.0 <= my <= by2 + 30.0)

            # CRITICAL GEOMETRY SEPARATION:
            # The bottle is in vertical drinking elevation if the top of the bottle
            # is at or near mouth height (NOT down on the table).
            # When the bottle is resting on the desk, by1 is typically > my + 70px.
            in_drinking_elevation = (by1 <= (my + 65.0))

            # Near mouth requires:
            # 1) Direct mouth overlap with bottle, OR
            # 2) Close proximity while bottle is in drinking elevation (not resting on desk), OR
            # 3) Wrist brought up to mouth with bottle close to mouth
            if mouth_overlap:
                near_mouth = True
            elif in_drinking_elevation and (dist_to_mouth < 140.0 or box_dist_to_mouth < 70.0):
                near_mouth = True
            elif in_drinking_elevation and wrist_dist_to_mouth < 150.0 and dist_to_mouth < 180.0:
                near_mouth = True
        elif mouth_region is not None and wrist_dist_to_mouth < 135.0 and self.target_lifted:
            # If bottle is occluded by hand/face while drinking, wrist at mouth maintains near_mouth
            near_mouth = True

        # Genuine Drinking Pose requires:
        # Case 1: Bottle detected near mouth and lifted
        if near_mouth:
            if target_box is not None:
                bottle_raised = is_lifted or target_box[1] < (h * 0.65) or target_box[3] < (h * 0.80)
                hand_holding = hand_contact_target or closest_wrist_dist < 140.0 or len(wrists) == 0
                not_on_table = not (in_table_zone and not is_lifted and target_box[3] > (h * 0.60))
                if bottle_raised and hand_holding and not_on_table:
                    is_drinking_pose = True
            elif self.target_lifted and wrist_dist_to_mouth < 150.0:
                # Bottle lifted previously, hand currently held up at mouth to drink
                is_drinking_pose = True

        # Case 2: Occlusion-resistant drinking check:
        # If user is in Step S03 (or just lifted bottle in S02), and hand/wrist is raised directly to mouth level
        active_step_preview = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
        if not is_drinking_pose and mouth_region is not None and len(wrists) > 0:
            mx, my = mouth_region
            for w in wrists:
                wx, wy = w["point"]
                # Wrist is right at mouth/chin level (within 135px radius, y around mouth)
                if math.hypot(wx - mx, wy - my) < 135.0 and (my - 60.0 <= wy <= my + 110.0):
                    if active_step_preview and active_step_preview.id in ["S02", "S03"] and (self.target_lifted or self.s02_lift_duration > 0.2):
                        is_drinking_pose = True
                        near_mouth = True
                        break

        # Grasp detection for Step 4 release checking:
        is_grasping_target = False
        if target_box is not None:
            bx1, by1, bx2, by2 = target_box
            b_width = bx2 - bx1
            for w_info in wrists:
                wx, wy = w_info["point"]
                dist_x = max(bx1 - wx, 0.0, wx - bx2)
                dist_y = max(by1 - wy, 0.0, wy - by2)
                dist = math.hypot(dist_x, dist_y)
                # Actively grasping requires wrist directly on bottle body
                if dist < max(20.0, b_width * 0.30) and (by1 - 10.0 <= wy <= by2 + 5.0):
                    is_grasping_target = True

        # Hands Released Condition:
        # User has let go of the bottle if:
        # 1. No wrists detected in frame, OR
        # 2. Not actively grasping the bottle, OR
        # 3. Closest wrist distance is >= 25px, OR
        # 4. Wrists are resting below the bottle (hands down on table / keyboard)
        hands_released = False
        if len(wrists) == 0:
            hands_released = True
        elif not is_grasping_target:
            hands_released = True
        elif closest_wrist_dist >= 25.0:
            hands_released = True
        elif target_box is not None and all(w["point"][1] > (target_box[3] + 5.0) for w in wrists):
            hands_released = True

        # Returned to Table Surface Condition:
        # Bottle has descended from mouth/drinking zone down onto table/desk plane
        is_returned_to_table = False
        if target_box is not None:
            b_top = target_box[1]
            b_bottom = target_box[3]
            b_center_y = (b_top + b_bottom) / 2.0

            # Top of bottle is clearly below mouth (at least 35px below mouth level)
            away_from_mouth = (not near_mouth) and (mouth_region is None or b_top > (mouth_region[1] + 35.0) or b_center_y > (mouth_region[1] + 65.0))

            # In desk surface zone (lower frame, calibrated baseline, or not lifted)
            in_surface_zone = in_table_zone or (b_bottom >= (ref_table - 50.0)) or (b_bottom > h * 0.42) or (b_top > h * 0.35) or (lift_delta < 30.0)

            if away_from_mouth and in_surface_zone:
                is_returned_to_table = True

        # Frame delta time already initialized for procedure hold timers

        # 4. Procedure Compliance State Machine with Strict Preconditions
        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None

        # Check for wrong object deviation
        if hand_contact_confusable and not hand_contact_target and confusable_box is not None:
            self._trigger_alert(
                step_id=active_step.id if active_step else "S02",
                severity="caution",
                kind="wrong_object",
                message="Wrong object grasped! Picked up the cup/mug instead of the target bottle.",
                tts="Wrong object grasped. Please use the water bottle.",
                t=current_time,
            )

        # Execute step state transitions only on real frames (not dummy blank frame)
        if active_step is not None and not is_dummy_frame:
            active_step.elapsed_s = current_time - (active_step.started_at or current_time)

            # ==========================================
            # STEP 1: S01 - Locate bottle on table
            # ==========================================
            if active_step.id == "S01":
                # Precondition: None (first step).
                # Requirements:
                # 1. Target bottle detected in frame.
                # 2. Bottle is resting in table zone (not held up in mid-air).
                # Check for out-of-order action during Step S01
                if (is_drinking_pose or (near_mouth and is_lifted)) and not self.target_lifted:
                    self._trigger_alert(
                        step_id="S01",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S01 requires resting bottle on table first to calibrate baseline.",
                        tts="Warning: Step out of order. Place bottle on table to calibrate baseline first.",
                        t=current_time,
                    )

                if target_box is not None and (in_table_zone or target_box[3] > (h * 0.40)) and not (is_drinking_pose or (near_mouth and is_lifted)):
                    if self.s01_stable_start is None:
                        self.s01_stable_start = current_time
                    self.s01_stable_duration = current_time - self.s01_stable_start

                    # Calibrate table surface and bottle baseline
                    self.baseline_table_y = target_box[3]
                    self.initial_bottle_y = (target_box[1] + target_box[3]) / 2.0

                    if self.s01_stable_duration >= self.S01_TARGET_S:
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                else:
                    self.s01_stable_start = None
                    self.s01_stable_duration = 0.0

            # ==========================================
            # STEP 2: S02 - Grasp and lift bottle
            # ==========================================
            elif active_step.id == "S02":
                # Strict Precondition: S01 MUST be completed!
                if self.steps[0].status != "completed":
                    # Cannot complete S02 without S01
                    pass
                else:
                    # Requirements:
                    # 1. Hand grasp detected on bottle (hand_contact_target or wrist < 90px)
                    # 2. Bottle lifted vertically off table surface (is_lifted == True and lift_delta > 25px)
                    # 3. Must maintain grasp + lift for >= 1.0 seconds (genuine physical hold)
                    # If user directly moves to drinking pose, auto-complete S02
                    if is_drinking_pose:
                        self.target_lifted = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                    else:
                        grasp_detected = hand_contact_target or (closest_wrist_dist < 90.0)
                        if grasp_detected and is_lifted:
                            if self.s02_lift_start is None:
                                self.s02_lift_start = current_time
                            self.s02_lift_duration += dt
                            if self.s02_lift_duration >= self.S02_TARGET_S:
                                self.target_lifted = True
                                active_step.status = "completed"
                                active_step.completed_at = current_time
                                self._advance_step(current_time)
                                active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        else:
                            self.s02_lift_start = None
                            self.s02_lift_duration = max(0.0, self.s02_lift_duration - (dt * 0.4))

            # ==========================================
            # STEP 3: S03 - Drink water from bottle
            # ==========================================
            elif active_step.id == "S03":
                # Precondition: S01 completed
                if self.steps[0].status != "completed":
                    pass
                else:
                    # Sustained drinking hold accumulation
                    if is_drinking_pose:
                        if self.drink_hold_start is None:
                            self.drink_hold_start = current_time
                        self.drink_hold_duration += dt
                        self.last_drinking_seen_time = current_time

                        # Check if genuine drinking duration (target: S03_TARGET_S = 0.8s) is achieved
                        if self.drink_hold_duration >= self.S03_TARGET_S:
                            self.water_consumed = True
                            active_step.status = "completed"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                    else:
                        # Allow generous pose flicker buffer (up to 1.8s) without wiping accumulated drinking progress
                        if (current_time - self.last_drinking_seen_time) > 1.8:
                            self.drink_hold_start = None
                            self.drink_hold_duration = max(0.0, self.drink_hold_duration - (dt * 0.2))

                    # Check for genuine premature lowering / skipping:
                    # ONLY if the bottle was previously lifted high towards the face (target_lifted),
                    # and then placed back down on table with hands released for >= 4.5 seconds continuously!
                    if self.target_lifted and in_table_zone and not is_lifted and not self.water_consumed and hands_released:
                        if self.abandon_table_start is None:
                            self.abandon_table_start = current_time

                        abandon_time = current_time - self.abandon_table_start
                        if abandon_time >= 4.5:
                            # User set bottle down on table and let go without drinking
                            self._trigger_alert(
                                step_id="S03",
                                severity="caution",
                                kind="skipped",
                                message="Step S03 Skipped! Bottle placed on table without drinking water.",
                                tts="Warning. Step three skipped. Water was not consumed.",
                                t=current_time,
                            )
                            active_step.status = "skipped"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                    else:
                        self.abandon_table_start = None

            # ==========================================
            # STEP 4: S04 - Return bottle to table & release
            # ==========================================
            elif active_step.id == "S04":
                # Precondition: Step 1 completed (bottle was part of protocol)
                if self.steps[0].status != "completed":
                    pass
                else:
                    # Requirements:
                    # 1. Bottle returned to table surface (away from mouth, in surface zone)
                    # 2. Hands released and pulled away from bottle
                    # 3. Stable released state accumulated for >= S04_TARGET_S (0.6s)
                    if is_returned_to_table and hands_released:
                        if self.s04_settle_start is None:
                            self.s04_settle_start = current_time
                        self.s04_settle_duration += dt
                        self.last_s04_seen_time = current_time

                        if self.s04_settle_duration >= self.S04_TARGET_S:
                            self.protocol_complete = True
                            active_step.status = "completed"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                            self._trigger_alert(
                                step_id="S04",
                                severity="info",
                                kind="nominal_completion",
                                message="Water Bottle Protocol Complete! Bottle returned to table and hands released.",
                                tts="Protocol verified nominal. Bottle returned to table surface and released.",
                                t=current_time,
                            )
                    else:
                        # Allow brief pose/detection flicker (up to 1.5s) without wiping accumulated progress
                        if self.last_s04_seen_time > 0 and (current_time - self.last_s04_seen_time) > 1.5:
                            self.s04_settle_start = None
                            self.s04_settle_duration = max(0.0, self.s04_settle_duration - (dt * 0.2))

        # 5. Draw High-Tech Mission Control Visual HUD (OpenCV)
        self.overlay_objects = [o for o in (
            {"label": "target bottle", "cls": "bottle", "box": list(target_box), "conf": round(float(target_conf), 2),
             "state": [n for n, f in (("IN HAND", hand_contact_target), ("LIFTED", is_lifted), ("AT MOUTH", near_mouth),
                                      ("DRINKING", is_drinking_pose)) if f],
             "color": "#a3e635" if is_drinking_pose else "#22d3ee"} if target_box is not None else None,
            {"label": "wrong object", "cls": "cup", "box": list(confusable_box), "conf": 0.0, "state": ["CONFUSABLE"],
             "color": "#f97316"} if confusable_box is not None else None,
        ) if o is not None]
        annotated = frame.copy() if self.render_hud else frame
        if self.render_hud: self._render_hud(
            annotated,
            target_box=target_box,
            confusable_box=confusable_box,
            wrists=wrists,
            mouth_region=mouth_region,
            hand_contact_target=hand_contact_target,
            is_lifted=is_lifted,
            lift_delta=lift_delta,
            near_mouth=near_mouth,
            is_drinking_pose=is_drinking_pose,
            dist_to_mouth=dist_to_mouth,
            in_table_zone=in_table_zone,
            active_step=active_step,
            current_time=current_time,
            is_returned_to_table=is_returned_to_table,
            hands_released=hands_released,
            closest_wrist_dist=closest_wrist_dist,
            skeleton_data=skeleton_data,
        )

        # 6. Generate Telemetry Package
        completed_count = sum(1 for s in self.steps if s.status == "completed")
        compliance_pct = int((completed_count / len(self.steps)) * 100) if self.steps else 100
        drinking_pct = min(100, int((self.drink_hold_duration / self.S03_TARGET_S) * 100))

        step_telemetry = []
        for s in self.steps:
            is_step_verifying = False
            step_pct = 0
            if s.status == "completed":
                step_pct = 100
            elif s.status == "active":
                if s.id == "S01":
                    is_step_verifying = (self.s01_stable_duration > 0.0)
                    step_pct = min(100, int((self.s01_stable_duration / self.S01_TARGET_S) * 100))
                elif s.id == "S02":
                    is_step_verifying = (self.s02_lift_duration > 0.0)
                    step_pct = min(100, int((self.s02_lift_duration / self.S02_TARGET_S) * 100))
                elif s.id == "S03":
                    is_step_verifying = (is_drinking_pose or self.drink_hold_duration > 0.0)
                    step_pct = min(100, int((self.drink_hold_duration / self.S03_TARGET_S) * 100))
                elif s.id == "S04":
                    is_step_verifying = ((is_returned_to_table and hands_released) or self.s04_settle_duration > 0.0)
                    step_pct = min(100, int((self.s04_settle_duration / self.S04_TARGET_S) * 100))

            step_telemetry.append({
                "id": s.id,
                "name": s.name,
                "prompt": s.prompt,
                "status": s.status,
                "elapsed_s": round(s.elapsed_s, 1),
                "is_verifying": is_step_verifying,
                "verification_pct": step_pct,
            })

        telemetry = {
            "experiment_id": self.experiment_id,
            "experiment_title": self.title,
            "rack_id": self.rack_id,
            "frame_idx": self.frame_count,
            "active_step_id": active_step.id if active_step else "DONE",
            "active_step_name": active_step.name if active_step else "Protocol Completed",
            "prompt": active_step.prompt if active_step else "Experiment sequence completed nominally.",
            "compliance_score": compliance_pct,
            "is_complete": self.protocol_complete,
            "drinking": {
                "in_progress": is_drinking_pose,
                "hold_duration_s": round(self.drink_hold_duration, 2),
                "target_duration_s": self.S03_TARGET_S,
                "progress_pct": drinking_pct,
                "water_consumed": self.water_consumed,
            },
            "settle": {
                "in_progress": is_returned_to_table and hands_released,
                "is_returned": is_returned_to_table,
                "hands_released": hands_released,
                "hold_duration_s": round(self.s04_settle_duration, 2),
                "target_duration_s": self.S04_TARGET_S,
                "progress_pct": min(100, int((self.s04_settle_duration / self.S04_TARGET_S) * 100)),
                "is_complete": self.protocol_complete,
            },
            "steps": step_telemetry,
            "geometry": {
                "target_detected": target_box is not None,
                "target_confidence": round(target_conf, 2),
                "person_detected": skeleton_data["detected"],
                "full_body_detected": skeleton_data["detected"] and skeleton_data["body_points_count"] >= 10,
                "body_points_count": skeleton_data["body_points_count"],
                "body_points_total": 17,
                "posture_status": skeleton_data["posture_status"],
                "posture_stability": skeleton_data["posture_stability"],
                "torso_angle_deg": skeleton_data["torso_angle_deg"],
                "body_velocity_px_s": skeleton_data["velocity_px_s"],
                "body_zones": skeleton_data["zones"],
                "arm_angles": skeleton_data["arm_angles"],
                "anchored": skeleton_data["anchored"],
                "skeleton": skeleton_data["keypoints"],
                "hand_contact": hand_contact_target,
                "closest_wrist_px": round(closest_wrist_dist, 1) if closest_wrist_dist < 9000 else None,
                "is_lifted": is_lifted,
                "lift_delta_px": round(lift_delta, 1),
                "near_mouth": near_mouth,
                "is_drinking_pose": is_drinking_pose,
                "in_table_zone": in_table_zone,
                "dist_to_mouth_px": round(dist_to_mouth, 1) if dist_to_mouth < 9000 else None,
            },
            "recent_alert": (
                {
                    "step_id": self.recent_alerts[-1].step_id,
                    "severity": self.recent_alerts[-1].severity,
                    "kind": self.recent_alerts[-1].kind,
                    "message": self.recent_alerts[-1].message,
                    "tts": self.recent_alerts[-1].spoken_tts,
                    "timestamp": self.recent_alerts[-1].timestamp,
                }
                if (self.recent_alerts and (current_time - self.recent_alerts[-1].timestamp) < 4.0)
                else None
            ),
            "alert_count": len(self.alerts),
        }

        return annotated, telemetry

    def _advance_step(self, current_time: float) -> None:
        self.step_idx += 1
        if self.step_idx < len(self.steps):
            next_step = self.steps[self.step_idx]
            next_step.status = "active"
            next_step.started_at = current_time

    def _trigger_alert(self, step_id: str, severity: str, kind: str, message: str, tts: str, t: float) -> None:
        # Avoid duplicate spam within 3.5 seconds
        if self.recent_alerts and (t - self.recent_alerts[-1].timestamp) < 3.5:
            if self.recent_alerts[-1].kind == kind and self.recent_alerts[-1].step_id == step_id:
                return

        alert = DeviationAlert(step_id, severity, kind, message, t, tts)
        self.alerts.append(alert)
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > 5:
            self.recent_alerts.pop(0)

    def _extract_full_body_skeleton(
        self, frame: np.ndarray, current_time: float, dt: float, is_dummy_frame: bool = False
    ) -> dict[str, Any]:
        """Runs YOLOv8-pose to extract all 17 COCO keypoints, computes spaceflight posture,
        torso inclination angles, arm flexion, and microgravity kinematic stability.
        """
        h, w = frame.shape[:2]
        empty_skeleton: dict[str, Any] = {
            "detected": False,
            "keypoints": [],
            "visible_indices": [],
            "body_points_count": 0,
            "body_points_total": 17,
            "zones": {"head": 0, "torso": 0, "upper_limbs": 0, "lower_limbs": 0},
            "wrists": [],
            "mouth_region": None,
            "sh_mid": None,
            "hip_mid": None,
            "torso_angle_deg": 0.0,
            "arm_angles": {"left_elbow_deg": None, "right_elbow_deg": None},
            "leg_angles": {"left_knee_deg": None, "right_knee_deg": None},
            "posture_status": "NO_ASTRONAUT_DETECTED",
            "posture_stability": "IDLE",
            "centroid": None,
            "velocity_px_s": 0.0,
            "anchored": False,
        }

        if is_dummy_frame:
            return empty_skeleton

        if self.external_pose_mode and self.external_pose is not None:
            kpts = np.asarray(self.external_pose, dtype=np.float64)
            self.last_pose_source = "blazepose"
        else:
            pose_results = self.pose_model(frame, imgsz=320, verbose=False)[0]
            if len(pose_results.keypoints) == 0 or pose_results.keypoints.data.shape[1] < 17:
                self.last_pose_source = "none"
                return empty_skeleton
            # Take primary detected person
            kpts = pose_results.keypoints.data[0].cpu().numpy()
            self.last_pose_source = "yolo-pose"

        keypoints_list = []
        visible_indices = set()
        wrists = []

        zone_counts = {"head": 0, "torso": 0, "upper_limbs": 0, "lower_limbs": 0}
        sum_x, sum_y, count_pts = 0.0, 0.0, 0

        for idx, name in enumerate(COCO_KEYPOINTS):
            kx, ky, conf = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
            is_vis = bool(conf >= 0.16 and 0 <= kx <= w and 0 <= ky <= h)
            if is_vis:
                visible_indices.add(idx)
                sum_x += kx
                sum_y += ky
                count_pts += 1

                if idx in [0, 1, 2, 3, 4]:
                    zone_counts["head"] += 1
                elif idx in [5, 6, 11, 12]:
                    zone_counts["torso"] += 1
                elif idx in [7, 8, 9, 10]:
                    zone_counts["upper_limbs"] += 1
                elif idx in [13, 14, 15, 16]:
                    zone_counts["lower_limbs"] += 1

                if idx == 9:
                    wrists.append({"side": "left", "point": (kx, ky), "conf": round(conf, 3)})
                elif idx == 10:
                    wrists.append({"side": "right", "point": (kx, ky), "conf": round(conf, 3)})

            keypoints_list.append({
                "id": idx,
                "name": name,
                "x": round(kx, 1),
                "y": round(ky, 1),
                "conf": round(conf, 3),
                "visible": is_vis,
            })

        if count_pts == 0:
            return empty_skeleton

        # 1. Mouth & Head Region Calculation
        mouth_region = None
        nose = kpts[0]
        left_eye, right_eye = kpts[1], kpts[2]
        left_ear, right_ear = kpts[3], kpts[4]
        left_sh, right_sh = kpts[5], kpts[6]

        if nose[2] > 0.20:
            mouth_region = (float(nose[0]), float(nose[1]) + 28.0)
            self.last_known_mouth = mouth_region
            self.last_known_mouth_time = current_time
        elif left_eye[2] > 0.20 and right_eye[2] > 0.20:
            eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
            eye_mid_y = (float(left_eye[1]) + float(right_eye[1])) / 2.0
            mouth_region = (eye_mid_x, eye_mid_y + 45.0)
            self.last_known_mouth = mouth_region
            self.last_known_mouth_time = current_time
        elif left_eye[2] > 0.20:
            mouth_region = (float(left_eye[0]) + 15.0, float(left_eye[1]) + 45.0)
            self.last_known_mouth = mouth_region
            self.last_known_mouth_time = current_time
        elif right_eye[2] > 0.20:
            mouth_region = (float(right_eye[0]) - 15.0, float(right_eye[1]) + 45.0)
            self.last_known_mouth = mouth_region
            self.last_known_mouth_time = current_time
        elif left_ear[2] > 0.20 and right_ear[2] > 0.20:
            ear_mid_x = (float(left_ear[0]) + float(right_ear[0])) / 2.0
            ear_mid_y = (float(left_ear[1]) + float(right_ear[1])) / 2.0
            mouth_region = (ear_mid_x, ear_mid_y + 35.0)
            self.last_known_mouth = mouth_region
            self.last_known_mouth_time = current_time
        elif left_sh[2] > 0.20 and right_sh[2] > 0.20:
            sh_x = (float(left_sh[0]) + float(right_sh[0])) / 2.0
            sh_y = (float(left_sh[1]) + float(right_sh[1])) / 2.0
            mouth_region = (sh_x, sh_y - 95.0)
            self.last_known_mouth = mouth_region
            self.last_known_mouth_time = current_time
        elif self.last_known_mouth is not None and (current_time - self.last_known_mouth_time) < 15.0:
            mouth_region = self.last_known_mouth

        # 2. Torso / Spine Inclination Angle
        torso_angle_deg = 0.0
        sh_mid = None
        if 5 in visible_indices and 6 in visible_indices:
            sh_mid = ((float(kpts[5][0]) + float(kpts[6][0])) / 2.0, (float(kpts[5][1]) + float(kpts[6][1])) / 2.0)
        elif 5 in visible_indices:
            sh_mid = (float(kpts[5][0]), float(kpts[5][1]))
        elif 6 in visible_indices:
            sh_mid = (float(kpts[6][0]), float(kpts[6][1]))

        hip_mid = None
        if 11 in visible_indices and 12 in visible_indices:
            hip_mid = ((float(kpts[11][0]) + float(kpts[12][0])) / 2.0, (float(kpts[11][1]) + float(kpts[12][1])) / 2.0)
        elif 11 in visible_indices:
            hip_mid = (float(kpts[11][0]), float(kpts[11][1]))
        elif 12 in visible_indices:
            hip_mid = (float(kpts[12][0]), float(kpts[12][1]))

        if sh_mid is not None and hip_mid is not None:
            dx = sh_mid[0] - hip_mid[0]
            dy = hip_mid[1] - sh_mid[1]
            torso_angle_deg = round(math.degrees(math.atan2(dx, max(1.0, dy))), 1)

        # 3. Arm & Leg Joint Angles (Flexion / Extension)
        def _calc_angle(pA, pB, pC):
            v1 = (float(pA[0]) - float(pB[0]), float(pA[1]) - float(pB[1]))
            v2 = (float(pC[0]) - float(pB[0]), float(pC[1]) - float(pB[1]))
            mag1 = math.hypot(v1[0], v1[1])
            mag2 = math.hypot(v2[0], v2[1])
            if mag1 < 1e-3 or mag2 < 1e-3:
                return None
            cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)))
            return round(math.degrees(math.acos(cos_a)), 1)

        l_elbow_angle = _calc_angle(kpts[5], kpts[7], kpts[9]) if (5 in visible_indices and 7 in visible_indices and 9 in visible_indices) else None
        r_elbow_angle = _calc_angle(kpts[6], kpts[8], kpts[10]) if (6 in visible_indices and 8 in visible_indices and 10 in visible_indices) else None
        l_knee_angle = _calc_angle(kpts[11], kpts[13], kpts[15]) if (11 in visible_indices and 13 in visible_indices and 15 in visible_indices) else None
        r_knee_angle = _calc_angle(kpts[12], kpts[14], kpts[16]) if (12 in visible_indices and 14 in visible_indices and 16 in visible_indices) else None

        # 4. Kinematic Centroid & Microgravity Velocity Tracking
        centroid = (sum_x / count_pts, sum_y / count_pts)
        inst_velocity = 0.0
        if self.last_body_centroid is not None and dt > 0.005:
            disp = math.hypot(centroid[0] - self.last_body_centroid[0], centroid[1] - self.last_body_centroid[1])
            inst_velocity = disp / dt
            self.body_motion_velocity_px = self.body_motion_velocity_px * 0.75 + inst_velocity * 0.25
        self.last_body_centroid = centroid
        self.last_centroid_time = current_time

        if self.body_motion_velocity_px < 18.0:
            posture_stability = "STABLE"
        elif self.body_motion_velocity_px < 65.0:
            posture_stability = "NOMINAL_MOTION"
        else:
            posture_stability = "RAPID_DISPLACEMENT"

        # 5. Spaceflight Anchorage & Posture Classification
        anchored = False
        posture_status = "UPPER_BODY_ALIGNED"
        if zone_counts["lower_limbs"] >= 1:
            lower_pts = [kpts[idx] for idx in [13, 14, 15, 16] if idx in visible_indices]
            avg_lower_y = sum(float(p[1]) for p in lower_pts) / len(lower_pts)
            if avg_lower_y > (h * 0.65):
                posture_status = "FOOT_RESTRAINT_ANCHORED"
                anchored = True
            else:
                posture_status = "MICROGRAVITY_FLOATING"
        elif zone_counts["torso"] >= 2 or zone_counts["upper_limbs"] >= 2:
            if abs(torso_angle_deg) > 30.0:
                posture_status = "POSTURAL_LEAN_EXCESS"
            else:
                posture_status = "UPPER_BODY_ALIGNED"
        else:
            posture_status = "HEAD_TRACKING"

        skeleton_data = {
            "detected": True,
            "keypoints": keypoints_list,
            "visible_indices": list(visible_indices),
            "body_points_count": count_pts,
            "body_points_total": 17,
            "zones": zone_counts,
            "wrists": wrists,
            "mouth_region": mouth_region,
            "sh_mid": sh_mid,
            "hip_mid": hip_mid,
            "torso_angle_deg": torso_angle_deg,
            "arm_angles": {"left_elbow_deg": l_elbow_angle, "right_elbow_deg": r_elbow_angle},
            "leg_angles": {"left_knee_deg": l_knee_angle, "right_knee_deg": r_knee_angle},
            "posture_status": posture_status,
            "posture_stability": posture_stability,
            "centroid": (round(centroid[0], 1), round(centroid[1], 1)),
            "velocity_px_s": round(self.body_motion_velocity_px, 1),
            "anchored": anchored,
        }

        self.last_skeleton_data = skeleton_data
        self.body_points_count = count_pts
        self.posture_status = posture_status
        self.posture_stability = posture_stability
        self.torso_angle_deg = torso_angle_deg

        return skeleton_data

    def _render_skeletal_rig(self, img: np.ndarray, skeleton_data: dict[str, Any] | None) -> None:
        """Renders high-aesthetic glowing cybernetic skeletal rig and biometric telemetry badge."""
        if not skeleton_data or not skeleton_data.get("detected"):
            return

        h, w = img.shape[:2]
        kpts = {kp["id"]: kp for kp in skeleton_data.get("keypoints", [])}
        vis_set = set(skeleton_data.get("visible_indices", []))

        overlay = img.copy()

        # 1. Draw glowing bones
        for idx1, idx2, bone_color in SKELETON_BONES:
            if idx1 in vis_set and idx2 in vis_set:
                p1 = (max(0, min(w - 1, int(kpts[idx1]["x"]))), max(0, min(h - 1, int(kpts[idx1]["y"]))))
                p2 = (max(0, min(w - 1, int(kpts[idx2]["x"]))), max(0, min(h - 1, int(kpts[idx2]["y"]))))
                # Outer glowing translucent tube
                cv2.line(overlay, p1, p2, bone_color, 5, cv2.LINE_AA)
                # Inner crisp laser core line
                cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend glow into image
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

        # 2. Draw central spine line if mid shoulders and mid hips available
        sh_mid = skeleton_data.get("sh_mid")
        hip_mid = skeleton_data.get("hip_mid")
        if sh_mid and hip_mid:
            sp1 = (max(0, min(w - 1, int(sh_mid[0]))), max(0, min(h - 1, int(sh_mid[1]))))
            sp2 = (max(0, min(w - 1, int(hip_mid[0]))), max(0, min(h - 1, int(hip_mid[1]))))
            cv2.line(img, sp1, sp2, (0, 245, 255), 2, cv2.LINE_AA)
            mid_sp = ((sp1[0] + sp2[0]) // 2, (sp1[1] + sp2[1]) // 2)
            deg_str = f"SPINE: {abs(skeleton_data.get('torso_angle_deg', 0.0))} deg"
            cv2.putText(img, deg_str, (mid_sp[0] + 8, mid_sp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 245, 255), 1, cv2.LINE_AA)

        # 3. Draw joint nodes & reticles
        for idx in vis_set:
            kp = kpts[idx]
            kx = max(0, min(w - 1, int(kp["x"])))
            ky = max(0, min(h - 1, int(kp["y"])))

            # Ring colors per zone
            if idx in [0, 1, 2, 3, 4]:
                ring_color = (255, 230, 0)   # Cyan (Head)
            elif idx in [5, 6]:
                ring_color = (0, 240, 255)   # Gold (Shoulders)
            elif idx in [7, 8, 9, 10]:
                ring_color = (60, 240, 120)  # Green (Elbows/Wrists)
            elif idx in [11, 12]:
                ring_color = (0, 180, 255)   # Amber (Hips)
            else:
                ring_color = (230, 80, 200)  # Magenta (Knees/Ankles)

            cv2.circle(img, (kx, ky), 5, ring_color, 1, cv2.LINE_AA)
            cv2.circle(img, (kx, ky), 2, (255, 255, 255), -1, cv2.LINE_AA)

            # Specific joint callouts:
            if idx == 9:
                cv2.putText(img, "L-HAND", (kx - 48, ky - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 240, 120), 1, cv2.LINE_AA)
            elif idx == 10:
                cv2.putText(img, "R-HAND", (kx + 8, ky - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 240, 120), 1, cv2.LINE_AA)
            elif idx == 7 and skeleton_data.get("arm_angles", {}).get("left_elbow_deg"):
                cv2.putText(img, f"{int(skeleton_data['arm_angles']['left_elbow_deg'])} deg", (kx - 38, ky + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 240, 180), 1, cv2.LINE_AA)
            elif idx == 8 and skeleton_data.get("arm_angles", {}).get("right_elbow_deg"):
                cv2.putText(img, f"{int(skeleton_data['arm_angles']['right_elbow_deg'])} deg", (kx + 8, ky + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 240, 180), 1, cv2.LINE_AA)
            elif idx in [15, 16] and skeleton_data.get("anchored"):
                cv2.putText(img, "ANCHOR", (kx - 18, ky + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 80, 200), 1, cv2.LINE_AA)

        # 4. Biometric Telemetry Matrix Card
        pts_count = skeleton_data.get("body_points_count", 0)
        posture = skeleton_data.get("posture_status", "NOMINAL")
        vel = skeleton_data.get("velocity_px_s", 0.0)
        torso_ang = skeleton_data.get("torso_angle_deg", 0.0)

        card_w, card_h = 245, 72
        cx1, cy1 = w - card_w - 12, 60
        card_overlay = img.copy()
        cv2.rectangle(card_overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (12, 16, 24), -1)
        cv2.addWeighted(card_overlay, 0.82, img, 0.18, 0, img)
        cv2.rectangle(img, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (60, 180, 240), 1)

        cv2.putText(img, f"ASTRONAUT SKELETON: {pts_count}/17 PTS", (cx1 + 8, cy1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1, cv2.LINE_AA)
        status_col = (60, 230, 100) if "ANCHORED" in posture or "ALIGNED" in posture else ((240, 180, 40) if "FLOATING" in posture else (80, 120, 255))
        cv2.putText(img, f"POSTURE: {posture}", (cx1 + 8, cy1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.33, status_col, 1, cv2.LINE_AA)
        cv2.putText(img, f"SPINE TILT: {abs(torso_ang)} deg | DRIFT: {vel} px/s", (cx1 + 8, cy1 + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 215, 230), 1, cv2.LINE_AA)

        zones = skeleton_data.get("zones", {})
        zone_str = f"H:{zones.get('head',0)}/5 A:{zones.get('upper_limbs',0)}/4 T:{zones.get('torso',0)}/4 L:{zones.get('lower_limbs',0)}/4"
        cv2.putText(img, zone_str, (cx1 + 8, cy1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (140, 170, 200), 1, cv2.LINE_AA)

    def _render_hud(
        self,
        img: np.ndarray,
        target_box: tuple[float, float, float, float] | None,
        confusable_box: tuple[float, float, float, float] | None,
        wrists: list[dict],
        mouth_region: tuple[float, float] | None,
        hand_contact_target: bool,
        is_lifted: bool,
        lift_delta: float,
        near_mouth: bool,
        is_drinking_pose: bool,
        dist_to_mouth: float,
        in_table_zone: bool,
        active_step: StepState | None,
        current_time: float,
        is_returned_to_table: bool = False,
        hands_released: bool = False,
        closest_wrist_dist: float = 9999.0,
        skeleton_data: dict[str, Any] | None = None,
    ) -> None:
        """Renders high-aesthetic Mission Control HUD directly on the OpenCV frame."""
        h, w = img.shape[:2]

        # 0. Render Full-Body Skeletal Rig & Biometric Telemetry Card
        if skeleton_data is not None:
            self._render_skeletal_rig(img, skeleton_data)

        # Draw Target Bounding Box
        if target_box is not None:
            x1, y1, x2, y2 = [int(v) for v in target_box]
            # Color: Cyan if drinking, Green if grasped, Amber if resting
            if is_drinking_pose:
                color = (255, 210, 0)  # Bright Cyan
                tag = "TARGET: DRINKING ACTION ACTIVE"
            elif near_mouth:
                color = (220, 200, 50)  # Cyan-Gold
                tag = "TARGET: AT MOUTH ZONE"
            elif hand_contact_target and is_lifted:
                color = (60, 220, 100)  # Bright Emerald
                tag = f"TARGET: GRASPED (LIFT +{int(lift_delta)}px)"
            elif hand_contact_target:
                color = (80, 210, 160)
                tag = "TARGET: GRASPED (TABLE)"
            else:
                color = (0, 200, 255)  # Gold/Yellow
                tag = "TARGET: BOTTLE (TABLE)"

            # Box & Corner Brackets
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            corner_len = min(20, (x2 - x1) // 3)
            # Top-left corner
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Top-right corner
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 4)
            # Bottom-left corner
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 4)
            # Bottom-right corner
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Label badge
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (x1, y1 - 22), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 15, 20), 1, cv2.LINE_AA)

        # Draw Confusable Object Bounding Box
        if confusable_box is not None:
            cx1, cy1, cx2, cy2 = [int(v) for v in confusable_box]
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (50, 50, 220), 2)
            cv2.putText(img, "CONFUSABLE (CUP)", (cx1, cy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 220), 1, cv2.LINE_AA)

        # Draw Wrists / Hands
        for w_info in wrists:
            wx, wy = int(w_info["point"][0]), int(w_info["point"][1])
            cv2.circle(img, (wx, wy), 7, (0, 240, 255), -1)
            cv2.circle(img, (wx, wy), 10, (0, 240, 255), 1)

            # Proximity line to target if grasping
            if target_box is not None and hand_contact_target:
                bx = int((target_box[0] + target_box[2]) / 2.0)
                by = int((target_box[1] + target_box[3]) / 2.0)
                cv2.line(img, (wx, wy), (bx, by), (60, 255, 120), 2, cv2.LINE_AA)

        # Draw Mouth / Head Region indicator & Contact Vector
        if mouth_region is not None:
            mx, my = int(mouth_region[0]), int(mouth_region[1])
            if is_drinking_pose:
                # Green contact circle
                cv2.circle(img, (mx, my), 18, (60, 240, 100), 2)
                cv2.putText(img, "DRINK CONTACT", (mx - 50, my - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 240, 100), 1, cv2.LINE_AA)
                # Line from bottle top to mouth
                if target_box is not None:
                    bx = int((target_box[0] + target_box[2]) / 2.0)
                    by = int(target_box[1])
                    cv2.line(img, (bx, by), (mx, my), (60, 240, 100), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, (mx, my), 14, (255, 140, 50), 2)
                cv2.putText(img, "DRINK ZONE", (mx - 40, my - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 50), 1, cv2.LINE_AA)

        # Table Surface Reference Line
        table_y = int(self.baseline_table_y or (h * 0.65))
        cv2.line(img, (20, table_y), (w - 20, table_y), (100, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(img, "TABLE SURFACE REFERENCE", (30, table_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 140, 160), 1, cv2.LINE_AA)

        # Header Bar (Dark Glassmorphic Banner)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        cv2.line(img, (0, 55), (w, 55), (45, 91, 216), 2)

        # Header Text
        cv2.putText(img, f"PARIKSHAK AI WITNESS | {self.experiment_id}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"RACK: {self.rack_id}", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 180, 210), 1, cv2.LINE_AA)

        # Status Badge (Nominal vs Alert)
        # Status Badge (Nominal vs Verifying vs Alert)
        has_alert = len(self.recent_alerts) > 0 and (current_time - self.recent_alerts[-1].timestamp) < 4.0
        if has_alert:
            badge_color = (40, 40, 220)  # Red
            badge_text = f"ALERT: {self.recent_alerts[-1].kind.upper()}"
        elif self.protocol_complete:
            badge_color = (60, 190, 80)  # Green
            badge_text = "PROTOCOL VERIFIED NOMINAL"
        elif active_step and (
            (active_step.id == "S01" and self.s01_stable_duration > 0)
            or (active_step.id == "S02" and self.s02_lift_duration > 0)
            or (active_step.id == "S03" and self.drink_hold_duration > 0)
            or (active_step.id == "S04" and self.s04_settle_duration > 0)
        ):
            badge_color = (40, 150, 240)  # Amber
            badge_text = f"VERIFYING: {active_step.id}"
        else:
            badge_color = (220, 140, 40)  # Cyan/Blue
            badge_text = f"MONITORING: {active_step.id if active_step else 'NOMINAL'}"

        cv2.rectangle(img, (w - 280, 12), (w - 20, 44), badge_color, -1)
        cv2.putText(img, badge_text, (w - 270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Center Action Bar depending on active step
        if active_step and active_step.id == "S01":
            # Show table stability calibration status
            calib_pct = min(1.0, self.s01_stable_duration / self.S01_TARGET_S)
            bar_w = int(260 * calib_pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), (60, 200, 120), -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)
            calib_text = f"VERIFYING TABLE: {int(calib_pct * 100)}% ({self.s01_stable_duration:.1f}s / {self.S01_TARGET_S:.1f}s)" if self.s01_stable_duration > 0 else "PLACE BOTTLE ON TABLE"
            cv2.putText(img, calib_text, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif active_step and active_step.id == "S02":
            # Grasp and Vertical Lift Progress Bar
            pct = min(1.0, self.s02_lift_duration / self.S02_TARGET_S)
            bar_w = int(260 * pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            grasp_detected = hand_contact_target or (closest_wrist_dist < 90.0)
            fill_color = (60, 230, 100) if (is_lifted and grasp_detected) else (40, 180, 240)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), fill_color, -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)

            if is_lifted and grasp_detected:
                lift_label = f"VERIFYING LIFT: {int(pct * 100)}% ({self.s02_lift_duration:.1f}s / {self.S02_TARGET_S:.1f}s)"
            elif grasp_detected:
                lift_label = "BOTTLE GRASPED -> LIFT OFF TABLE"
            else:
                lift_label = "GRASP AND LIFT BOTTLE"

            cv2.putText(img, lift_label, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif active_step and active_step.id == "S03":
            # Sustained Drinking Action Progress Bar
            pct = min(1.0, self.drink_hold_duration / self.S03_TARGET_S)
            bar_w = int(260 * pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            fill_color = (60, 230, 100) if is_drinking_pose else (40, 180, 240)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), fill_color, -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)

            if is_drinking_pose:
                drink_label = f"VERIFYING DRINK: {int(pct * 100)}% ({self.drink_hold_duration:.1f}s / {self.S03_TARGET_S:.1f}s)"
            elif self.drink_hold_duration > 0:
                drink_label = f"HOLD TO MOUTH ({self.drink_hold_duration:.1f}s / {self.S03_TARGET_S:.1f}s)"
            else:
                drink_label = "BRING BOTTLE TO MOUTH TO DRINK"

            cv2.putText(img, drink_label, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif active_step and active_step.id == "S04":
            # Return to Table Surface & Hand Release Progress Bar
            pct = min(1.0, self.s04_settle_duration / self.S04_TARGET_S)
            bar_w = int(260 * pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            fill_color = (60, 230, 100) if (is_returned_to_table and hands_released) else (40, 180, 240)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), fill_color, -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)

            if not is_returned_to_table:
                settle_label = "PLACE BOTTLE ON TABLE SURFACE"
            elif not hands_released:
                settle_label = "BOTTLE ON TABLE -> RELEASE HANDS"
            else:
                settle_label = f"VERIFYING RELEASE: {int(pct * 100)}% ({self.s04_settle_duration:.1f}s / {self.S04_TARGET_S:.1f}s)"

            cv2.putText(img, settle_label, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif self.protocol_complete:
            cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 35, 25), -1)
            cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (60, 220, 100), -1)
            cv2.putText(img, "ALL 4 STEPS VERIFIED NOMINAL", (w // 2 - 125, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Prompt & Alert Bar
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.line(img, (0, h - 60), (w, h - 60), (45, 91, 216), 1)

        if has_alert:
            alert = self.recent_alerts[-1]
            cv2.putText(img, f"WARNING: {alert.message}", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 255), 2, cv2.LINE_AA)
        elif active_step:
            step_prompt = f"[{active_step.id}] {active_step.prompt}"
            cv2.putText(img, step_prompt, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "All procedure steps verified successfully.", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 120), 1, cv2.LINE_AA)

    def _process_moa1_frame(
        self, frame: np.ndarray, current_time: float, is_dummy_frame: bool
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Processes one video frame for MOA-1: Multi-Object Experiment (Chair, Phone & Bottle)."""
        h, w = frame.shape[:2]
        dt = (current_time - self.last_frame_time) if self.last_frame_time else 0.05
        dt = min(max(dt, 0.01), 0.35)
        self.last_frame_time = current_time

        # 1. Full-Body 17-Point Skeleton & Posture
        skeleton_data = self._extract_full_body_skeleton(frame, current_time, dt, is_dummy_frame)
        wrists = skeleton_data["wrists"]
        mouth_region = skeleton_data["mouth_region"]
        leg_angles = skeleton_data.get("leg_angles", {})
        l_knee = leg_angles.get("left_knee_deg")
        r_knee = leg_angles.get("right_knee_deg")
        knee_angle = l_knee if l_knee is not None else r_knee
        if knee_angle is not None:
            self.moa1_knee_angle_deg = knee_angle
        hip_mid = skeleton_data.get("hip_mid")

        # 2. Run Object Detection for chair, cell phone, and bottle
        chair_box = None
        phone_box = None
        bottle_box = None
        best_chair_conf = 0.0
        best_phone_conf = 0.0
        best_bottle_conf = 0.0

        if not is_dummy_frame and self.det_model is not None:
            det_results = self._detect(frame)

            for box in det_results.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.det_model.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                if conf < 0.15:
                    continue

                if cls_name in ["chair", "couch"] and conf > best_chair_conf:
                    chair_box = (x1, y1, x2, y2)
                    best_chair_conf = conf
                elif cls_name in ["cell phone", "remote", "mouse", "wallet"] and conf > best_phone_conf:
                    phone_box = (x1, y1, x2, y2)
                    best_phone_conf = conf
                elif cls_name in ["bottle", "cup", "wine glass", "vase", "bowl"] and conf > best_bottle_conf:
                    bottle_box = (x1, y1, x2, y2)
                    best_bottle_conf = conf

        # Fallbacks & smoothing for chair, phone, bottle
        if chair_box is not None:
            self.moa1_chair_box = chair_box
            self.moa1_last_chair_box = chair_box
            if self.moa1_initial_chair_y is None:
                self.moa1_initial_chair_y = (chair_box[1] + chair_box[3]) / 2.0
        elif self.moa1_last_chair_box is not None:
            chair_box = self.moa1_last_chair_box

        if phone_box is not None:
            self.moa1_phone_box = phone_box
            self.moa1_last_phone_box = phone_box
            if self.moa1_initial_phone_y is None:
                self.moa1_initial_phone_y = (phone_box[1] + phone_box[3]) / 2.0
        elif self.moa1_last_phone_box is not None:
            phone_box = self.moa1_last_phone_box

        if bottle_box is not None:
            self.moa1_bottle_box = bottle_box
            self.moa1_last_bottle_box = bottle_box
            if self.moa1_initial_bottle_y is None:
                self.moa1_initial_bottle_y = (bottle_box[1] + bottle_box[3]) / 2.0
        elif self.moa1_last_bottle_box is not None:
            bottle_box = self.moa1_last_bottle_box

        def _min_dist_to_box(b: tuple[float, float, float, float] | None) -> float:
            if b is None or not wrists:
                return 999.0
            bx1, by1, bx2, by2 = b
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            min_d = 999.0
            for w_item in wrists:
                wx, wy = w_item["point"]
                d = math.hypot(wx - bcx, wy - bcy)
                if d < min_d:
                    min_d = d
            return min_d

        chair_wrist_dist = _min_dist_to_box(chair_box)
        phone_wrist_dist = _min_dist_to_box(phone_box)
        bottle_wrist_dist = _min_dist_to_box(bottle_box)

        # 3. Procedure Step Verification Logic
        # Precompute activity indicators for verification and out-of-order detection
        is_seated_pose = False
        if knee_angle is not None and (65.0 <= knee_angle <= 130.0):
            is_seated_pose = True
        elif hip_mid is not None and hip_mid[1] >= (h * 0.70):
            is_seated_pose = True

        phone_grasped = (phone_wrist_dist <= 75.0) or (phone_box is not None and phone_wrist_dist <= 95.0)
        is_phone_lifted = False
        if phone_box is not None and self.moa1_initial_phone_y is not None:
            pcy = (phone_box[1] + phone_box[3]) / 2.0
            if (self.moa1_initial_phone_y - pcy) >= 15.0:
                is_phone_lifted = True

        bottle_grasped = (bottle_wrist_dist <= 75.0) or (bottle_box is not None and bottle_wrist_dist <= 95.0)
        is_bottle_lifted = False
        if bottle_box is not None and self.moa1_initial_bottle_y is not None:
            bcy = (bottle_box[1] + bottle_box[3]) / 2.0
            if (self.moa1_initial_bottle_y - bcy) >= 20.0:
                is_bottle_lifted = True

        is_near_mouth = False
        if bottle_box is not None and mouth_region is not None:
            bcx, bcy = (bottle_box[0] + bottle_box[2]) / 2.0, bottle_box[1]
            if math.hypot(bcx - mouth_region[0], bcy - mouth_region[1]) <= 85.0:
                is_near_mouth = True

        dist_to_mouth = 999.0
        if mouth_region is not None:
            if bottle_box is not None:
                bcx = (bottle_box[0] + bottle_box[2]) / 2.0
                b_top_y = bottle_box[1]
                dist_to_mouth = min(dist_to_mouth, math.hypot(bcx - mouth_region[0], b_top_y - mouth_region[1]))
            for w_item in wrists:
                wx, wy = w_item["point"]
                dist_to_mouth = min(dist_to_mouth, math.hypot(wx - mouth_region[0], wy - mouth_region[1]))
        is_drinking = (dist_to_mouth <= 85.0)

        # 3. Procedure Step Verification Logic
        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None

        if active_step:
            active_step.elapsed_s += dt

            # -----------------------------------------------------------------
            # S01: Pull Chair into Position
            # -----------------------------------------------------------------
            if active_step.id == "S01":
                # Out-of-order checks for Step S01
                if (phone_grasped or is_phone_lifted) and not self.moa1_chair_pulled:
                    self._trigger_alert(
                        step_id="S01",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S01 requires positioning chair and sitting before picking up phone.",
                        tts="Warning: Step out of order. Pull chair and sit down before picking up phone.",
                        t=current_time,
                    )
                elif (bottle_grasped or is_bottle_lifted) and not self.moa1_chair_pulled:
                    self._trigger_alert(
                        step_id="S01",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S01 requires positioning chair and sitting before touching water bottle.",
                        tts="Warning: Step out of order. Pull chair and sit down before touching water bottle.",
                        t=current_time,
                    )

                hand_contact_chair = (chair_wrist_dist <= 110.0) or (chair_box is not None and chair_wrist_dist <= 140.0) or (len(wrists) > 0 and any(w["point"][1] > h * 0.48 for w in wrists))
                if hand_contact_chair or self.moa1_chair_pulled:
                    if self.moa1_s01_hold_start is None:
                        self.moa1_s01_hold_start = current_time
                    self.moa1_s01_hold_duration += dt
                    if self.moa1_s01_hold_duration >= 0.8:
                        self.moa1_chair_pulled = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S01",
                            severity="info",
                            kind="step_complete",
                            message="Chair pulled into position. Now sit down on the chair.",
                            tts="Chair positioned. Please sit down on the chair.",
                            t=current_time,
                        )
                else:
                    self.moa1_s01_hold_start = None
                    self.moa1_s01_hold_duration = max(0.0, self.moa1_s01_hold_duration - dt * 0.3)

            # -----------------------------------------------------------------
            # S02: Sit Down on Chair
            # -----------------------------------------------------------------
            elif active_step.id == "S02":
                # Out-of-order checks for Step S02
                if (phone_grasped or is_phone_lifted) and not (is_seated_pose or self.moa1_is_seated):
                    self._trigger_alert(
                        step_id="S02",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S02 requires sitting down on the chair before picking up smartphone.",
                        tts="Warning: Step out of order. Sit down on chair before picking up smartphone.",
                        t=current_time,
                    )
                elif (bottle_grasped or is_bottle_lifted) and not (is_seated_pose or self.moa1_is_seated):
                    self._trigger_alert(
                        step_id="S02",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S02 requires sitting down on the chair before grasping water bottle.",
                        tts="Warning: Step out of order. Sit down on chair before grasping water bottle.",
                        t=current_time,
                    )

                is_seated_now = is_seated_pose or self.moa1_is_seated or (hip_mid is not None and hip_mid[1] > h * 0.50) or (len(wrists) > 0 and any(w["point"][1] > h * 0.45 for w in wrists))
                if is_seated_now:
                    if self.moa1_s02_hold_start is None:
                        self.moa1_s02_hold_start = current_time
                    self.moa1_s02_hold_duration += dt
                    if self.moa1_s02_hold_duration >= 0.8:
                        self.moa1_is_seated = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S02",
                            severity="info",
                            kind="step_complete",
                            message="Seated posture confirmed. Now pick up the smartphone from the desk.",
                            tts="Seated posture confirmed. Pick up the smartphone.",
                            t=current_time,
                        )
                else:
                    self.moa1_s02_hold_start = None
                    self.moa1_s02_hold_duration = max(0.0, self.moa1_s02_hold_duration - dt * 0.3)

            # -----------------------------------------------------------------
            # S03: Pick Up Smartphone
            # -----------------------------------------------------------------
            elif active_step.id == "S03":
                # Out-of-order checks for Step S03
                if (bottle_grasped or is_bottle_lifted) and not self.moa1_phone_picked:
                    self._trigger_alert(
                        step_id="S03",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S03 requires picking up smartphone before the water bottle.",
                        tts="Warning: Step out of order. Pick up smartphone before interacting with water bottle.",
                        t=current_time,
                    )

                if phone_grasped or is_phone_lifted or self.moa1_phone_picked:
                    if self.moa1_s03_hold_start is None:
                        self.moa1_s03_hold_start = current_time
                    self.moa1_s03_hold_duration += dt
                    if self.moa1_s03_hold_duration >= 0.6:
                        self.moa1_phone_picked = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S03",
                            severity="info",
                            kind="step_complete",
                            message="Smartphone picked up. Place it back down onto the desk surface.",
                            tts="Phone picked up. Return the smartphone to the desk.",
                            t=current_time,
                        )
                else:
                    self.moa1_s03_hold_start = None
                    self.moa1_s03_hold_duration = max(0.0, self.moa1_s03_hold_duration - dt * 0.3)

            # -----------------------------------------------------------------
            # S04: Return Smartphone to Desk
            # -----------------------------------------------------------------
            elif active_step.id == "S04":
                # Out-of-order checks for Step S04
                if (bottle_grasped or is_bottle_lifted) and not self.moa1_phone_stowed:
                    self._trigger_alert(
                        step_id="S04",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S04 requires stowing smartphone back onto desk before lifting water bottle.",
                        tts="Warning: Step out of order. Return phone to desk before lifting water bottle.",
                        t=current_time,
                    )

                phone_stowed = (phone_wrist_dist >= 60.0) or (phone_box is not None and phone_wrist_dist >= 55.0)
                if phone_stowed or self.moa1_phone_stowed:
                    if self.moa1_s04_hold_start is None:
                        self.moa1_s04_hold_start = current_time
                    self.moa1_s04_hold_duration += dt
                    if self.moa1_s04_hold_duration >= 0.6:
                        self.moa1_phone_stowed = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S04",
                            severity="info",
                            kind="step_complete",
                            message="Smartphone stowed on desk. Next, grasp and lift the water bottle.",
                            tts="Smartphone stowed. Grasp and lift the water bottle.",
                            t=current_time,
                        )
                else:
                    self.moa1_s04_hold_start = None
                    self.moa1_s04_hold_duration = max(0.0, self.moa1_s04_hold_duration - dt * 0.3)

            # -----------------------------------------------------------------
            # S05: Grasp and Lift Water Bottle
            # -----------------------------------------------------------------
            elif active_step.id == "S05":
                if (bottle_grasped and (is_bottle_lifted or is_near_mouth)) or is_near_mouth or self.moa1_bottle_lifted:
                    if self.moa1_s05_hold_start is None:
                        self.moa1_s05_hold_start = current_time
                    self.moa1_s05_hold_duration += dt
                    if self.moa1_s05_hold_duration >= 0.6 or is_near_mouth:
                        self.moa1_bottle_lifted = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S05",
                            severity="info",
                            kind="step_complete",
                            message="Water bottle lifted. Bring to mouth and drink water (hold >= 1.5s).",
                            tts="Bottle lifted. Bring to mouth and drink water.",
                            t=current_time,
                        )
                else:
                    self.moa1_s05_hold_start = None
                    self.moa1_s05_hold_duration = max(0.0, self.moa1_s05_hold_duration - dt * 0.3)

            # -----------------------------------------------------------------
            # S06: Drink Water from Bottle
            # -----------------------------------------------------------------
            elif active_step.id == "S06":
                if is_drinking or self.moa1_water_consumed:
                    if self.moa1_drinking_hold_start is None:
                        self.moa1_drinking_hold_start = current_time
                    self.moa1_drinking_hold_duration += dt
                    self.moa1_last_drinking_seen_time = current_time

                    if self.moa1_drinking_hold_duration >= 1.5:
                        self.moa1_water_consumed = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S06",
                            severity="info",
                            kind="step_complete",
                            message="Drinking verified (held >= 1.5s). Return bottle to table and release hands.",
                            tts="Water consumed. Return bottle to table and release hands.",
                            t=current_time,
                        )
                else:
                    # Check for skipped drinking (bottle returned to table plane without drinking)
                    if bottle_box is not None and self.moa1_initial_bottle_y is not None:
                        bcy = (bottle_box[1] + bottle_box[3]) / 2.0
                        if abs(bcy - self.moa1_initial_bottle_y) < 25.0 and bottle_wrist_dist > 55.0 and self.moa1_drinking_hold_duration < 1.0:
                            self._trigger_alert(
                                step_id="S06",
                                severity="critical",
                                kind="skipped",
                                message="Step S06 skipped! Water bottle returned to table without drinking (hold >= 1.5s required).",
                                tts="Warning: Step skipped. Drink water from the bottle before returning it.",
                                t=current_time,
                            )
                            active_step.status = "skipped"
                    if (current_time - self.moa1_last_drinking_seen_time) > 1.6:
                        self.moa1_drinking_hold_start = None
                        self.moa1_drinking_hold_duration = max(0.0, self.moa1_drinking_hold_duration - dt * 0.2)

            # -----------------------------------------------------------------
            # S07: Return Bottle to Table & Release Hands
            # -----------------------------------------------------------------
            elif active_step.id == "S07":
                # Out-of-order check: if astronaut drinks again or holds bottle to mouth
                if is_drinking:
                    self._trigger_alert(
                        step_id="S07",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S07 requires returning bottle to table and releasing hands.",
                        tts="Warning: Step out of order. Return bottle to table and release hands.",
                        t=current_time,
                    )

                hands_released = (bottle_wrist_dist >= 65.0)
                is_on_table = True
                if bottle_box is not None and self.moa1_initial_bottle_y is not None:
                    bcy = (bottle_box[1] + bottle_box[3]) / 2.0
                    is_on_table = (bcy >= self.moa1_initial_bottle_y - 25.0)

                if (hands_released and is_on_table) or self.moa1_bottle_returned:
                    if self.moa1_s07_hold_start is None:
                        self.moa1_s07_hold_start = current_time
                    self.moa1_s07_hold_duration += dt
                    if self.moa1_s07_hold_duration >= 0.6:
                        self.moa1_bottle_returned = True
                        self.protocol_complete = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S07",
                            severity="info",
                            kind="nominal_completion",
                            message="Multi-Object Experiment Complete! All seven activities verified nominal.",
                            tts="Multi-object experiment complete. All seven steps verified nominal.",
                            t=current_time,
                        )
                else:
                    self.moa1_s07_hold_start = None
                    self.moa1_s07_hold_duration = max(0.0, self.moa1_s07_hold_duration - dt * 0.3)

        # 4. Render HUD
        self.overlay_objects = [o for o in (
            {"label": "chair", "cls": "chair", "box": list(chair_box), "conf": round(best_chair_conf, 2),
             "state": ["PULLED"] if self.moa1_chair_pulled else [], "color": "#22d3ee"} if chair_box is not None else None,
            {"label": "phone", "cls": "cell phone", "box": list(phone_box), "conf": round(best_phone_conf, 2),
             "state": ["PICKED"] if self.moa1_phone_picked else [], "color": "#22d3ee"} if phone_box is not None else None,
            {"label": "bottle", "cls": "bottle", "box": list(bottle_box), "conf": round(best_bottle_conf, 2),
             "state": ["LIFTED"] if self.moa1_bottle_lifted else [], "color": "#22d3ee"} if bottle_box is not None else None,
        ) if o is not None]
        annotated = frame.copy() if self.render_hud else frame
        if self.render_hud: self._render_moa1_hud(
            annotated,
            chair_box=chair_box,
            phone_box=phone_box,
            bottle_box=bottle_box,
            chair_pulled=self.moa1_chair_pulled,
            is_seated=self.moa1_is_seated,
            knee_angle=self.moa1_knee_angle_deg,
            phone_picked=self.moa1_phone_picked,
            phone_stowed=self.moa1_phone_stowed,
            bottle_lifted=self.moa1_bottle_lifted,
            water_consumed=self.moa1_water_consumed,
            drinking_hold_s=self.moa1_drinking_hold_duration,
            bottle_returned=self.moa1_bottle_returned,
            active_step=active_step,
            current_time=current_time,
            skeleton_data=skeleton_data,
        )

        # 5. Format Telemetry
        completed_count = sum(1 for s in self.steps if s.status == "completed")
        compliance_pct = int((completed_count / len(self.steps)) * 100) if self.steps else 100

        step_telemetry = []
        for s in self.steps:
            is_verifying = False
            pct = 0
            if s.status == "completed":
                pct = 100
            elif s.status == "active":
                if s.id == "S01":
                    is_verifying = self.moa1_s01_hold_duration > 0
                    pct = min(100, int((self.moa1_s01_hold_duration / 0.8) * 100))
                elif s.id == "S02":
                    is_verifying = self.moa1_s02_hold_duration > 0
                    pct = min(100, int((self.moa1_s02_hold_duration / 0.8) * 100))
                elif s.id == "S03":
                    is_verifying = self.moa1_s03_hold_duration > 0
                    pct = min(100, int((self.moa1_s03_hold_duration / 0.6) * 100))
                elif s.id == "S04":
                    is_verifying = self.moa1_s04_hold_duration > 0
                    pct = min(100, int((self.moa1_s04_hold_duration / 0.6) * 100))
                elif s.id == "S05":
                    is_verifying = self.moa1_s05_hold_duration > 0
                    pct = min(100, int((self.moa1_s05_hold_duration / 0.6) * 100))
                elif s.id == "S06":
                    is_verifying = self.moa1_drinking_hold_duration > 0
                    pct = min(100, int((self.moa1_drinking_hold_duration / 1.5) * 100))
                elif s.id == "S07":
                    is_verifying = self.moa1_s07_hold_duration > 0
                    pct = min(100, int((self.moa1_s07_hold_duration / 0.6) * 100))

            step_telemetry.append({
                "id": s.id,
                "name": s.name,
                "prompt": s.prompt,
                "status": s.status,
                "elapsed_s": round(s.elapsed_s, 1),
                "is_verifying": is_verifying,
                "verification_pct": pct,
            })

        recent_alert_payload = None
        if self.recent_alerts and (current_time - self.recent_alerts[-1].timestamp) < 4.0:
            a = self.recent_alerts[-1]
            recent_alert_payload = {
                "step_id": a.step_id,
                "severity": a.severity,
                "kind": a.kind,
                "message": a.message,
                "tts": a.spoken_tts,
                "timestamp": a.timestamp,
            }

        telemetry = {
            "experiment_id": self.experiment_id,
            "experiment_title": self.title,
            "rack_id": self.rack_id,
            "frame_idx": self.frame_count,
            "active_step_id": active_step.id if active_step else "DONE",
            "active_step_name": active_step.name if active_step else "Protocol Completed",
            "prompt": active_step.prompt if active_step else "Multi-object experiment completed nominally.",
            "compliance_score": compliance_pct,
            "is_complete": self.protocol_complete,
            "chair": {
                "detected": chair_box is not None,
                "pulled": self.moa1_chair_pulled,
                "hold_s": round(self.moa1_s01_hold_duration, 2),
                "target_s": 0.8,
                "progress_pct": min(100, int((self.moa1_s01_hold_duration / 0.8) * 100)),
            },
            "posture": {
                "is_seated": self.moa1_is_seated,
                "knee_angle_deg": round(self.moa1_knee_angle_deg, 1) if self.moa1_knee_angle_deg is not None else None,
                "hold_s": round(self.moa1_s02_hold_duration, 2),
                "target_s": 0.8,
                "progress_pct": min(100, int((self.moa1_s02_hold_duration / 0.8) * 100)),
                "status": skeleton_data["posture_status"],
            },
            "phone": {
                "detected": phone_box is not None,
                "picked": self.moa1_phone_picked,
                "stowed": self.moa1_phone_stowed,
                "hold_s": round(self.moa1_s03_hold_duration if not self.moa1_phone_picked else self.moa1_s04_hold_duration, 2),
                "progress_pct": min(100, int(((self.moa1_s03_hold_duration if not self.moa1_phone_picked else self.moa1_s04_hold_duration) / 0.6) * 100)),
            },
            "drinking": {
                "in_progress": self.moa1_drinking_hold_duration > 0,
                "hold_duration_s": round(self.moa1_drinking_hold_duration, 2),
                "target_duration_s": 1.5,
                "progress_pct": min(100, int((self.moa1_drinking_hold_duration / 1.5) * 100)),
                "water_consumed": self.moa1_water_consumed,
            },
            "bottle": {
                "detected": bottle_box is not None,
                "lifted": self.moa1_bottle_lifted,
                "returned": self.moa1_bottle_returned,
                "hands_released": self.moa1_bottle_returned or (bottle_wrist_dist >= 65.0),
                "hold_duration_s": round(self.moa1_s07_hold_duration, 2),
                "progress_pct": min(100, int((self.moa1_s07_hold_duration / 0.6) * 100)),
            },
            "steps": step_telemetry,
            "geometry": {
                "target_detected": (bottle_box is not None) or (phone_box is not None) or (chair_box is not None),
                "target_confidence": round(max(best_bottle_conf, best_phone_conf, best_chair_conf), 2) if (best_bottle_conf or best_phone_conf or best_chair_conf) else 0.85,
                "person_detected": skeleton_data["detected"],
                "full_body_detected": skeleton_data["detected"] and skeleton_data["body_points_count"] >= 8,
                "body_points_count": skeleton_data["body_points_count"],
                "body_points_total": 17,
                "posture_status": skeleton_data["posture_status"],
                "posture_stability": skeleton_data["posture_stability"],
                "torso_angle_deg": skeleton_data["torso_angle_deg"],
                "knee_angle_deg": round(self.moa1_knee_angle_deg, 1) if self.moa1_knee_angle_deg is not None else 90.0,
                "chair_pulled": self.moa1_chair_pulled,
                "is_seated": self.moa1_is_seated,
                "phone_picked": self.moa1_phone_picked,
                "phone_stowed": self.moa1_phone_stowed,
                "bottle_lifted": self.moa1_bottle_lifted,
                "water_consumed": self.moa1_water_consumed,
                "bottle_returned": self.moa1_bottle_returned,
                "skeleton": skeleton_data["keypoints"],
                "hand_contact": (bottle_wrist_dist <= 85.0) or (phone_wrist_dist <= 85.0) or (chair_wrist_dist <= 110.0),
                "closest_wrist_px": round(min(bottle_wrist_dist, phone_wrist_dist, chair_wrist_dist), 1) if min(bottle_wrist_dist, phone_wrist_dist, chair_wrist_dist) < 900 else None,
                "is_lifted": self.moa1_bottle_lifted or is_phone_lifted,
                "lift_delta_px": 30.0 if (self.moa1_bottle_lifted or is_phone_lifted) else 0.0,
                "near_mouth": is_near_mouth,
                "is_drinking_pose": is_drinking,
                "in_table_zone": True,
                "dist_to_mouth_px": round(dist_to_mouth, 1) if dist_to_mouth < 900 else None,
                "body_zones": skeleton_data["zones"],
                "body_velocity_px_s": skeleton_data["velocity_px_s"],
                "arm_angles": skeleton_data["arm_angles"],
                "anchored": skeleton_data["anchored"],
            },
            "recent_alert": recent_alert_payload,
            "alert_count": len(self.alerts),
        }

        return annotated, telemetry

    def _render_moa1_hud(
        self,
        img: np.ndarray,
        chair_box: tuple[float, float, float, float] | None,
        phone_box: tuple[float, float, float, float] | None,
        bottle_box: tuple[float, float, float, float] | None,
        chair_pulled: bool,
        is_seated: bool,
        knee_angle: float | None,
        phone_picked: bool,
        phone_stowed: bool,
        bottle_lifted: bool,
        water_consumed: bool,
        drinking_hold_s: float,
        bottle_returned: bool,
        active_step: StepState | None,
        current_time: float,
        skeleton_data: dict[str, Any] | None,
    ) -> None:
        """Renders cyberpunk aerospace HUD for MOA-1 Multi-Object Experiment."""
        h, w = img.shape[:2]
        overlay = img.copy()

        # 1. Render Astronaut 17-Point Skeleton Rig
        self._render_skeletal_rig(img, skeleton_data)

        # 2. Top Banner Background (Translucent Glassmorphic Dark)
        cv2.rectangle(overlay, (0, 0), (w, 54), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, 54), (w, 54), (45, 91, 216), 1)

        # Title
        cv2.putText(img, "PARIKSHAK MISSION CONTROL", (18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 230, 80), 1, cv2.LINE_AA)
        cv2.putText(img, "MOA-1: MULTI-OBJECT HAR", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

        # Top Status Badges
        # Chair Badge
        chair_col = (60, 230, 100) if chair_pulled else (240, 180, 40)
        chair_txt = "CHAIR: PULLED" if chair_pulled else "CHAIR: TARGET"
        cv2.rectangle(img, (w - 380, 10), (w - 275, 42), (20, 25, 35), -1)
        cv2.rectangle(img, (w - 380, 10), (w - 275, 42), chair_col, 1)
        cv2.putText(img, chair_txt, (w - 372, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, chair_col, 1, cv2.LINE_AA)

        # Sitting Posture Badge
        seat_col = (60, 230, 100) if is_seated else (240, 180, 40)
        seat_txt = f"SEATED {int(knee_angle)}deg" if (is_seated and knee_angle) else ("SEATED" if is_seated else "POSTURE")
        cv2.rectangle(img, (w - 265, 10), (w - 150, 42), (20, 25, 35), -1)
        cv2.rectangle(img, (w - 265, 10), (w - 150, 42), seat_col, 1)
        cv2.putText(img, seat_txt, (w - 257, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, seat_col, 1, cv2.LINE_AA)

        # Drinking / Bottle Badge
        drink_col = (60, 230, 100) if water_consumed else (0, 220, 255)
        drink_txt = "WATER: CONSUMED" if water_consumed else (f"DRINK: {drinking_hold_s:.1f}s" if drinking_hold_s > 0 else "BOTTLE: READY")
        cv2.rectangle(img, (w - 140, 10), (w - 10, 42), (20, 25, 35), -1)
        cv2.rectangle(img, (w - 140, 10), (w - 10, 42), drink_col, 1)
        cv2.putText(img, drink_txt, (w - 132, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.34, drink_col, 1, cv2.LINE_AA)

        # 3. Draw Bounding Boxes
        # A. Chair Box (Cyan / Electric Blue)
        if chair_box is not None:
            cx1, cy1, cx2, cy2 = [int(v) for v in chair_box]
            ch_col = (60, 230, 100) if chair_pulled else (255, 220, 0)
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), ch_col, 2)
            cv2.rectangle(img, (cx1, cy1 - 18), (cx1 + 140, cy1), ch_col, -1)
            ch_lbl = "CHAIR [PULLED]" if chair_pulled else "CHAIR [POSITION]"
            cv2.putText(img, ch_lbl, (cx1 + 4, cy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 15, 25), 1, cv2.LINE_AA)

        # B. Phone Box (Amber / Gold)
        if phone_box is not None:
            px1, py1, px2, py2 = [int(v) for v in phone_box]
            ph_col = (60, 230, 100) if phone_stowed else ((0, 180, 255) if phone_picked else (0, 220, 255))
            cv2.rectangle(img, (px1, py1), (px2, py2), ph_col, 2)
            cv2.rectangle(img, (px1, py1 - 18), (px1 + 130, py1), ph_col, -1)
            ph_lbl = "PHONE [STOWED]" if phone_stowed else ("PHONE [LIFTED]" if phone_picked else "PHONE [DESK]")
            cv2.putText(img, ph_lbl, (px1 + 4, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 15, 25), 1, cv2.LINE_AA)

        # C. Bottle Box (Emerald Green)
        if bottle_box is not None:
            bx1, by1, bx2, by2 = [int(v) for v in bottle_box]
            bt_col = (60, 230, 100) if water_consumed else (60, 240, 120)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), bt_col, 2)
            cv2.rectangle(img, (bx1, by1 - 18), (bx1 + 130, by1), bt_col, -1)
            bt_lbl = "BOTTLE [DRUNK]" if water_consumed else ("BOTTLE [LIFTED]" if bottle_lifted else "BOTTLE [TABLE]")
            cv2.putText(img, bt_lbl, (bx1 + 4, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 15, 25), 1, cv2.LINE_AA)

        # 4. Step Progress Bar
        if active_step:
            if active_step.id == "S01":
                pct = min(1.0, self.moa1_s01_hold_duration / 0.8)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (255, 200, 0), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (255, 230, 80), 1)
                cv2.putText(img, f"PULL CHAIR INTO DESK: {int(pct * 100)}%", (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S02":
                pct = min(1.0, self.moa1_s02_hold_duration / 0.8)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (40, 220, 120), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 250, 160), 1)
                angle_str = f" ({int(knee_angle)}deg)" if knee_angle else ""
                cv2.putText(img, f"SIT DOWN ON CHAIR: {int(pct * 100)}%{angle_str}", (w // 2 - 125, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S03":
                pct = min(1.0, self.moa1_s03_hold_duration / 0.6)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (0, 180, 255), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 220, 255), 1)
                cv2.putText(img, f"PICK UP SMARTPHONE: {int(pct * 100)}%", (w // 2 - 115, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 15, 25), 1, cv2.LINE_AA)

            elif active_step.id == "S04":
                pct = min(1.0, self.moa1_s04_hold_duration / 0.6)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (0, 220, 240), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 240, 255), 1)
                cv2.putText(img, f"STOW SMARTPHONE ON DESK: {int(pct * 100)}%", (w // 2 - 130, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 15, 25), 1, cv2.LINE_AA)

            elif active_step.id == "S05":
                pct = min(1.0, self.moa1_s05_hold_duration / 0.6)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (60, 230, 100), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (100, 255, 140), 1)
                cv2.putText(img, f"GRASP & LIFT BOTTLE: {int(pct * 100)}%", (w // 2 - 110, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S06":
                pct = min(1.0, drinking_hold_s / 1.5)
                bar_w = int(280 * pct)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                fill_color = (60, 230, 100) if drinking_hold_s > 0 else (40, 140, 255)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + bar_w, 88), fill_color, -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 180, 250), 1)
                drink_lbl = f"DRINKING HOLD: {int(pct * 100)}% ({drinking_hold_s:.1f}s / 1.5s)" if drinking_hold_s > 0 else "BRING BOTTLE TO MOUTH TO DRINK"
                cv2.putText(img, drink_lbl, (w // 2 - 130, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S07":
                pct = min(1.0, self.moa1_s07_hold_duration / 0.6)
                bar_w = int(280 * pct)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + bar_w, 88), (60, 230, 100), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 240, 180), 1)
                cv2.putText(img, f"RETURN BOTTLE & RELEASE: {int(pct * 100)}%", (w // 2 - 125, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        elif self.protocol_complete:
            cv2.rectangle(img, (w // 2 - 150, 62), (w // 2 + 150, 88), (20, 35, 25), -1)
            cv2.rectangle(img, (w // 2 - 150, 62), (w // 2 + 150, 88), (60, 220, 100), -1)
            cv2.putText(img, "ALL 7 ACTIVITIES VERIFIED NOMINAL", (w // 2 - 135, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Bottom Prompt & Alert Bar
        has_alert = bool(self.recent_alerts and (current_time - self.recent_alerts[-1].timestamp) < 4.0)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, h - 60), (w, h - 60), (45, 91, 216), 1)

        if has_alert:
            alert = self.recent_alerts[-1]
            cv2.putText(img, f"ALERT [{alert.kind.upper()}]: {alert.message}", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 255), 2, cv2.LINE_AA)
        elif active_step:
            step_prompt = f"[{active_step.id}] {active_step.prompt}"
            cv2.putText(img, step_prompt, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "All procedure steps verified successfully.", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 120), 1, cv2.LINE_AA)

    def _process_bcx1_frame(
        self, frame: np.ndarray, current_time: float, is_dummy_frame: bool
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Processes one video frame for BCX-1: Two-Box Collision in Container."""
        h, w = frame.shape[:2]
        dt = (current_time - self.last_frame_time) if self.last_frame_time else 0.05
        dt = min(max(dt, 0.01), 0.35)
        self.last_frame_time = current_time

        # Extract Full-Body 17-Point Skeleton & Posture
        skeleton_data = self._extract_full_body_skeleton(frame, current_time, dt, is_dummy_frame)

        # 1. Dual-Space Color Segmentation for Red and Yellow Boxes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b_ch, g_ch, r_ch = cv2.split(frame)
        k_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # --- RED DETECTION (Dual HSV range + RGB Red Dominance) ---
        mask_r1 = cv2.inRange(hsv, np.array([0, 32, 28]), np.array([14, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([160, 32, 28]), np.array([180, 255, 255]))
        hsv_red = cv2.bitwise_or(mask_r1, mask_r2)

        # Red dominance: R higher than G and B with adaptive floor
        r_int = r_ch.astype(np.int16)
        g_int = g_ch.astype(np.int16)
        b_int = b_ch.astype(np.int16)
        rgb_red = ((r_int - g_int > 12) & (r_int - b_int > 12) & (r_ch > 42)).astype(np.uint8) * 255

        # Skin is red-orange in HSV and passes loose red/yellow tests, so faces and
        # hands were being detected as boxes. Exclude the YCrCb skin cluster and
        # require real saturation (painted boxes are far more saturated than skin).
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycc, np.array([0, 133, 77]), np.array([255, 173, 127]))
        not_skin = cv2.bitwise_not(skin)
        sat_ok = cv2.inRange(hsv, np.array([0, 90, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_and(cv2.bitwise_and(hsv_red, rgb_red), cv2.bitwise_and(not_skin, sat_ok))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k_clean)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k_clean)

        # --- YELLOW DETECTION (HSV [12, 35] + RGB Yellow Dominance) ---
        hsv_yellow = cv2.inRange(hsv, np.array([12, 35, 35]), np.array([42, 255, 255]))
        rgb_yellow = (
            (r_ch > 48)
            & (g_ch > 45)
            & (r_int - b_int > 16)
            & (g_int - b_int > 14)
            & (np.abs(r_int - g_int) < 75)
        ).astype(np.uint8) * 255

        mask_yellow = cv2.bitwise_and(cv2.bitwise_and(hsv_yellow, rgb_yellow), cv2.bitwise_and(not_skin, sat_ok))
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k_clean)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, k_clean)

        # Contour extraction for Red Box (minimum area 180px for distant or compact boxes)
        cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_box = None
        if cnts_red:
            best_c = max(cnts_red, key=cv2.contourArea)
            if cv2.contourArea(best_c) >= 180.0:
                rx, ry, rw, rh = cv2.boundingRect(best_c)
                red_box = (float(rx), float(ry), float(rx + rw), float(ry + rh))
                self.bcx1_last_red_box = red_box
                self.bcx1_last_seen_red_time = current_time
        elif self.bcx1_last_red_box and (current_time - self.bcx1_last_seen_red_time) < 1.0:
            # Grace period for hand occlusion during handling
            red_box = self.bcx1_last_red_box

        # Contour extraction for Yellow Box (minimum area 180px)
        cnts_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_box = None
        if cnts_yellow:
            best_y = max(cnts_yellow, key=cv2.contourArea)
            if cv2.contourArea(best_y) >= 180.0:
                yx, yy, yw, yh = cv2.boundingRect(best_y)
                yellow_box = (float(yx), float(yy), float(yx + yw), float(yy + yh))
                self.bcx1_last_yellow_box = yellow_box
                self.bcx1_last_seen_yellow_time = current_time
        elif self.bcx1_last_yellow_box and (current_time - self.bcx1_last_seen_yellow_time) < 1.0:
            # Grace period for hand occlusion during handling
            yellow_box = self.bcx1_last_yellow_box

        # 2. Outer Container Box Detection & Exponential Moving Average (EMA) Locking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 110)
        k_cont = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges_sealed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_cont)
        cnts_edges, _ = cv2.findContours(edges_sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_container = None
        best_cont_area = 0.0
        # The crew member's own silhouette is the biggest edge contour in the
        # frame; a candidate that contains the shoulders/hips is the person, not
        # the container.
        torso_pts = [(k["x"], k["y"]) for k in skeleton_data.get("keypoints", [])
                     if k.get("visible") and k.get("id") in (5, 6, 11, 12)]
        for c in cnts_edges:
            ox, oy, ow, oh = cv2.boundingRect(c)
            b_area = float(ow * oh)
            c_area = cv2.contourArea(c)
            eff_area = max(b_area, c_area)
            if sum(1 for px_, py_ in torso_pts if ox <= px_ <= ox + ow and oy <= py_ <= oy + oh) >= 2:
                continue
            if 15000.0 <= eff_area <= 280000.0 and ow >= 150 and oh >= 110:
                if (float(ow) / max(1.0, float(oh))) < 3.5:
                    if eff_area > best_cont_area:
                        best_cont_area = eff_area
                        candidate_container = (float(ox), float(oy), float(ox + ow), float(oy + oh))

        if candidate_container is not None:
            if self.bcx1_locked_container is None:
                self.bcx1_locked_container = candidate_container
            else:
                # Smooth coordinates with EMA so box never flickers or detaches
                prev = self.bcx1_locked_container
                smoothed = (
                    prev[0] * 0.85 + candidate_container[0] * 0.15,
                    prev[1] * 0.85 + candidate_container[1] * 0.15,
                    prev[2] * 0.85 + candidate_container[2] * 0.15,
                    prev[3] * 0.85 + candidate_container[3] * 0.15,
                )
                self.bcx1_locked_container = smoothed
            self.bcx1_container_locked = True
        elif self.bcx1_locked_container is None:
            # Default workspace container if not yet detected
            self.bcx1_locked_container = (60.0, 50.0, float(w - 60), float(h - 70))

        outer_box = self.bcx1_locked_container
        self.bcx1_outer_box = outer_box
        self.bcx1_red_box = red_box
        self.bcx1_yellow_box = yellow_box

        # Containment logic: center of box inside outer_box with 16px margin
        ox1, oy1, ox2, oy2 = outer_box
        red_inside = False
        if red_box:
            rcx = (red_box[0] + red_box[2]) / 2.0
            rcy = (red_box[1] + red_box[3]) / 2.0
            red_inside = (ox1 - 16.0 <= rcx <= ox2 + 16.0) and (oy1 - 16.0 <= rcy <= oy2 + 16.0)
        self.bcx1_red_inside = red_inside

        yellow_inside = False
        if yellow_box:
            ycx = (yellow_box[0] + yellow_box[2]) / 2.0
            ycy = (yellow_box[1] + yellow_box[3]) / 2.0
            yellow_inside = (ox1 - 16.0 <= ycx <= ox2 + 16.0) and (oy1 - 16.0 <= ycy <= oy2 + 16.0)
        self.bcx1_yellow_inside = yellow_inside

        # 3. Collision & Distance Dynamics
        dist_px = 999.0
        is_colliding = False
        boxes_overlap = False
        if red_box and yellow_box:
            rx1, ry1, rx2, ry2 = red_box
            yx1, yy1, yx2, yy2 = yellow_box
            dx = max(0.0, max(rx1 - yx2, yx1 - rx2))
            dy = max(0.0, max(ry1 - yy2, yy1 - ry2))
            dist_px = math.hypot(dx, dy)
            boxes_overlap = (max(rx1, yx1) < min(rx2, yx2)) and (max(ry1, yy1) < min(ry2, yy2))
            if dist_px <= 20.0 or boxes_overlap:
                is_colliding = True

        self.bcx1_distance_px = dist_px
        self.bcx1_collision = is_colliding

        # 4. Procedure Compliance State Machine for 6-Step BCX-1 Protocol
        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None

        if active_step is not None and not is_dummy_frame:
            active_step.elapsed_s = current_time - (active_step.started_at or current_time)

            # S01: Identify Bigger Container Box
            if active_step.id == "S01":
                container_detected = self.bcx1_container_locked or (candidate_container is not None)
                if container_detected:
                    self.bcx1_s01_hold_duration += dt
                    if self.bcx1_s01_hold_duration >= 0.8:
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S01",
                            severity="info",
                            kind="nominal_step",
                            message="Container box identified and locked. Proceed to Step Two: Present Red and Yellow boxes to verify colors.",
                            tts="Container box identified and locked. Proceed to step two: present the Red and Yellow boxes to verify colors.",
                            t=current_time,
                        )
                else:
                    self.bcx1_s01_hold_duration = max(0.0, self.bcx1_s01_hold_duration - (dt * 0.5))

            # S02: Detect & Verify Box Colors (Red & Yellow)
            elif active_step.id == "S02":
                # Check for premature placement before color calibration
                if (red_inside or yellow_inside) and not self.bcx1_colors_verified:
                    if not (red_box and yellow_box):
                        self._trigger_alert(
                            step_id="S02",
                            severity="caution",
                            kind="skipped",
                            message="Present both Red and Yellow boxes to the camera to calibrate colors before placing them.",
                            tts="Caution: Present both the Red and Yellow boxes to the camera to verify colors first.",
                            t=current_time,
                        )

                both_colors_present = (red_box is not None) and (yellow_box is not None)
                if both_colors_present:
                    self.bcx1_s02_hold_duration += dt
                    if self.bcx1_s02_hold_duration >= 0.8:
                        self.bcx1_colors_verified = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S02",
                            severity="info",
                            kind="nominal_step",
                            message="Red and Yellow box colors verified accurately. Proceed to Step Three: Place Red Box into container.",
                            tts="Colors verified accurately. Proceed to step three: place the Red Box inside the container.",
                            t=current_time,
                        )
                else:
                    self.bcx1_s02_hold_duration = max(0.0, self.bcx1_s02_hold_duration - (dt * 0.4))

            # S03: Place Red Box into Container
            elif active_step.id == "S03":
                # Check if Yellow box placed prematurely without Red box
                if yellow_inside and not red_inside:
                    self._trigger_alert(
                        step_id="S03",
                        severity="critical",
                        kind="out_of_order",
                        message="Out of order! Step S03 requires placing Red Box first. Yellow box detected inside.",
                        tts="Warning: Step out of order. Place the Red Box inside the container first.",
                        t=current_time,
                    )
                elif red_inside:
                    self.bcx1_s03_hold_duration += dt
                    if self.bcx1_s03_hold_duration >= 0.8:
                        self.bcx1_red_placed = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S03",
                            severity="info",
                            kind="nominal_step",
                            message="Red Box verified inside container. Proceed to Step Four: Place Yellow Box into container.",
                            tts="Step three verified. Red Box is inside container. Proceed to step four: place the Yellow Box inside.",
                            t=current_time,
                        )
                else:
                    self.bcx1_s03_hold_duration = max(0.0, self.bcx1_s03_hold_duration - (dt * 0.4))

            # S04: Place Yellow Box into Container (separated)
            elif active_step.id == "S04":
                # Check premature collision
                if is_colliding and (red_inside or yellow_inside):
                    self._trigger_alert(
                        step_id="S04",
                        severity="caution",
                        kind="out_of_order",
                        message="Premature Collision! Place Yellow Box separated from Red Box before colliding.",
                        tts="Caution: Place the Yellow Box inside separated from the Red Box before colliding.",
                        t=current_time,
                    )
                elif yellow_inside and red_inside and dist_px > 25.0:
                    self.bcx1_s04_hold_duration += dt
                    if self.bcx1_s04_hold_duration >= 0.8:
                        self.bcx1_yellow_placed = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S04",
                            severity="info",
                            kind="nominal_step",
                            message="Yellow Box placed and verified separated. Proceed to Step Five: Bring boxes together to collide.",
                            tts="Step four verified. Both boxes inside container. Proceed to step five: bring boxes together to collide.",
                            t=current_time,
                        )
                else:
                    self.bcx1_s04_hold_duration = max(0.0, self.bcx1_s04_hold_duration - (dt * 0.4))

            # S05: Collide Red and Yellow Boxes
            elif active_step.id == "S05":
                # Check out-of-bounds hazard: colliding outside container
                if is_colliding and not (red_inside and yellow_inside):
                    self._trigger_alert(
                        step_id="S05",
                        severity="critical",
                        kind="hazard",
                        message="Out-of-Bounds Hazard! Collision must take place inside the container.",
                        tts="Hazard warning: Collision must occur inside the container.",
                        t=current_time,
                    )
                elif is_colliding and red_inside and yellow_inside:
                    self.bcx1_collision_hold_duration += dt
                    self.bcx1_last_collision_time = current_time
                    if self.bcx1_collision_hold_duration >= 0.8:
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S05",
                            severity="info",
                            kind="collision_confirmed",
                            message="Box Collision Confirmed! Contact verified. Proceed to Step Six: Separate boxes by at least 20px.",
                            tts="Collision confirmed and verified. Proceed to step six: separate the boxes.",
                            t=current_time,
                        )
                else:
                    if (current_time - self.bcx1_last_collision_time) > 0.8:
                        self.bcx1_collision_hold_duration = max(0.0, self.bcx1_collision_hold_duration - (dt * 0.4))

            # S06: Separate Boxes & Confirm
            elif active_step.id == "S06":
                is_separated = (not is_colliding) and (dist_px >= 20.0 or (red_box is not None and yellow_box is not None and dist_px > 18.0 and not boxes_overlap))

                if is_separated:
                    self.bcx1_last_separation_time = current_time
                    self.bcx1_s06_hold_duration += dt
                    if self.bcx1_s06_hold_duration >= 0.7:
                        self.protocol_complete = True
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                        self._trigger_alert(
                            step_id="S06",
                            severity="info",
                            kind="nominal_completion",
                            message="BCX-1 Protocol Complete! Two-box collision dynamics and separation verified nominal.",
                            tts="Two-box collision experiment complete. All six steps verified nominal.",
                            t=current_time,
                        )
                else:
                    # Provide grace period against brief hand occlusion while pulling boxes apart
                    if (current_time - self.bcx1_last_separation_time) < 0.6 and not is_colliding:
                        pass
                    else:
                        self.bcx1_s06_hold_duration = max(0.0, self.bcx1_s06_hold_duration - (dt * 0.3))

        # 5. Draw High-Tech Mission Control Visual HUD for BCX-1
        self.overlay_objects = [o for o in (
            {"label": "container", "cls": "container", "box": list(outer_box), "conf": 1.0,
             "state": ["LOCKED"], "color": "#60a5fa"} if (outer_box is not None and self.bcx1_container_locked) else None,
            {"label": "red box", "cls": "red_box", "box": list(red_box), "conf": 0.9,
             "state": [n for n, f in (("INSIDE", red_inside), ("COLLISION", is_colliding)) if f], "color": "#ef4444"} if red_box is not None else None,
            {"label": "yellow box", "cls": "yellow_box", "box": list(yellow_box), "conf": 0.9,
             "state": [n for n, f in (("INSIDE", yellow_inside), ("COLLISION", is_colliding)) if f], "color": "#facc15"} if yellow_box is not None else None,
        ) if o is not None]
        annotated = frame.copy() if self.render_hud else frame
        if self.render_hud: self._render_bcx1_hud(
            annotated,
            outer_box=outer_box,
            red_box=red_box,
            yellow_box=yellow_box,
            red_inside=red_inside,
            yellow_inside=yellow_inside,
            is_colliding=is_colliding,
            dist_px=dist_px,
            active_step=active_step,
            current_time=current_time,
            skeleton_data=skeleton_data,
        )

        # 6. Format Telemetry
        completed_count = sum(1 for s in self.steps if s.status == "completed")
        compliance_pct = int((completed_count / len(self.steps)) * 100) if self.steps else 100

        step_telemetry = []
        for s in self.steps:
            is_verifying = False
            pct = 0
            if s.status == "completed":
                pct = 100
            elif s.status == "active":
                if s.id == "S01":
                    is_verifying = self.bcx1_s01_hold_duration > 0
                    pct = min(100, int((self.bcx1_s01_hold_duration / 0.8) * 100))
                elif s.id == "S02":
                    is_verifying = (red_box is not None) and (yellow_box is not None)
                    pct = min(100, int((self.bcx1_s02_hold_duration / 0.8) * 100))
                elif s.id == "S03":
                    is_verifying = red_inside
                    pct = min(100, int((self.bcx1_s03_hold_duration / 0.8) * 100))
                elif s.id == "S04":
                    is_verifying = yellow_inside and dist_px > 25.0
                    pct = min(100, int((self.bcx1_s04_hold_duration / 0.8) * 100))
                elif s.id == "S05":
                    is_verifying = is_colliding
                    pct = min(100, int((self.bcx1_collision_hold_duration / 0.8) * 100))
                elif s.id == "S06":
                    is_verifying = (not is_colliding) and (dist_px >= 20.0)
                    pct = min(100, int((self.bcx1_s06_hold_duration / 0.7) * 100))

            step_telemetry.append({
                "id": s.id,
                "name": s.name,
                "prompt": s.prompt,
                "status": s.status,
                "elapsed_s": round(s.elapsed_s, 1),
                "is_verifying": is_verifying,
                "verification_pct": pct,
            })

        # Alert expiry: only emit recent_alert if within 4.0 seconds of generation
        recent_alert_payload = None
        if self.recent_alerts and (current_time - self.recent_alerts[-1].timestamp) < 4.0:
            a = self.recent_alerts[-1]
            recent_alert_payload = {
                "step_id": a.step_id,
                "severity": a.severity,
                "kind": a.kind,
                "message": a.message,
                "tts": a.spoken_tts,
                "timestamp": a.timestamp,
            }

        telemetry = {
            "experiment_id": self.experiment_id,
            "experiment_title": self.title,
            "rack_id": self.rack_id,
            "frame_idx": self.frame_count,
            "active_step_id": active_step.id if active_step else "DONE",
            "active_step_name": active_step.name if active_step else "Protocol Completed",
            "prompt": active_step.prompt if active_step else "Two-box collision experiment completed nominally.",
            "compliance_score": compliance_pct,
            "is_complete": self.protocol_complete,
            "drinking": {  # Fallback field so frontend doesn't crash on null
                "in_progress": is_colliding,
                "hold_duration_s": round(self.bcx1_collision_hold_duration, 2),
                "target_duration_s": 0.8,
                "progress_pct": min(100, int((self.bcx1_collision_hold_duration / 0.8) * 100)),
                "water_consumed": self.protocol_complete,
            },
            "settle": {
                "in_progress": (not is_colliding) and (dist_px >= 20.0),
                "is_returned": red_inside and yellow_inside,
                "hands_released": True,
                "hold_duration_s": round(self.bcx1_s06_hold_duration, 2),
                "target_duration_s": 0.7,
                "progress_pct": min(100, int((self.bcx1_s06_hold_duration / 0.7) * 100)),
                "is_complete": self.protocol_complete,
            },
            "collision": {
                "in_progress": is_colliding,
                "is_colliding": is_colliding,
                "distance_px": round(dist_px, 1) if dist_px < 900 else None,
                "red_inside": red_inside,
                "yellow_inside": yellow_inside,
                "is_separated": (not is_colliding) and (dist_px >= 20.0),
                "hold_duration_s": round(self.bcx1_collision_hold_duration, 2),
                "target_duration_s": 0.8,
                "progress_pct": min(100, int((self.bcx1_collision_hold_duration / 0.8) * 100)),
                "separation_hold_s": round(self.bcx1_s06_hold_duration, 2),
                "separation_target_s": 0.7,
                "separation_pct": min(100, int((self.bcx1_s06_hold_duration / 0.7) * 100)),
            },
            "steps": step_telemetry,
            "geometry": {
                "target_detected": red_box is not None or yellow_box is not None,
                "target_confidence": 0.95 if (red_box or yellow_box) else 0.0,
                "person_detected": skeleton_data["detected"],
                "full_body_detected": skeleton_data["detected"] and skeleton_data["body_points_count"] >= 10,
                "body_points_count": skeleton_data["body_points_count"],
                "body_points_total": 17,
                "posture_status": skeleton_data["posture_status"],
                "posture_stability": skeleton_data["posture_stability"],
                "torso_angle_deg": skeleton_data["torso_angle_deg"],
                "body_velocity_px_s": skeleton_data["velocity_px_s"],
                "body_zones": skeleton_data["zones"],
                "arm_angles": skeleton_data["arm_angles"],
                "anchored": skeleton_data["anchored"],
                "skeleton": skeleton_data["keypoints"],
                "hand_contact": False,
                "closest_wrist_px": None,
                "is_lifted": False,
                "lift_delta_px": 0.0,
                "near_mouth": False,
                "is_drinking_pose": is_colliding,
                "in_table_zone": True,
                "dist_to_mouth_px": None,
                "outer_box_detected": outer_box is not None,
                "container_locked": self.bcx1_container_locked,
                "red_box_detected": red_box is not None,
                "yellow_box_detected": yellow_box is not None,
                "red_inside": red_inside,
                "yellow_inside": yellow_inside,
                "distance_px": round(dist_px, 1) if dist_px < 900 else None,
                "is_colliding": is_colliding,
            },
            "recent_alert": recent_alert_payload,
            "alert_count": len(self.alerts),
        }

        return annotated, telemetry

    def _render_bcx1_hud(
        self,
        img: np.ndarray,
        outer_box: tuple[float, float, float, float] | None,
        red_box: tuple[float, float, float, float] | None,
        yellow_box: tuple[float, float, float, float] | None,
        red_inside: bool,
        yellow_inside: bool,
        is_colliding: bool,
        dist_px: float,
        active_step: StepState | None,
        current_time: float,
        skeleton_data: dict[str, Any] | None = None,
    ) -> None:
        """Renders Mission Control Visual HUD for BCX-1 Two-Box Collision."""
        h, w = img.shape[:2]

        # 0. Render Full-Body Skeletal Rig & Biometric Telemetry Card
        if skeleton_data is not None:
            self._render_skeletal_rig(img, skeleton_data)

        overlay = img.copy()

        # Top Header Bar
        cv2.rectangle(overlay, (0, 0), (w, 52), (10, 15, 22), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, 52), (w, 52), (40, 180, 240), 1)

        cv2.putText(img, "PARIKSHAK MISSION CONTROL", (18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(img, "EXP: BCX-1 (RED & YELLOW BOX COLLISION)", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 200, 255), 2, cv2.LINE_AA)

        # Status badge top right
        status_text = "● VERIFYING COLLISION" if is_colliding else ("● SEPARATED" if dist_px >= 20.0 and dist_px < 900 else "● TRACKING ACTIVE")
        status_color = (40, 140, 255) if is_colliding else ((60, 230, 100) if dist_px >= 20.0 and dist_px < 900 else (100, 200, 255))
        cv2.putText(img, status_text, (w - 230, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2, cv2.LINE_AA)

        # 1. Draw Outer Container Box
        if outer_box is not None:
            ox1, oy1, ox2, oy2 = [int(v) for v in outer_box]
            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (255, 200, 0), 2)
            c_len = 20
            for cx, cy in [(ox1, oy1), (ox2, oy1), (ox1, oy2), (ox2, oy2)]:
                dx = c_len if cx == ox1 else -c_len
                dy = c_len if cy == oy1 else -c_len
                cv2.line(img, (cx, cy), (cx + dx, cy), (255, 230, 80), 3)
                cv2.line(img, (cx, cy), (cx, cy + dy), (255, 230, 80), 3)

            lock_status = "LOCKED" if self.bcx1_container_locked else "SEARCHING"
            cv2.rectangle(img, (ox1, oy1 - 22), (ox1 + 225, oy1), (255, 200, 0), -1)
            cv2.putText(img, f"CONTAINER BOX [{lock_status}]", (ox1 + 6, oy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 15, 25), 1, cv2.LINE_AA)

        # 2. Draw Red Box
        rcx, rcy = None, None
        if red_box is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in red_box]
            rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.circle(img, (rcx, rcy), 5, (0, 0, 255), -1)
            badge_lbl = "RED BOX: IN CONTAINER" if red_inside else "RED BOX: OUTSIDE"
            cv2.rectangle(img, (rx1, ry1 - 20), (rx1 + 160, ry1), (0, 0, 200), -1)
            cv2.putText(img, badge_lbl, (rx1 + 5, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Yellow Box
        ycx, ycy = None, None
        if yellow_box is not None:
            yx1, yy1, yx2, yy2 = [int(v) for v in yellow_box]
            ycx, ycy = (yx1 + yx2) // 2, (yy1 + yy2) // 2
            cv2.rectangle(img, (yx1, yy1), (yx2, yy2), (0, 230, 255), 2)
            cv2.circle(img, (ycx, ycy), 5, (0, 230, 255), -1)
            badge_lbl = "YELLOW BOX: IN CONTAINER" if yellow_inside else "YELLOW BOX: OUTSIDE"
            cv2.rectangle(img, (yx1, yy1 - 20), (yx1 + 180, yy1), (0, 200, 220), -1)
            cv2.putText(img, badge_lbl, (yx1 + 5, yy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 20, 30), 1, cv2.LINE_AA)

        # 4. Connecting Vector Line & Collision Dynamics
        if rcx is not None and ycx is not None:
            mid_x, mid_y = (rcx + ycx) // 2, (rcy + ycy) // 2

            if is_colliding:
                pulse_color = (40, 100, 255) if int(current_time * 6) % 2 == 0 else (0, 230, 255)
                cv2.line(img, (rcx, rcy), (ycx, ycy), pulse_color, 4)
                badge_w, badge_h = 240, 32
                cv2.rectangle(img, (mid_x - badge_w // 2, mid_y - badge_h // 2), (mid_x + badge_w // 2, mid_y + badge_h // 2), (20, 25, 35), -1)
                cv2.rectangle(img, (mid_x - badge_w // 2, mid_y - badge_h // 2), (mid_x + badge_w // 2, mid_y + badge_h // 2), pulse_color, 2)
                cv2.putText(img, "ACTIVE COLLISION DETECTED", (mid_x - 110, mid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                line_color = (60, 230, 100) if dist_px >= 20.0 else (240, 180, 40)
                cv2.line(img, (rcx, rcy), (ycx, ycy), line_color, 2, cv2.LINE_AA)
                dist_str = f"d: {int(dist_px)}px"
                cv2.rectangle(img, (mid_x - 38, mid_y - 12), (mid_x + 38, mid_y + 12), (20, 25, 35), -1)
                cv2.rectangle(img, (mid_x - 38, mid_y - 12), (mid_x + 38, mid_y + 12), line_color, 1)
                cv2.putText(img, dist_str, (mid_x - 30, mid_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 245, 250), 1, cv2.LINE_AA)

        # 5. Dynamic Progress Meters based on Active Step
        if active_step:
            if active_step.id == "S01":
                pct = min(1.0, self.bcx1_s01_hold_duration / 0.8)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (255, 200, 0), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (255, 230, 80), 1)
                cv2.putText(img, f"IDENTIFY CONTAINER: {int(pct * 100)}%", (w // 2 - 110, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S02":
                pct = min(1.0, self.bcx1_s02_hold_duration / 0.8)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (40, 220, 120), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 250, 160), 1)
                cv2.putText(img, f"VERIFY COLORS: {int(pct * 100)}% (RED & YELLOW)", (w // 2 - 130, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S03":
                pct = min(1.0, self.bcx1_s03_hold_duration / 0.8)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (0, 0, 255), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (100, 100, 255), 1)
                cv2.putText(img, f"PLACE RED BOX: {int(pct * 100)}%", (w // 2 - 105, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S04":
                pct = min(1.0, self.bcx1_s04_hold_duration / 0.8)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + int(280 * pct), 88), (0, 220, 240), -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 240, 255), 1)
                cv2.putText(img, f"PLACE YELLOW BOX: {int(pct * 100)}%", (w // 2 - 115, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 15, 25), 1, cv2.LINE_AA)

            elif active_step.id == "S05":
                pct = min(1.0, self.bcx1_collision_hold_duration / 0.8)
                bar_w = int(280 * pct)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                fill_color = (60, 230, 100) if is_colliding else (40, 140, 255)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + bar_w, 88), fill_color, -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 180, 250), 1)
                col_lbl = f"COLLISION HOLD: {int(pct * 100)}% ({self.bcx1_collision_hold_duration:.1f}s / 0.8s)" if is_colliding else "BRING RED & YELLOW BOXES TOGETHER"
                cv2.putText(img, col_lbl, (w // 2 - 130, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (255, 255, 255), 1, cv2.LINE_AA)

            elif active_step.id == "S06":
                pct = min(1.0, self.bcx1_s06_hold_duration / 0.7)
                bar_w = int(280 * pct)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 25, 35), -1)
                is_sep = (not is_colliding) and (dist_px >= 20.0)
                fill_color = (60, 230, 100) if is_sep else (255, 140, 40)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 - 140 + bar_w, 88), fill_color, -1)
                cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (80, 240, 180), 1)
                sep_lbl = f"SEPARATE HOLD: {int(pct * 100)}% (d: {int(dist_px)}px / 20px)" if is_sep else f"SEPARATE BOXES (d: {int(dist_px)}px / target: 20px)"
                cv2.putText(img, sep_lbl, (w // 2 - 135, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        elif self.protocol_complete:
            cv2.rectangle(img, (w // 2 - 150, 62), (w // 2 + 150, 88), (20, 35, 25), -1)
            cv2.rectangle(img, (w // 2 - 150, 62), (w // 2 + 150, 88), (60, 220, 100), -1)
            cv2.putText(img, "ALL 6 STEPS VERIFIED NOMINAL", (w // 2 - 130, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 6. Bottom Prompt & Alert Bar
        has_alert = bool(self.recent_alerts and (current_time - self.recent_alerts[-1].timestamp) < 4.0)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, h - 60), (w, h - 60), (45, 91, 216), 1)

        if has_alert:
            alert = self.recent_alerts[-1]
            cv2.putText(img, f"ALERT [{alert.kind.upper()}]: {alert.message}", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 255), 2, cv2.LINE_AA)
        elif active_step:
            step_prompt = f"[{active_step.id}] {active_step.prompt}"
            cv2.putText(img, step_prompt, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "All procedure steps verified successfully.", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 120), 1, cv2.LINE_AA)




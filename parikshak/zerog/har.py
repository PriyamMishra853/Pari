"""Objects in 3D, the body-payload interaction graph, and temporal HAR.

Object depth: a detected box has no depth, so it is placed with the pinhole
model and a per-class physical size prior (Z = f * size / pixels). When a hand
is touching the object, the hand's measured depth is used instead - it comes
from the body PnP and is the better estimate.

HAR v1 is rule-based and temporal: per-frame evidence (hand speed, hand-object
distance, contact, object motion, object-to-mouth distance) is scored for every
activity, and the label is chosen by a decaying vote over the last frames with a
minimum dwell, so a single noisy frame cannot flip it. It is not a learned
model; the dataset recorder exists to collect training data for one (ST-GCN).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from parikshak.zerog.config import CameraModel

SIZE_M = {
    "bottle": 0.22, "cup": 0.10, "wine glass": 0.17, "bowl": 0.15, "vase": 0.22,
    "cell phone": 0.15, "remote": 0.18, "laptop": 0.34, "keyboard": 0.44, "mouse": 0.11,
    "book": 0.24, "scissors": 0.18, "spoon": 0.16, "potted plant": 0.28, "chair": 0.90,
    "red_box": 0.08, "yellow_box": 0.08, "container": 0.32, "sports ball": 0.20,
    "apple": 0.08, "orange": 0.08, "banana": 0.19, "clock": 0.25, "toothbrush": 0.18,
    "teddy bear": 0.30, "backpack": 0.45, "handbag": 0.35, "person": 1.0,
}
DEFAULT_SIZE = 0.20

ACTIVITIES = [
    "IDLE", "APPROACH_PAYLOAD", "REACH_TOOL", "GRASP_TOOL", "MOVE_TOOL", "PLACE_TOOL",
    "RELEASE_TOOL", "INSPECT_PAYLOAD", "DRINK", "RETURN_POSITION", "UNEXPECTED_INTERACTION",
]


def contact_probability(d_m: float) -> float:
    """Logistic in distance: ~0.5 at 7 cm, ~0.95 at 2 cm, ~0.05 at 12 cm."""
    return float(1.0 / (1.0 + math.exp((d_m - 0.07) / 0.017)))


def object_cam_position(box_raw, cls: str, cam: CameraModel, hands_px: dict[str, tuple[np.ndarray, np.ndarray]] | None):
    """3D centre of a detected object in camera coordinates (m) and how it was placed."""
    x1, y1, x2, y2 = box_raw
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ext = max(x2 - x1, y2 - y1, 4.0)
    z = cam.fx * SIZE_M.get(cls, DEFAULT_SIZE) / ext
    how = "size-prior"
    if hands_px:
        pad = 0.25 * ext
        for _, (hpx, hcam) in hands_px.items():
            if x1 - pad <= hpx[0] <= x2 + pad and y1 - pad <= hpx[1] <= y2 + pad and hcam[2] > 0.1:
                z, how = float(hcam[2]), "hand-depth"
                break
    z = float(min(max(z, 0.15), 6.0))
    return np.array([(u - cam.cx) * z / cam.fx, (v - cam.cy) * z / cam.fy, z]), how


@dataclass
class _Track:
    pos: np.ndarray
    t: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))


class InteractionGraph:
    """Nodes: head, torso, hands, feet, objects. Edges: hand-object distance,
    relative position, contact probability, closing velocity."""

    def __init__(self) -> None:
        self._tracks: dict[str, _Track] = {}

    def _velocity(self, key: str, pos: np.ndarray, t: float) -> np.ndarray:
        tr = self._tracks.get(key)
        if tr is None or t - tr.t > 1.0:
            self._tracks[key] = _Track(pos.copy(), t)
            return np.zeros(3)
        dt = max(1e-3, t - tr.t)
        v = (pos - tr.pos) / dt
        tr.vel = 0.6 * tr.vel + 0.4 * v
        tr.pos, tr.t = pos.copy(), t
        return tr.vel

    def build(self, body_nodes: dict[str, np.ndarray], objects: list[dict[str, Any]], t: float) -> dict[str, Any]:
        nodes = []
        for name, p in body_nodes.items():
            vel = self._velocity(name, p, t)
            nodes.append({"id": name, "type": "body", "pos": _r(p), "speed_mps": round(float(np.linalg.norm(vel)), 3)})
        for o in objects:
            vel = self._velocity("obj:" + o["id"], o["pos"], t)
            o["speed_mps"] = float(np.linalg.norm(vel))
            nodes.append({"id": o["id"], "type": "object", "cls": o["cls"], "pos": _r(o["pos"]),
                          "speed_mps": round(o["speed_mps"], 3), "placement": o["placement"]})
        edges = []
        for hand in ("left_hand", "right_hand"):
            if hand not in body_nodes:
                continue
            hp = body_nodes[hand]
            hv = self._tracks.get(hand).vel if hand in self._tracks else np.zeros(3)
            for o in objects:
                rel = o["pos"] - hp
                d = float(np.linalg.norm(rel))
                closing = float(-np.dot(hv - self._tracks.get("obj:" + o["id"], _Track(o["pos"], t)).vel, rel) / max(d, 1e-6))
                edges.append({"from": hand, "to": o["id"], "cls": o["cls"], "distance_m": round(d, 3),
                              "relative": _r(rel), "contact_prob": round(contact_probability(d), 3),
                              "closing_mps": round(closing, 3)})
        if "head" in body_nodes:
            m = body_nodes["mouth" if "mouth" in body_nodes else "head"]
            for o in objects:
                d = float(np.linalg.norm(o["pos"] - m))
                if o.get("top") is not None:
                    d = min(d, float(np.linalg.norm(np.asarray(o["top"]) - m)))
                edges.append({"from": "mouth", "to": o["id"], "cls": o["cls"], "distance_m": round(d, 3),
                              "relative": _r(o["pos"] - body_nodes["head"]), "contact_prob": round(contact_probability(d), 3),
                              "closing_mps": 0.0})
        return {"nodes": nodes, "edges": edges}


def _r(v) -> list[float]:
    return [round(float(x), 3) for x in np.asarray(v).ravel()]


class TemporalHAR:
    """Decaying-vote temporal activity recogniser with a minimum dwell time."""

    def __init__(self, window: int = 6, min_dwell_s: float = 0.4) -> None:
        self.window = window
        self.min_dwell_s = min_dwell_s
        self.history: deque[dict[str, float]] = deque(maxlen=window)
        self.label = "IDLE"
        self.since = 0.0
        self.prev_contact: dict[str, float] = {}
        self.prev_hand_speed = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.label, self.since = "IDLE", 0.0
        self.prev_contact.clear()

    def update(self, t: float, person: bool, graph: dict[str, Any], expected: list[str] | None = None) -> dict[str, Any]:
        scores = {a: 0.0 for a in ACTIVITIES}
        evidence: dict[str, Any] = {}
        if not person:
            scores["IDLE"] = 1.0
        else:
            hand_edges = [e for e in graph["edges"] if e["from"] in ("left_hand", "right_hand")]
            mouth_edges = [e for e in graph["edges"] if e["from"] == "mouth"]
            speeds = {n["id"]: n["speed_mps"] for n in graph["nodes"]}
            hand_speed = max(speeds.get("left_hand", 0.0), speeds.get("right_hand", 0.0))
            best = min(hand_edges, key=lambda e: e["distance_m"]) if hand_edges else None
            evidence["hand_speed_mps"] = round(hand_speed, 3)
            if best is None:
                scores["IDLE"] = 0.6 if hand_speed < 0.15 else 0.2
                scores["RETURN_POSITION"] = 0.4 if hand_speed >= 0.15 else 0.0
            else:
                d, c = best["distance_m"], best["contact_prob"]
                obj_speed = speeds.get(best["to"], 0.0)
                prev_c = self.prev_contact.get(best["to"], 0.0)
                evidence.update({"nearest_object": best["cls"], "hand_object_m": d, "contact": c,
                                 "object_speed_mps": round(obj_speed, 3)})
                unexpected = expected is not None and len(expected) > 0 and best["cls"] not in expected
                mouth = next((e for e in mouth_edges if e["to"] == best["to"]), None)
                if mouth is not None:
                    evidence["object_mouth_m"] = mouth["distance_m"]
                if c > 0.5:
                    if unexpected:
                        scores["UNEXPECTED_INTERACTION"] = 0.9
                    if mouth is not None and mouth["distance_m"] < 0.15:
                        scores["DRINK"] = 1.0
                    elif mouth is not None and mouth["distance_m"] < 0.35 and obj_speed < 0.12:
                        scores["INSPECT_PAYLOAD"] = 0.8
                    if obj_speed > 0.10:
                        scores["MOVE_TOOL"] = 0.7 + min(0.3, obj_speed)
                    elif obj_speed > 0.03 and self.label in ("MOVE_TOOL", "PLACE_TOOL"):
                        scores["PLACE_TOOL"] = 0.8  # decelerating while still held
                    else:
                        scores["GRASP_TOOL"] = 0.75 if prev_c < 0.5 or self.label in ("REACH_TOOL", "GRASP_TOOL") else 0.55
                        scores["PLACE_TOOL"] = max(scores["PLACE_TOOL"], 0.5 if self.label in ("MOVE_TOOL", "PLACE_TOOL") else 0.0)
                else:
                    if prev_c > 0.5:
                        scores["RELEASE_TOOL"] = 0.9
                    if d < 0.35 and best["closing_mps"] > 0.05:
                        scores["REACH_TOOL"] = 0.8
                    elif d < 0.6 and best["closing_mps"] > 0.05:
                        scores["APPROACH_PAYLOAD"] = 0.7
                    elif best["closing_mps"] < -0.08 and hand_speed > 0.1:
                        scores["RETURN_POSITION"] = 0.7
                    else:
                        scores["IDLE"] = 0.6 if hand_speed < 0.15 else 0.3
                self.prev_contact[best["to"]] = c
            self.prev_hand_speed = hand_speed

        self.history.append(scores)
        # decaying vote: newest frame weighs most
        agg = {a: 0.0 for a in ACTIVITIES}
        n = len(self.history)
        for i, s in enumerate(self.history):
            w = 0.6 ** (n - 1 - i)
            for a, v in s.items():
                agg[a] += w * v
        total = sum(0.6 ** k for k in range(n))
        cand = max(agg, key=agg.get)
        conf = agg[cand] / total if total else 0.0
        if cand != self.label and (t - self.since >= self.min_dwell_s or agg[cand] > 1.6 * agg.get(self.label, 0.0)):
            self.label, self.since = cand, t
        label_conf = agg.get(self.label, 0.0) / total if total else 0.0
        return {
            "activity": self.label,
            "confidence": round(float(min(1.0, label_conf)), 2),
            "candidate": cand,
            "candidate_confidence": round(float(min(1.0, conf)), 2),
            "status": "UNCERTAIN" if label_conf < 0.45 else "CONFIDENT",
            "evidence": evidence,
            "model": "rule-based temporal HAR v1 (decaying vote, 6-frame window)",
        }

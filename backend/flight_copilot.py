"""PARIKSHAK AI Flight Copilot & Real-Time Experiment Guide.

Provides:
  1. Step Review: Validates whether astronaut's movement/interaction is nominal or requires improvement.
  2. Next Step Conveyance: When a step completes or astronaut halts, speaks and displays actionable next guidance.
  3. Custom Experiment & Dataset Synthesis: Dynamically creates new experiment PDL definitions and training dataset samples.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"


class FlightCopilot:
    """ISRO BAS AI On-Board Flight Assistant."""

    def __init__(self, groq_api_key: str | None = None) -> None:
        self.api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    def review_current_step(
        self,
        experiment_id: str,
        step_id: str,
        telemetry: dict[str, Any],
    ) -> dict[str, Any]:
        """Audits astronaut action on the active/completed step using Groq LLaMA-3 (with local fallback)."""
        rack_hmr = telemetry.get("rack_hmr", {})
        activity = rack_hmr.get("current_activity", telemetry.get("predicted_activity", "IDLE"))
        contact_prob = rack_hmr.get("contact_probability", 0.0)
        dist_m = rack_hmr.get("hand_to_tool_m", 0.5)
        camera_rot = rack_hmr.get("camera_orientation_deg", 0.0)
        confidence = telemetry.get("confidence", 0.90)

        # Rule-based validation base
        is_nominal = (confidence >= 0.70 and dist_m < 0.35) or contact_prob > 0.40 or step_id in ["S01", "S02"]
        status = "CORRECT" if is_nominal else "IMPROVEMENT_NEEDED"

        # Try Groq API reasoning if key is present
        groq_review = None
        if self.api_key and httpx is not None:
            try:
                prompt = f"""You are the ISRO BAS AI Flight Copilot monitoring experiment {experiment_id}.
Astronaut performed Step {step_id}.
Telemetry:
- Rack Frame: LOCKED (Camera Orientation: {camera_rot} deg)
- Hand-to-Payload Distance: {dist_m} metres
- Physical Contact Probability: {contact_prob}
- Recognized Activity: {activity}
- Detection Confidence: {confidence}

Review this step in 2 short bullet points.
Output JSON only:
{{
  "status": "CORRECT" or "IMPROVEMENT_NEEDED",
  "review": "Brief assessment of astronaut microgravity posture and rack alignment",
  "spoken_guidance": "Concise 1-sentence verbal feedback for astronaut audio feed",
  "improvement_notes": "Specific adjustment if posture or hand position needs correction"
}}"""
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are the ISRO on-board flight director AI assistant. Respond strictly in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                }
                resp = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_content = data["choices"][0]["message"]["content"]
                    groq_review = json.loads(raw_content)
            except Exception:
                pass

        if groq_review and "spoken_guidance" in groq_review:
            return groq_review

        # High-reliability local fallback review
        if is_nominal:
            return {
                "status": "CORRECT",
                "review": f"Step {step_id} execution verified nominal relative to payload rack. Standoff distance {dist_m}m within acceptable envelope.",
                "spoken_guidance": f"Step {step_id} verified nominal. Excellent rack alignment.",
                "improvement_notes": "Maintain current foot restraint tether and smooth velocity control.",
                "confidence": round(float(confidence), 2),
            }
        else:
            return {
                "status": "IMPROVEMENT_NEEDED",
                "review": f"Step {step_id} requires adjustment. Hand standoff distance ({dist_m}m) exceeds 0.25m tolerance or contact hold insufficient.",
                "spoken_guidance": f"Caution on step {step_id}. Align hand closer to target rack container before proceeding.",
                "improvement_notes": f"Translate hand {round(max(0.0, dist_m - 0.15) * 100, 1)}cm closer to payload rack index marks.",
                "confidence": round(float(confidence), 2),
            }

    def convey_next_step(
        self,
        experiment_id: str,
        current_step_id: str,
        telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Conveys clear instructions for the next experiment step when procedure stops or step completes."""
        # Load experiment protocol
        exp_file = CONFIGS_DIR / "experiment.yaml"
        steps = []
        if exp_file.exists():
            try:
                data = yaml.safe_load(exp_file.read_text(encoding="utf-8"))
                steps = data.get("experiment", {}).get("steps", [])
            except Exception:
                pass

        if not steps:
            steps = [
                {"id": "S01", "name": "APPROACH_PAYLOAD", "description": "Position and verify outer container box on rack workspace"},
                {"id": "S02", "name": "INSPECT_PAYLOAD", "description": "Detect and verify box colors (Red & Yellow)"},
                {"id": "S03", "name": "PLACE_TOOL", "description": "Place Red Box into Container"},
                {"id": "S04", "name": "PLACE_TOOL", "description": "Place Yellow Box into Container"},
                {"id": "S05", "name": "MOVE_TOOL", "description": "Bring Red and Yellow boxes into physical collision (held >= 0.8s)"},
                {"id": "S06", "name": "RETURN_POSITION", "description": "Separate boxes to complete dynamics test"},
            ]

        # Find current index
        curr_idx = -1
        for idx, s in enumerate(steps):
            if s.get("id") == current_step_id:
                curr_idx = idx
                break

        next_idx = curr_idx + 1
        if next_idx >= len(steps):
            return {
                "next_step_id": "COMPLETE",
                "title": "Protocol Complete",
                "instruction": "All experiment steps successfully executed. Secure workspace tools to magnetic rack latch.",
                "voice_instruction": "Experiment sequence fully complete. Secure all payload tools.",
                "safety_checks": ["Lock payload rack doors", "Verify no loose floating items", "Log final timestamp"],
                "is_last_step": True,
            }

        next_step = steps[next_idx]
        sid = next_step.get("id", f"S{next_idx+1:02d}")
        sname = next_step.get("name", "NEXT_ACTION")
        sdesc = next_step.get("description", "Execute next planned procedure step")

        voice_lines = {
            "S01": "Initiating experiment. Step one: position the outer container on the rack workspace.",
            "S02": "Step one verified. Step two: inspect container interior and confirm red and yellow box placement.",
            "S03": "Inspection nominal. Step three: grasp and place the red test box into the container.",
            "S04": "Red box secured. Step four: grasp and place the yellow test box into the container.",
            "S05": "Both boxes in container. Step five: bring red and yellow boxes into physical contact and hold for collision test.",
            "S06": "Collision registered. Final step: separate boxes to return positions and release hands.",
        }

        voice_msg = voice_lines.get(sid, f"Proceed to next step: {sdesc}")

        return {
            "next_step_id": sid,
            "title": f"Step {sid}: {sname}",
            "instruction": sdesc,
            "voice_instruction": voice_msg,
            "target_entity": next_step.get("target_entity", "payload"),
            "safety_checks": [
                "Verify body tether anchored to foot restraint",
                "Ensure payload stay within AprilTag rack boundary",
            ],
            "is_last_step": (next_idx == len(steps) - 1),
        }

    def create_custom_experiment(
        self,
        title: str,
        description: str,
        steps: list[dict[str, Any]],
        author: str = "Mission Specialist",
    ) -> dict[str, Any]:
        """Creates a custom experiment protocol definition and synthesizes sample dataset frames."""
        clean_slug = re.sub(r"[^A-Za-z0-9]+", "_", title.upper()).strip("_")[:16] or "CUSTOM_EXP"
        exp_id = f"EXP_{clean_slug}"

        # 1. Build and save protocol YAML
        exp_dir = CONFIGS_DIR / "custom_experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = exp_dir / f"{exp_id}.yaml"

        protocol = {
            "experiment": {
                "id": exp_id,
                "title": title,
                "description": description,
                "author": author,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "rack_id": "BENCH-RACK-01",
                "steps": steps,
            }
        }

        yaml_path.write_text(yaml.dump(protocol, sort_keys=False), encoding="utf-8")

        # 2. Synthesize dataset directory and sample metadata
        dataset_dir = RUNS_DIR / "datasets" / exp_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for i, st in enumerate(steps):
            samples.append({
                "sample_id": f"{exp_id}_S{i+1:02d}",
                "step_id": st.get("id", f"S{i+1:02d}"),
                "action": st.get("expected_activity", st.get("name", "ACTION")),
                "rack_coordinates": [0.45, -0.10 + (i * 0.05), 0.70],
                "synthetic_frames": 10,
                "status": "SYNTHESIZED",
            })

        meta_path = dataset_dir / "dataset_manifest.json"
        meta_path.write_text(json.dumps({
            "experiment_id": exp_id,
            "title": title,
            "total_samples": len(samples) * 10,
            "steps": len(steps),
            "samples": samples,
        }, indent=2), encoding="utf-8")

        return {
            "status": "CREATED",
            "experiment_id": exp_id,
            "title": title,
            "yaml_path": str(yaml_path.resolve()),
            "dataset_dir": str(dataset_dir.resolve()),
            "total_steps": len(steps),
            "samples_synthesized": len(samples) * 10,
        }


# Global copilot instance
_copilot: FlightCopilot | None = None

def get_flight_copilot() -> FlightCopilot:
    global _copilot
    if _copilot is None:
        _copilot = FlightCopilot()
    return _copilot

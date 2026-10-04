"""Copilot reasoning: turns pipeline telemetry into a compact experiment state,
asks Groq for guidance, and falls back to deterministic local rules when Groq is
unavailable. Every message says which source produced it."""

from __future__ import annotations

import json
from typing import Any

from backend.app.groq import prompts
from backend.app.groq.client import GroqClient, get_client


def experiment_state(tel: dict[str, Any], spec: dict[str, Any] | None, event: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = tel.get("steps", [])
    active_id = tel.get("active_step_id")
    idx = next((i for i, s in enumerate(steps) if s.get("id") == active_id), None)
    spec_steps = {s.get("id"): s for s in (spec or {}).get("steps", [])}

    def step_info(i: int | None) -> dict[str, Any] | None:
        if i is None or i >= len(steps):
            return None
        s = steps[i]
        sp = spec_steps.get(s["id"], {})
        return {
            "id": s["id"], "name": s["name"], "instruction": s.get("prompt") or sp.get("prompt"),
            "index": i + 1, "total": len(steps), "status": s.get("status"),
            "progress_pct": s.get("verification_pct"), "elapsed_s": s.get("elapsed_s"),
            "hint": sp.get("hint"), "expected_activity": sp.get("expected_activity"),
        }

    z = tel.get("zerog", {})
    body, har, status = z.get("body", {}), z.get("har", {}), z.get("status", {})
    edges = [e for e in z.get("interaction_graph", {}).get("edges", []) if e.get("from") in ("left_hand", "right_hand")]
    edges = sorted(edges, key=lambda e: e["distance_m"])[:3]
    cur = step_info(idx)
    if cur is not None:
        cur["unmet_checks"] = tel.get("unmet", [])
        cur["met_checks"] = tel.get("met", [])
    return {
        "experiment": {"id": tel.get("experiment_id"), "title": tel.get("experiment_title")},
        "active_step": cur,
        "next_step": step_info(idx + 1 if idx is not None else None),
        "completed_steps": sum(1 for s in steps if s.get("status") == "completed"),
        "skipped_steps": [s["id"] for s in steps if s.get("status") == "skipped"],
        "procedure_complete": bool(tel.get("is_complete")),
        "activity": {"label": har.get("activity"), "confidence": har.get("confidence"), "status": har.get("status")},
        "astronaut": status.get("astronaut"),
        "rack_frame": status.get("rack_frame"),
        "body": {k: body.get(k) for k in ("frame", "inclination_deg", "posture", "head_direction") if k in body},
        "hand_object": [{"hand": e["from"], "object": e.get("cls"), "distance_m": e["distance_m"],
                         "contact_prob": e["contact_prob"]} for e in edges],
        "recent_alerts": [{"step": a.get("step_id"), "kind": a.get("kind"), "message": a.get("message")}
                          for a in (tel.get("alerts") or [])[-3:]],
        "event": event,
    }


class CopilotReasoner:
    def __init__(self, client: GroqClient | None = None) -> None:
        self.client = client or get_client()

    def _ask(self, template: str, state: dict[str, Any], interactive: bool = False, **kw: Any):
        """interactive=True (a button / voice request): one model, 4 s budget, then local rules."""
        user = template.format(state=json.dumps(state, separators=(",", ":"), default=str), **kw)
        if interactive:
            return self.client.chat_json(prompts.SYSTEM, user, timeout=4.0, allow_fallback=False)
        return self.client.chat_json(prompts.SYSTEM, user, timeout=8.0)

    # -------------------------------------------------------------- actions
    def guidance(self, state: dict[str, Any]) -> dict[str, Any]:
        idle = (state.get("event") or {}).get("idle_s", "several")
        data, meta = self._ask(prompts.GUIDANCE, state, idle_s=idle)
        if data and data.get("spoken"):
            return {**data, **meta}
        return {**self._local_guidance(state), **meta, "source": "local-rules"}

    def review(self, state: dict[str, Any]) -> dict[str, Any]:
        data, meta = self._ask(prompts.REVIEW, state, interactive=True)
        if data and data.get("spoken"):
            return {**data, **meta}
        return {**self._local_review(state), **meta, "source": "local-rules"}

    def next_step(self, state: dict[str, Any]) -> dict[str, Any]:
        data, meta = self._ask(prompts.NEXT, state, interactive=True)
        if data and data.get("spoken"):
            return {**data, **meta}
        return {**self._local_next(state), **meta, "source": "local-rules"}

    def explain_alert(self, state: dict[str, Any], alert: dict[str, Any]) -> dict[str, Any]:
        data, meta = self._ask(prompts.ALERT, state, alert=json.dumps(alert, default=str),
                               severity=alert.get("severity", "caution"))
        if data and data.get("spoken"):
            return {**data, **meta}
        return {"severity": alert.get("severity"), "display": alert.get("message", ""),
                "spoken": alert.get("tts") or alert.get("message", ""), **meta, "source": "local-rules"}

    def summary(self, state: dict[str, Any]) -> dict[str, Any]:
        data, meta = self._ask(prompts.SUMMARY, state)
        if data and data.get("spoken"):
            return {**data, **meta}
        n = state.get("completed_steps", 0)
        sk = state.get("skipped_steps", [])
        txt = f"Procedure ended with {n} steps verified" + (f" and steps {', '.join(sk)} skipped." if sk else ".")
        return {"display": txt, "spoken": txt, **meta, "source": "local-rules"}

    # ------------------------------------------------------- local fallbacks
    @staticmethod
    def _local_guidance(state: dict[str, Any]) -> dict[str, Any]:
        s = state.get("active_step") or {}
        unmet = s.get("unmet_checks") or []
        hint = s.get("hint") or s.get("instruction") or "Continue the procedure."
        if state.get("astronaut") != "DETECTED":
            spoken = "I cannot see you. Move back into the camera view."
        elif unmet:
            spoken = f"Step {s.get('index')}: {s.get('name')}. Still needed: {unmet[0]}. {hint}"
        else:
            spoken = f"Step {s.get('index')}: {hint}"
        return {"status": "STALLED", "display": spoken, "spoken": spoken[:220], "checks": unmet[:3]}

    @staticmethod
    def _local_review(state: dict[str, Any]) -> dict[str, Any]:
        s = state.get("active_step") or {}
        unmet = s.get("unmet_checks") or []
        if state.get("astronaut") != "DETECTED":
            return {"status": "IMPROVEMENT_NEEDED", "display": "Crew member not visible - cannot verify this step.",
                    "spoken": "I cannot verify this step. Please move into the camera view.", "improvement": "Move into view."}
        if not unmet or (s.get("progress_pct") or 0) >= 50:
            msg = f"Step {s.get('index')} looks correct so far. Hold it until it verifies."
            return {"status": "CORRECT", "display": msg, "spoken": msg, "improvement": ""}
        msg = f"Step {s.get('index')} needs correction: {unmet[0]} is not observed yet. {s.get('hint') or ''}".strip()
        return {"status": "IMPROVEMENT_NEEDED", "display": msg, "spoken": msg[:220], "improvement": s.get("hint") or unmet[0]}

    @staticmethod
    def _local_next(state: dict[str, Any]) -> dict[str, Any]:
        s = state.get("active_step")
        if not s:
            msg = "The procedure is complete. Secure all payload items."
            return {"display": msg, "spoken": msg, "safety": ["Secure loose items"]}
        msg = f"Step {s['index']} of {s['total']}: {s.get('instruction') or s['name']}"
        return {"display": msg, "spoken": msg, "safety": []}

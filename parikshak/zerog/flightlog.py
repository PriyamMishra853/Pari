"""Per-session hash-chained flight log (wraps parikshak.engine.logger.RunLogger)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from parikshak.zerog.paths import LOGS as LOG_DIR


class FlightLog:
    """Hash-chained run log (parikshak/engine/logger.py) for one experiment session:
    steps, alerts, stalls and every copilot message, with a text mirror. The file
    is only written once something happened, so browsing experiments leaves no
    empty logs behind."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.log = None
        self.t0 = time.time()
        self.dirty = False
        self.step_id = ""

    def start(self, exp_id: str, spec: dict[str, Any] | None) -> None:
        import hashlib

        from parikshak.engine.logger import RunLogger

        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"RUN_{exp_id}_{ts}"
        src = (spec or {}).get("_path")
        raw = Path(src).read_bytes() if src and Path(src).exists() else exp_id.encode()
        with self.lock:
            self.t0 = time.time()
            self.log = RunLogger(
                path=LOG_DIR / f"{run_id}.log.jsonl", procedure_id=exp_id,
                procedure_sha256=hashlib.sha256(raw).hexdigest(), run_id=run_id, belief_schema="zerog-1",
                operating_point="rack-centric HAR v1 (rule-based, temporal vote)",
                model_versions={"pose": "BlazePose GHUM lite (MediaPipe 1.0.1)", "detector": "YOLOv8n ONNX 320",
                                "rack": "AprilTag 36h11 + SQPnP", "copilot": "Groq (see copilot records)"},
            )
            self.dirty = False

    def add(self, kind: str, payload: dict[str, Any]) -> None:
        with self.lock:
            if self.log is not None:
                self.log.append(kind, payload, time.time() - self.t0)
                self.dirty = True

    def flush(self) -> None:
        with self.lock:
            if self.log is not None and self.dirty and len(self.log.records) > 1:
                try:
                    self.log.write()
                except OSError:
                    pass
                self.dirty = False

    @property
    def path(self) -> Path | None:
        return self.log.path if self.log is not None else None


def list_logs(log_dir: Path = LOG_DIR) -> list[dict[str, Any]]:
    from parikshak.engine.logger import read_log, verify_chain

    out = []
    for f in sorted(log_dir.glob("*.log.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        try:
            recs = read_log(f)
            ok, msg = verify_chain(recs)
        except Exception as exc:
            recs, ok, msg = [], False, f"unreadable: {exc}"
        out.append({"name": f.name, "records": len(recs), "verified": ok, "message": msg,
                    "size_kb": round(f.stat().st_size / 1024, 1), "mtime": f.stat().st_mtime,
                    "text_name": f.with_suffix(".txt").name if f.with_suffix(".txt").exists() else None})
    return out

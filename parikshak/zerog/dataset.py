"""Dataset recorder: every processed frame becomes a labelled training sample.

Layout (data/datasets/<EXPERIMENT>/<session>/):
    manifest.json        experiment, steps, label set, counts
    samples.jsonl        one record per frame: step label, activity label, 33 joints
                         (rack frame if locked, else camera frame), objects, rack status
    frames/000123.jpg    every Nth upright view frame (for re-labelling / image models)

Labels come from the procedure engine (which step is active, which completed)
and the HAR output - this is the data an ST-GCN would be trained on.
"""

from __future__ import annotations

import json
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    import importlib

    cv2 = importlib.import_module("cv2")
except ImportError:  # vision stack not installed: the engine still imports
    cv2 = None  # type: ignore
import numpy as np

from parikshak.zerog.config import ROOT

from parikshak.zerog.paths import DATASETS as DATA_DIR  # noqa: E402


class DatasetRecorder:
    def __init__(self) -> None:
        self.active = False
        self.dir: Path | None = None
        self.n = 0
        self.n_images = 0
        self.every = 3
        self.experiment_id = ""
        self._fh = None
        self.started = 0.0

    def start(self, experiment_id: str, spec: dict[str, Any] | None, every: int = 3) -> dict[str, Any]:
        self.stop()
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = DATA_DIR / experiment_id / ts
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.experiment_id, self.every, self.n, self.n_images = experiment_id, max(1, int(every)), 0, 0
        self.started = time.time()
        manifest = {
            "experiment_id": experiment_id,
            "title": (spec or {}).get("title"),
            "created": ts,
            "steps": [{"id": s.get("id"), "name": s.get("name"), "expected_activity": s.get("expected_activity")}
                      for s in (spec or {}).get("steps", [])],
            "joint_names": "BlazePose 33 (see parikshak/zerog/pose3d.py LANDMARKS)",
            "coordinate_frame": "rack (x right, y up, z out of rack face) when rack_frame is LOCKED/HOLD, else camera (y-up converted)",
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._fh = open(self.dir / "samples.jsonl", "a", encoding="utf-8")
        self.active = True
        return self.status()

    def stop(self) -> dict[str, Any]:
        if self._fh:
            self._fh.close()
            self._fh = None
        if self.active and self.dir is not None:
            mp = self.dir / "manifest.json"
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                m.update({"samples": self.n, "images": self.n_images, "duration_s": round(time.time() - self.started, 1)})
                mp.write_text(json.dumps(m, indent=2), encoding="utf-8")
            except Exception:
                pass
        self.active = False
        return self.status()

    def add(self, view: np.ndarray | None, tel: dict[str, Any]) -> None:
        if not self.active or self._fh is None:
            return
        z = tel.get("zerog", {})
        w = z.get("world", {})
        rec = {
            "i": self.n,
            "t": round(time.time() - self.started, 3),
            "experiment_id": tel.get("experiment_id"),
            "step_id": tel.get("active_step_id"),
            "step_status": {s["id"]: s["status"] for s in tel.get("steps", [])},
            "activity": z.get("har", {}).get("activity"),
            "activity_confidence": z.get("har", {}).get("confidence"),
            "frame": w.get("frame"),
            "rack_frame": z.get("status", {}).get("rack_frame"),
            "camera_roll_deg": z.get("rack", {}).get("camera_roll_deg"),
            "joints": w.get("joints"),
            "visibility": w.get("vis"),
            "objects": w.get("objects"),
            "inclination_deg": z.get("body", {}).get("inclination_deg"),
        }
        img_name = None
        if view is not None and self.n % self.every == 0:
            img_name = f"frames/{self.n:06d}.jpg"
            cv2.imwrite(str(self.dir / img_name), view, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.n_images += 1
        rec["image"] = img_name
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.n += 1
        if self.n % 30 == 0:
            self._fh.flush()

    def status(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "experiment_id": self.experiment_id,
            "samples": self.n,
            "images": self.n_images,
            "directory": str(self.dir) if self.dir else None,
        }

    @staticmethod
    def list_sessions() -> list[dict[str, Any]]:
        out = []
        if not DATA_DIR.exists():
            return out
        for m in sorted(DATA_DIR.glob("*/*/manifest.json"), reverse=True):
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({"experiment_id": d.get("experiment_id"), "session": m.parent.name,
                        "samples": d.get("samples"), "images": d.get("images"), "path": str(m.parent)})
        return out

    @staticmethod
    def zip_session(experiment_id: str, session: str) -> bytes | None:
        d = DATA_DIR / experiment_id / session
        if not d.exists() or ".." in experiment_id or ".." in session:
            return None
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in d.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(d.parent))
        return buf.getvalue()

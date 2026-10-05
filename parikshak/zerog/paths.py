"""Where runtime artefacts live.

Locally everything sits in the repo (recordings/, reports/, runs/, data/).
On a host with an ephemeral filesystem (Railway, Docker) set
PARIKSHAK_DATA_DIR to a mounted volume so recordings, flight logs, reports,
datasets and custom experiments survive redeploys.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_env = os.environ.get("PARIKSHAK_DATA_DIR", "").strip()
DATA_ROOT = Path(_env).resolve() if _env else ROOT
ON_VOLUME = bool(_env)

RECORDINGS = DATA_ROOT / "recordings"
REPORTS = DATA_ROOT / "reports"
UPLOADS = DATA_ROOT / "runs" / "uploads"
LOGS = DATA_ROOT / "runs" / "logs"
DATASETS = DATA_ROOT / "data" / "datasets"
CUSTOM_EXPERIMENTS = (DATA_ROOT / "custom_experiments") if ON_VOLUME else (ROOT / "configs" / "experiments" / "custom")

for _d in (RECORDINGS, REPORTS, UPLOADS, LOGS, DATASETS, CUSTOM_EXPERIMENTS):
    _d.mkdir(parents=True, exist_ok=True)

"""Hardware detection and mesh-backend availability (SAM 3D Body vs CPU pose mesh).

SAM 3D Body needs an NVIDIA CUDA GPU, the official package and its gated
checkpoint. If any of those is missing the backend reports MODEL UNAVAILABLE
with the exact reason - it is never silently replaced by another model's output.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from parikshak.zerog.config import ROOT

SAM3D_REPO = "https://github.com/facebookresearch/sam-3d-body"


@lru_cache(maxsize=1)
def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"nvidia_smi": False, "gpus": [], "cuda_torch": False, "torch": None}
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            for line in out.splitlines():
                name, mem, drv = [s.strip() for s in line.split(",")]
                info["gpus"].append({"name": name, "vram_mb": int(float(mem)), "driver": drv})
            info["nvidia_smi"] = bool(info["gpus"])
        except Exception as exc:  # pragma: no cover
            info["nvidia_smi_error"] = str(exc)
    try:
        from importlib.metadata import version

        info["torch"] = version("torch")
    except Exception:
        info["torch"] = None
    if info["nvidia_smi"] and info["torch"]:
        # only worth the (slow) torch import when an NVIDIA GPU is actually present
        try:
            import torch

            info["cuda_torch"] = bool(torch.cuda.is_available())
            info["torch_cuda_build"] = torch.version.cuda
        except Exception:
            pass
    info["display_adapters"] = _display_adapters()
    info["cpu"] = platform.processor() or platform.machine()
    info["cpu_threads"] = os.cpu_count()
    return info


def _display_adapters() -> list[str]:
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def sam3d_status() -> dict[str, Any]:
    g = gpu_info()
    ckpt = Path(os.environ.get("SAM3D_CHECKPOINT", str(ROOT / "models" / "sam3d" / "model.ckpt")))
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    has_pkg = importlib.util.find_spec("sam_3d_body") is not None
    reasons = []
    if not g["nvidia_smi"]:
        adapters = ", ".join(g.get("display_adapters") or []) or "none reported"
        reasons.append(f"No NVIDIA GPU found (nvidia-smi unavailable; display adapters: {adapters}).")
    if not g["cuda_torch"]:
        reasons.append(f"PyTorch {g.get('torch')} has no CUDA support on this machine.")
    if not has_pkg:
        reasons.append("Python package 'sam_3d_body' is not installed.")
    if not ckpt.exists():
        reasons.append(f"Checkpoint not found at {ckpt}.")
    available = not reasons
    return {
        "name": "SAM 3D Body",
        "available": available,
        "status": "READY" if available else "MODEL UNAVAILABLE",
        "reasons": reasons,
        "install": [
            "Requires an NVIDIA GPU with CUDA (check with: nvidia-smi).",
            "Install the CUDA build of PyTorch that matches the driver's CUDA version (pytorch.org/get-started).",
            f"Clone and install the official package: git clone {SAM3D_REPO} && pip install -e sam-3d-body",
            "Request access to and download the official checkpoint as described in that repository's README.",
            "Set SAM3D_CHECKPOINT in .env to the checkpoint path, restart the server, then switch the mesh backend.",
        ],
        "checkpoint_path": str(ckpt),
    }


def pose_backend_status(pose_available: bool, pose_error: str | None) -> dict[str, Any]:
    return {
        "name": "BlazePose GHUM 3D (MediaPipe, CPU) + pose-driven mesh",
        "available": pose_available,
        "status": "RUNNING" if pose_available else "MODEL UNAVAILABLE",
        "reasons": [] if pose_available else [pose_error or "unknown error"],
        "note": "33 metric 3D joints regressed from the GHUM body model; mesh = capsules/ellipsoids on those joints.",
    }

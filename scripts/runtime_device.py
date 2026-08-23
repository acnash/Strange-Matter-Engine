#!/usr/bin/env python3
"""Cross-platform runtime helpers for Strange Matter Engine."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


VALID_DEVICES = ("auto", "cpu", "cuda")


def requested_device(value: str | None = None) -> str:
    """Return and validate the requested computation device."""
    selected = (value or os.environ.get("SME_DEVICE", "auto")).lower()
    if selected not in VALID_DEVICES:
        choices = ", ".join(VALID_DEVICES)
        raise ValueError(f"Unknown device {selected!r}; choose one of: {choices}")
    return selected


def resolve_torch_device(torch_module, value: str | None = None):
    """Resolve auto to CUDA when available and CPU otherwise."""
    selected = requested_device(value)
    if selected == "auto":
        selected = "cuda" if torch_module.cuda.is_available() else "cpu"
    if selected == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access CUDA")
    return torch_module.device(selected)


def python_executable(value: str | None = None) -> Path:
    """Use an explicit interpreter, SME_PYTHON, or the active interpreter."""
    selected = value or os.environ.get("SME_PYTHON") or sys.executable
    return Path(selected).expanduser().resolve()


def runtime_metadata(torch_module, device) -> dict:
    """Record enough runtime information to interpret and reproduce a run."""
    return {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "pytorch": torch_module.__version__,
        "pytorch_cuda": torch_module.version.cuda,
        "device": str(device),
        "gpu": (torch_module.cuda.get_device_name(device)
                if device.type == "cuda" else None),
    }


def load_checkpoint(torch_module, path, device):
    """Load CPU- or CUDA-produced tensors onto the selected local device."""
    return torch_module.load(Path(path), map_location=device, weights_only=False)

"""Pick the torch device to run on.

Order of preference: NVIDIA GPU (``cuda``), Apple GPU (``mps``), then ``cpu``.
Set the ``GS_DEVICE`` environment variable (e.g. ``GS_DEVICE=cpu``) to override.
"""

from __future__ import annotations

import os

import torch

ENV_VAR = "GS_DEVICE"


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def get_device(prefer: str | None = None) -> torch.device:
    """Return the device to use.

    Args:
        prefer: ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"`` (or e.g. ``"cuda:1"``).
            When ``None``, the ``GS_DEVICE`` environment variable is used, and if
            that is unset, ``"auto"``.

    Raises:
        RuntimeError: if a specific device is requested but not available.
    """
    choice = (prefer or os.environ.get(ENV_VAR) or "auto").strip().lower()

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{ENV_VAR}/prefer asked for {choice!r}, but CUDA is not available")
    if device.type == "mps" and not _mps_available():
        raise RuntimeError(f"{ENV_VAR}/prefer asked for {choice!r}, but MPS is not available")
    return device

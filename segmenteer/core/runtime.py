"""Runtime helpers shared by optional accelerated segmentation backends."""

from __future__ import annotations

import os


def resolve_torch_device(
    requested: str | None = None,
    *,
    allow_mps: bool = True,
) -> str:
    """Return a usable Torch device, preferring CUDA then Apple Metal.

    ``None`` and ``"auto"`` honour ``SEGMENTEER_DEVICE`` when present and
    otherwise select the best locally available backend.  Explicit device
    choices are preserved so callers can deliberately force CPU execution.
    """
    raw = requested
    if raw is None or str(raw).strip().casefold() in {"", "auto"}:
        raw = os.environ.get("SEGMENTEER_DEVICE", "auto")
    device = str(raw).strip().casefold()

    if device != "auto":
        if device == "mps" and not allow_mps:
            return "cpu"
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if allow_mps:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    return "cpu"

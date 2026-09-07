"""Runtime configuration helpers shared by runner and accelerated backends."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Resolved process-wide runtime settings applied before methods are built."""

    device: str
    trident_device: str
    wsi_reader: str
    native_mpp: float | None
    models_dir: Path | None


def resolve_torch_device(
    requested: str | None = None,
    *,
    allow_mps: bool = True,
) -> str:
    """Return a usable Torch device, preferring CUDA then Apple Metal.

    ``None`` and ``"auto"`` honour ``SEGMENTEER_DEVICE`` when present and
    otherwise select the best locally available backend. Explicit choices are
    preserved so callers can deliberately force CPU/MPS/CUDA execution.
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


def configure_runtime(
    *,
    device: str = "auto",
    trident_device: str = "auto",
    wsi_reader: str = "auto",
    native_mpp: float | None = None,
    models_dir: str | Path | None = None,
) -> RuntimeSettings:
    """Apply runner runtime settings before segmentation methods are created.

    Segmenteer uses lazy imports, so a launcher can safely ``import segmenteer``
    first, call this function, and only then instantiate methods. This keeps the
    user-facing ``run.py`` small while ensuring optional backends observe the
    requested process environment.
    """
    reader = str(wsi_reader).strip().casefold()
    allowed_readers = {"auto", "openslide", "tifffile"}
    if reader not in allowed_readers:
        raise ValueError(
            f"wsi_reader must be one of: {', '.join(sorted(allowed_readers))}."
        )

    if native_mpp is not None:
        native_mpp = float(native_mpp)
        if not math.isfinite(native_mpp) or native_mpp <= 0:
            raise ValueError("native_mpp must be a positive finite number or None.")

    resolved_device = resolve_torch_device(device)
    requested_trident = str(trident_device).strip().casefold()
    resolved_trident = (
        resolved_device
        if requested_trident in {"", "auto"}
        else requested_trident
    )

    model_root = Path(models_dir).expanduser() if models_dir is not None else None

    os.environ["SEGMENTEER_DEVICE"] = resolved_device
    os.environ["SEGMENTEER_TRIDENT_DEVICE"] = resolved_trident
    os.environ["WSI_READER"] = reader

    if native_mpp is None:
        os.environ.pop("WSI_NATIVE_MPP", None)
    else:
        os.environ["WSI_NATIVE_MPP"] = str(native_mpp)

    if model_root is not None:
        os.environ["SEGMENTEER_MODELS_DIR"] = str(model_root)

    return RuntimeSettings(
        device=resolved_device,
        trident_device=resolved_trident,
        wsi_reader=reader,
        native_mpp=native_mpp,
        models_dir=model_root,
    )

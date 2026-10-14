"""Lazy Trident segmenter adapters.

Importing this module does not import Trident, Torch, torchvision, or a model.
A concrete model is built only when its public adapter name is accessed.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from segmenteer.model_cache import get_method_model_dir

_TRIDENT_HOME = get_method_model_dir("trident")
os.environ.setdefault("TRIDENT_HOME", str(_TRIDENT_HOME))


def _trident_segmenter_class():
    from segmenteer.core.base import TRIDENTSegmenter

    return TRIDENTSegmenter


def _load_trident_model(model_name: str):
    try:
        models = import_module("trident.segmentation_models.load")
        return getattr(models, model_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Trident-based segmenters require the Trident extra. Run: "
            "uv sync --extra trident"
        ) from exc


def _build_trident_hest() -> Any:
    return _trident_segmenter_class()(
        _load_trident_model("HESTSegmenter")(),
        model_id="HEST",
    )


def _build_trident_grandqc() -> Any:
    return _trident_segmenter_class()(
        _load_trident_model("GrandQCSegmenter")(),
        model_id="GrandQC",
    )


def _build_trident_pathprofiler() -> Any:
    try:
        from segmenteer.methods.dl.pathprofiler import LIBTRIDENTPathProfilerSegmenter

        return _trident_segmenter_class()(
            LIBTRIDENTPathProfilerSegmenter(),
            model_id="PathProfiler",
        )
    except ImportError as exc:
        raise ImportError(
            "TRIDENTPathProfilerSegmenter requires Torch, torchvision, and Trident. "
            "Run: uv sync --extra pathprofiler"
        ) from exc


def _build_trident_cpg() -> Any:
    try:
        from segmenteer.methods.dl.cpg import LIBTRIDENTCPGSegmenter

        return _trident_segmenter_class()(
            LIBTRIDENTCPGSegmenter(),
            model_id="CPG",
        )
    except ImportError as exc:
        raise ImportError(
            "TRIDENTCPGSegmenter requires Torch, torchvision, ONNX Runtime, and "
            "Trident. Run: uv sync --extra cpg"
        ) from exc


def _build_trident_rtlucassenslidesegmenter() -> Any:
    try:
        from segmenteer.methods.dl.rtlucassen import LIBTRIDENTRTLucassenSlideSegmenterSegmenter
        return _trident_segmenter_class()(
            LIBTRIDENTRTLucassenSlideSegmenterSegmenter(),
            model_id="rtlucassen",
        )
    except ImportError as exc:
        raise ImportError(
            "TRIDENTRTLucassenSlideSegmenterSegmenter requires Torch, torchvision, and "
            "Trident. Run: uv sync --extra rtlucassen"
        ) from exc

_FACTORIES: dict[str, Callable[[], Any]] = {
    "TRIDENTHESTSegmenter": _build_trident_hest,
    "TRIDENTGrandQCSegmenter": _build_trident_grandqc,
    "TRIDENTPathProfilerSegmenter": _build_trident_pathprofiler,
    "TRIDENTCPGSegmenter": _build_trident_cpg,
    "TRIDENTRTLucassenSlideSegmenterSegmenter": _build_trident_rtlucassenslidesegmenter,
}


def __getattr__(name: str) -> Any:
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'segmenteer.methods.dl.trident' has no attribute {name!r}"
        ) from exc
    value = factory()
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FACTORIES))


__all__ = list(_FACTORIES)

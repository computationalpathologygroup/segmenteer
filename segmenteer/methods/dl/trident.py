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
            "Trident-based segmenters require the Trident extra. Run: uv sync --extra trident"
        ) from exc


class DeferredTridentModel:
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._model = None

    def resolve(self) -> Any:
        if self._model is None:
            self._model = self._factory()
        return self._model


def _build_pathprofiler_model() -> Any:
    try:
        from segmenteer.methods.dl.pathprofiler import LIBTRIDENTPathProfilerSegmenter

        return LIBTRIDENTPathProfilerSegmenter()
    except ImportError as exc:
        raise ImportError(
            "TRIDENTPathProfilerSegmenter requires Torch, torchvision, and Trident. Run: uv sync --extra pathprofiler"
        ) from exc


def _build_cpg_model() -> Any:
    try:
        from segmenteer.methods.dl.cpg import LIBTRIDENTCPGSegmenter

        return LIBTRIDENTCPGSegmenter()
    except ImportError as exc:
        raise ImportError(
            "TRIDENTCPGSegmenter requires Torch, torchvision, ONNX Runtime, and Trident. Run: uv sync --extra cpg"
        ) from exc


def _build_rtlucassen_model() -> Any:
    try:
        from segmenteer.methods.dl.rtl_trident import (
            LIBTRIDENTRTLucassenSlideSegmenterSegmenter,
        )

        return LIBTRIDENTRTLucassenSlideSegmenterSegmenter()
    except ImportError as exc:
        raise ImportError(
            "TRIDENTRTLucassenSlideSegmenterSegmenter requires Torch, torchvision, and Trident. Run: uv sync --extra rtlucassen"
        ) from exc


def _build_trident_hest() -> Any:
    return _trident_segmenter_class()(
        DeferredTridentModel(lambda: _load_trident_model("HESTSegmenter")()),
        model_id="HEST",
    )


def _build_trident_grandqc() -> Any:
    return _trident_segmenter_class()(
        DeferredTridentModel(lambda: _load_trident_model("GrandQCSegmenter")()),
        model_id="GrandQC",
    )


def _build_trident_pathprofiler() -> Any:
    return _trident_segmenter_class()(
        DeferredTridentModel(_build_pathprofiler_model), model_id="PathProfiler"
    )


def _build_trident_cpg() -> Any:
    return _trident_segmenter_class()(
        DeferredTridentModel(_build_cpg_model), model_id="CPG"
    )


def _build_trident_rtlucassenslidesegmenter() -> Any:
    return _trident_segmenter_class()(
        DeferredTridentModel(_build_rtlucassen_model), model_id="rtlucassen"
    )


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

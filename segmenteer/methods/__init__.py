from __future__ import annotations
from importlib import import_module
from typing import Any

_EXPORTS = {
    "CannyForegroundSegmenter": (
        "segmenteer.methods.classical.canny_foreground",
        "CannyForegroundSegmenter",
    ),
    "DoublePassSegmenter": (
        "segmenteer.methods.classical.double_pass",
        "DoublePassSegmenter",
    ),
    "OtsuTissueSegmenter": (
        "segmenteer.methods.classical.threshold",
        "OtsuTissueSegmenter",
    ),
    "LiTissueSegmenter": (
        "segmenteer.methods.classical.threshold",
        "LiTissueSegmenter",
    ),
    "YenTissueSegmenter": (
        "segmenteer.methods.classical.threshold",
        "YenTissueSegmenter",
    ),
    "EntropyMaskerSegmenter": (
        "segmenteer.methods.classical.threshold",
        "EntropyMaskerSegmenter",
    ),
    "MorphologicalSegmenter": (
        "segmenteer.methods.classical.morphological",
        "MorphologicalSegmenter",
    ),
    "WatershedTissueSegmenter": (
        "segmenteer.methods.classical.morphological",
        "WatershedTissueSegmenter",
    ),
    "HistomicsTKMaskType": (
        "segmenteer.methods.classical.histomicstk",
        "HistomicsTKMaskType",
    ),
    "HistomicsTKTissueSegmenter": (
        "segmenteer.methods.classical.histomicstk",
        "HistomicsTKTissueSegmenter",
    ),
    "FESISegmenter": ("segmenteer.methods.classical.fesi", "FESISegmenter"),
    "BackgroundSubtractorMOG2Segmenter": (
        "segmenteer.methods.classical.background_subtractor",
        "BackgroundSubtractorMOG2Segmenter",
    ),
    "HSVThresholdSegmenter": (
        "segmenteer.methods.classical.hsv_threshold",
        "HSVThresholdSegmenter",
    ),
    "HSVThreshold_v2": (
        "segmenteer.methods.classical.hsv_threshold_v2",
        "HSVThreshold_v2",
    ),
    "RTLucassenSlideSegmenter": (
        "segmenteer.methods.dl.rtlucassen",
        "RTLucassenSlideSegmenter",
    ),
    "FastSAMSegmenter": ("segmenteer.methods.dl.fastsam", "FastSAMSegmenter"),
    "BigPictureSegmenter": ("segmenteer.methods.dl.bigpicture", "BigPictureSegmenter"),
    "HESTSegmenter": ("segmenteer.methods.dl.hest", "HESTSegmenter"),
    "GrandQCSegmenter": ("segmenteer.methods.dl.grandqc", "GrandQCSegmenter"),
    "GrandQCPatchwiseSegmenter": (
        "segmenteer.methods.dl.grandqc_patchwise",
        "GrandQCPatchwiseSegmenter",
    ),
    "AtlasPatchSAM2Segmenter": (
        "segmenteer.methods.dl.atlaspatch_sam2",
        "AtlasPatchSAM2Segmenter",
    ),
    "TRIDENTHESTSegmenter": ("segmenteer.methods.dl.trident", "TRIDENTHESTSegmenter"),
    "TRIDENTGrandQCSegmenter": (
        "segmenteer.methods.dl.trident",
        "TRIDENTGrandQCSegmenter",
    ),
    "TRIDENTPathProfilerSegmenter": (
        "segmenteer.methods.dl.trident",
        "TRIDENTPathProfilerSegmenter",
    ),
    "TRIDENTCPGSegmenter": ("segmenteer.methods.dl.trident", "TRIDENTCPGSegmenter"),
    "TRIDENTRTLucassenSlideSegmenterSegmenter": (
        "segmenteer.methods.dl.trident",
        "TRIDENTRTLucassenSlideSegmenterSegmenter",
    ),
}
_SUBMODULES = {"classical", "dl", "ml"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        result = import_module(f"{__name__}.{name}")
    else:
        try:
            module_name, attribute_name = _EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        result = getattr(import_module(module_name), attribute_name)
    globals()[name] = result
    return result


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | _SUBMODULES)


__all__ = [*_EXPORTS, *_SUBMODULES]

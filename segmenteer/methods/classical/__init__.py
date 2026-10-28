from __future__ import annotations
from importlib import import_module
from typing import Any

_EXPORTS = {
    "CannyForegroundSegmenter": ("canny_foreground", "CannyForegroundSegmenter"),
    "DoublePassSegmenter": ("double_pass", "DoublePassSegmenter"),
    "FESISegmenter": ("fesi", "FESISegmenter"),
    "MorphologicalSegmenter": ("morphological", "MorphologicalSegmenter"),
    "WatershedTissueSegmenter": ("morphological", "WatershedTissueSegmenter"),
    "EntropyMaskerSegmenter": ("threshold", "EntropyMaskerSegmenter"),
    "LiTissueSegmenter": ("threshold", "LiTissueSegmenter"),
    "OtsuTissueSegmenter": ("threshold", "OtsuTissueSegmenter"),
    "YenTissueSegmenter": ("threshold", "YenTissueSegmenter"),
    "HistomicsTKMaskType": ("histomicstk", "HistomicsTKMaskType"),
    "HistomicsTKTissueSegmenter": ("histomicstk", "HistomicsTKTissueSegmenter"),
    "BackgroundSubtractorMOG2Segmenter": (
        "background_subtractor",
        "BackgroundSubtractorMOG2Segmenter",
    ),
    "HSVThresholdSegmenter": ("hsv_threshold", "HSVThresholdSegmenter"),
    "HSVThreshold_v2": ("hsv_threshold_v2", "HSVThreshold_v2"),
}


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    result = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = result
    return result


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = list(_EXPORTS)

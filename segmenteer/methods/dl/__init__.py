from __future__ import annotations
from importlib import import_module
from typing import Any

_EXPORTS = {
    "BigPictureSegmenter": ("bigpicture", "BigPictureSegmenter"),
    "FastSAMSegmenter": ("fastsam", "FastSAMSegmenter"),
    "GrandQCSegmenter": ("grandqc", "GrandQCSegmenter"),
    "GrandQCPatchwiseSegmenter": ("grandqc_patchwise", "GrandQCPatchwiseSegmenter"),
    "NKIAllStainsSegmenter": ("nki_all_stains", "NKIAllStainsSegmenter"),
    "HESTSegmenter": ("hest", "HESTSegmenter"),
    "RTLucassenSlideSegmenter": ("rtlucassen", "RTLucassenSlideSegmenter"),
    "AtlasPatchSAM2Segmenter": ("atlaspatch_sam2", "AtlasPatchSAM2Segmenter"),
    "TRIDENTGrandQCSegmenter": ("trident", "TRIDENTGrandQCSegmenter"),
    "TRIDENTHESTSegmenter": ("trident", "TRIDENTHESTSegmenter"),
    "TRIDENTPathProfilerSegmenter": ("trident", "TRIDENTPathProfilerSegmenter"),
    "TRIDENTCPGSegmenter": ("trident", "TRIDENTCPGSegmenter"),
    "TRIDENTRTLucassenSlideSegmenterSegmenter": (
        "trident",
        "TRIDENTRTLucassenSlideSegmenterSegmenter",
    ),
    "ICGINnUNetV2Segmenter": ("icgi", "ICGINnUNetV2Segmenter"),
    "ICGIResEncSegmenter": ("icgi", "ICGIResEncSegmenter"),
    "CxTissueSegSegmenter": ("cx_tissue_seg", "CxTissueSegSegmenter"),
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

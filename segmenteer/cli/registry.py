"""Lazy, allowlisted CLI names for every public segmentation implementation.

TRIDENT names are different from ordinary methods: their exports are *instances*
of TRIDENTSegmenter with deferred third-party model loading. Never call them as
classes or try to load arbitrary saved class paths.
"""
from __future__ import annotations

# CLI spelling -> public lazy export from segmenteer.__init__.
# Maintain stable, descriptive names for the native and TRIDENT adapters.
METHODS: dict[str, str] = {
    # Classical methods
    "li": "LiTissueSegmenter",
    "otsu": "OtsuTissueSegmenter",
    "yen": "YenTissueSegmenter",
    "morphological": "MorphologicalSegmenter",
    "watershed": "WatershedTissueSegmenter",
    "hsv-threshold": "HSVThresholdSegmenter",
    "hsv-threshold-v2": "HSVThreshold_v2",
    "fesi": "FESISegmenter",
    "fesi-basic": "FESISegmenter",  # Named preset: improved=False
    "entropy-masker": "EntropyMaskerSegmenter",
    "histomicstk": "HistomicsTKTissueSegmenter",
    "double-pass": "DoublePassSegmenter",
    "canny-foreground": "CannyForegroundSegmenter",
    "background-subtractor": "BackgroundSubtractorMOG2Segmenter",
    # Standalone models
    "atlaspatch-sam2": "AtlasPatchSAM2Segmenter",
    "grandqc": "GrandQCSegmenter",
    "grandqc-patchwise": "GrandQCPatchwiseSegmenter",
    "nki-all-stains": "NKIAllStainsSegmenter",
    "fastsam": "FastSAMSegmenter",
    "bigpicture": "BigPictureSegmenter",
    "rtlucassen": "RTLucassenSlideSegmenter",
    "hest": "HESTSegmenter",
    "cx-tissue-seg": "CxTissueSegSegmenter",
    "icgi-nnunetv2": "ICGINnUNetV2Segmenter",
    "icgi-resenc": "ICGIResEncSegmenter",
    # TRIDENT adapters (not standalone constructors)
    "trident-hest": "TRIDENTHESTSegmenter",
    "trident-grandqc": "TRIDENTGrandQCSegmenter",
    "trident-pathprofiler": "TRIDENTPathProfilerSegmenter",
    "trident-cpg": "TRIDENTCPGSegmenter",
    "trident-rtlucassen": "TRIDENTRTLucassenSlideSegmenterSegmenter",
}

# Matches the model_id persisted by TRIDENTSegmenter in config.yaml.
TRIDENT_MODEL_IDS: dict[str, str] = {
    "trident-hest": "HEST",
    "trident-grandqc": "GrandQC",
    "trident-pathprofiler": "PathProfiler",
    "trident-cpg": "CPG",
    "trident-rtlucassen": "rtlucassen",
}

_TRIDENT_CLASS_PATH = "segmenteer.core.base.TRIDENTSegmenter"


def get_class(name: str):
    """Return native class or preconfigured deferred TRIDENT instance.

    Kept under its previous name to preserve the original CLI's integration
    points. `segmenteer list-methods` never calls this function.
    """
    import segmenteer

    return getattr(segmenteer, METHODS[name])


def name_from_class_path(class_path: str, params: dict | None = None) -> str:
    """Resolve a stored config from an *allowlist*, without importing its path.

    All TRIDENT variants serialize as the same class, so model_id is required
    to select the correct adapter. FESI basic also shares a class with FESI.
    """
    import segmenteer

    params = params or {}
    if class_path == _TRIDENT_CLASS_PATH:
        model_id = params.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError(
                "Saved TRIDENT configuration requires params.model_id to identify the model."
            )
        for name, stored_id in TRIDENT_MODEL_IDS.items():
            if model_id.casefold() == stored_id.casefold():
                return name
        raise ValueError(f"Unsupported saved TRIDENT model_id {model_id!r}.")
    for name, export in METHODS.items():
        if name in TRIDENT_MODEL_IDS:
            continue
        path = ".".join(segmenteer._LAZY_EXPORTS[export])
        if path == class_path:
            if export == "FESISegmenter":
                return "fesi-basic" if params.get("improved") is False else "fesi"
            return name
    raise ValueError(
        f"Unsupported saved method class {class_path!r}; choose a method from 'segmenteer list-methods'."
    )

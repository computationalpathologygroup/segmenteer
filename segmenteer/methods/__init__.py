from segmenteer.methods.classical import (
    OtsuSegmenter,
    LiSegmenter,
    YenSegmenter,
    EntropyMaskerSegmenter,
    MorphologicalSegmenter,
    WatershedSegmenter,
    ODGMMSlideSegmenter,
    HistomicsTKSegmenter,
    FESISegmenter,
)

from segmenteer.methods import classical, ml

# Background subtractor requires opencv (cv2). Import it only if available so
# the package can be used without heavy optional deps.
try:
    from segmenteer.methods.classical.background_subtractor import (
        BackgroundSubtractorMOG2Segmenter,
    )
    _has_bgsub = True
except Exception:
    BackgroundSubtractorMOG2Segmenter = None
    _has_bgsub = False

# Optional DL imports
try:
    from segmenteer.methods import dl
    _has_dl = True
except ImportError:
    _has_dl = False
    dl = None

__all__ = [
    "OtsuSegmenter",
    "LiSegmenter",
    "YenSegmenter",
    "EntropyMaskerSegmenter",
    "MorphologicalSegmenter",
    "WatershedSegmenter",
    "ODGMMSlideSegmenter",
    "HistomicsTKSegmenter",
    "RTLucassenSlideSegmenter",
    "FESISegmenter",
    "classical",
    "ml",
]

if _has_dl:
    __all__.append("dl")

if _has_bgsub:
    __all__.append("BackgroundSubtractorMOG2Segmenter")

from segmenteer.methods import classical, ensemble, ml
from segmenteer.methods.classical import (ConnectedComponentSegmenter,
                                          EdgeBasedSegmenter,
                                          EntropyMaskerSegmenter,
                                          FESISegmenter, HistomicsTKSegmenter,
                                          LiSegmenter, MorphologicalSegmenter,
                                          ODGMMSlideSegmenter, OtsuSegmenter,
                                          WatershedSegmenter, YenSegmenter)
from segmenteer.methods.ensemble import fuse_members, run_ensemble

# opencv-dependent methods — graceful fallback if cv2 is absent
try:
    from segmenteer.methods.classical.background_subtractor import \
        BackgroundSubtractorMOG2Segmenter
    from segmenteer.methods.classical.hsv_threshold import \
        HSVThresholdSegmenter

    _has_cv2 = True
except Exception:
    BackgroundSubtractorMOG2Segmenter = None
    HSVThresholdSegmenter = None
    _has_cv2 = False

# Optional DL imports
try:
    from segmenteer.methods import dl

    _has_dl = True
except ImportError:
    _has_dl = False
    dl = None

__all__ = [
    "fuse_members",
    "run_ensemble",
    "OtsuSegmenter",
    "LiSegmenter",
    "YenSegmenter",
    "EntropyMaskerSegmenter",
    "ConnectedComponentSegmenter",
    "EdgeBasedSegmenter",
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

if _has_cv2:
    __all__.append("BackgroundSubtractorMOG2Segmenter")
    __all__.append("HSVThresholdSegmenter")

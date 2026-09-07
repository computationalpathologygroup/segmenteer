"""Classical image processing methods for segmentation."""

try:
    from segmenteer.methods.classical.background_subtractor import (
        BackgroundSubtractorMOG2Segmenter,
    )
except ImportError:
    BackgroundSubtractorMOG2Segmenter = None  # type: ignore[assignment]

from segmenteer.methods.classical.fesi import FESISegmenter
from segmenteer.methods.classical.morphological import (
    MorphologicalSegmenter,
    WatershedSegmenter,
    WatershedTissueSegmenter,
)
try:
    from segmenteer.methods.classical.od_gmm import ODGMMSlideSegmenter
except ImportError:
    ODGMMSlideSegmenter = None  # type: ignore[assignment]
from segmenteer.methods.classical.threshold import (
    EntropyMaskerSegmenter,
    LiSegmenter,
    LiTissueSegmenter,
    OtsuSegmenter,
    OtsuTissueSegmenter,
    YenSegmenter,
    YenTissueSegmenter,
)

try:
    from segmenteer.methods.classical.histomicstk import (
        HistomicsTKSegmenter,
        HistomicsTKTissueSegmenter,
    )
except ImportError:
    HSVThresholdSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.classical.histomicstk import HistomicsTKSegmenter
except ImportError:
    HistomicsTKSegmenter = None  # type: ignore[assignment]

__all__ = [
    "OtsuSegmenter",
    "OtsuTissueSegmenter",
    "LiSegmenter",
    "YenSegmenter",
    "EntropyMaskerSegmenter",
    "MorphologicalSegmenter",
    "WatershedSegmenter",
    "WatershedTissueSegmenter",
    "BackgroundSubtractorMOG2Segmenter",
    "ODGMMSlideSegmenter",
    "HistomicsTKSegmenter",
    "HistomicsTKTissueSegmenter",
    "FESISegmenter",
    "HSVThresholdSegmenter",
    "LiTissueSegmenter",
    "YenTissueSegmenter",
]

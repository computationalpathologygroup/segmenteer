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
)
from segmenteer.methods.classical.od_gmm import ODGMMSlideSegmenter
from segmenteer.methods.classical.threshold import (
    EntropyMaskerSegmenter,
    LiSegmenter,
    OtsuSegmenter,
    YenSegmenter,
)

# cv2-based methods — graceful fallback if opencv is absent
try:
    from segmenteer.methods.classical.hsv_threshold import HSVThresholdSegmenter
except ImportError:
    HSVThresholdSegmenter = None  # type: ignore[assignment]

# histomicstk is optional. Import it only if available.
try:
    from segmenteer.methods.classical.histomicstk import HistomicsTKSegmenter
except ImportError:
    HistomicsTKSegmenter = None  # type: ignore[assignment]

__all__ = [
    "OtsuSegmenter",
    "LiSegmenter",
    "YenSegmenter",
    "EntropyMaskerSegmenter",
    "MorphologicalSegmenter",
    "WatershedSegmenter",
    "BackgroundSubtractorMOG2Segmenter",
    "ODGMMSlideSegmenter",
    "HistomicsTKSegmenter",
    "FESISegmenter",
    "HSVThresholdSegmenter",
]

try:
    from segmenteer.methods.dl.bigpicture import BigPictureSegmenter
except ImportError:
    BigPictureSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.fastsam import FastSAMSegmenter
except ImportError:
    FastSAMSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.grandqc import GrandQCSegmenter
except ImportError:
    GrandQCSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.hest import HESTSegmenter
except ImportError:
    HESTSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.rtlucassen import RTLucassenSlideSegmenter
except ImportError:
    RTLucassenSlideSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.trident import (
        TRIDENTGrandQCSegmenter,
        TRIDENTHESTSegmenter,
        TRIDENTPathProfilerSegmenter,
        TRIDENTCPGSegmenter
    )
except ImportError:
    TRIDENTGrandQCSegmenter = None  # type: ignore[assignment]
    TRIDENTHESTSegmenter = None  # type: ignore[assignment]
    TRIDENTPathProfilerSegmenter = None  # type: ignore[assignment]
    TRIDENTCPGSegmenter = None  # type: ignore[assignment]

__all__ = [
    "HESTSegmenter",
    "GrandQCSegmenter",
    "FastSAMSegmenter",
    "RTLucassenSlideSegmenter",
    "TRIDENTHESTSegmenter",
    "TRIDENTGrandQCSegmenter",
    "BigPictureSegmenter",
    "TRIDENTPathProfilerSegmenter",
    "TRIDENTCPGSegmenter",
]

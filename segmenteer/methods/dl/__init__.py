try:
    from segmenteer.methods.dl.atlaspatch_sam2 import AtlasPatchSAM2Segmenter
except ImportError:
    AtlasPatchSAM2Segmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.bigpicture import BigPictureSegmenter
except ImportError:
    BigPictureSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.fastsam import FastSAMSegmenter
except ImportError:
    FastSAMSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.sam3 import SAM3Segmenter
except ImportError:
    SAM3Segmenter = None  # type: ignore[assignment]

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
    )
except ImportError:
    TRIDENTGrandQCSegmenter = None  # type: ignore[assignment]
    TRIDENTHESTSegmenter = None  # type: ignore[assignment]
    TRIDENTPathProfilerSegmenter = None  # type: ignore[assignment]

try:
    from segmenteer.methods.dl.cpg import CPGSegmenter
except ImportError:
    CPGSegmenter = None  # type: ignore[assignment]

__all__ = [
    "AtlasPatchSAM2Segmenter",
    "CPGSegmenter",
    "HESTSegmenter",
    "GrandQCSegmenter",
    "FastSAMSegmenter",
    "RTLucassenSlideSegmenter",
    "TRIDENTHESTSegmenter",
    "TRIDENTGrandQCSegmenter",
    "BigPictureSegmenter",
    "TRIDENTPathProfilerSegmenter",
    "SAM3Segmenter",
]

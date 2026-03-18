from segmenteer.methods.dl.hest import HESTSegmenter
from segmenteer.methods.dl.grandqc import GrandQCSegmenter
from segmenteer.methods.dl.trident import TRIDENTHESTSegmenter, TRIDENTGrandQCSegmenter
from segmenteer.methods.dl.fastsam import FastSAMSegmenter
from segmenteer.methods.dl.rtlucassen import RTLucassenSlideSegmenter
from segmenteer.methods.dl.bigpicture import BigPictureSegmenter

try:
    from segmenteer.methods.dl.cpg import CPGSegmenter
    _has_cpg = True
except ImportError:
    CPGSegmenter = None
    _has_cpg = False

__all__ = [
    "CPGSegmenter",
    "HESTSegmenter",
    "GrandQCSegmenter",
    "FastSAMSegmenter",
    "RTLucassenSlideSegmenter",
    "TRIDENTHESTSegmenter",
    "TRIDENTGrandQCSegmenter",
    "BigPictureSegmenter",
]

import os
from pathlib import Path

os.environ["TRIDENT_HOME"] = str(
    Path(__file__).parent.parent.parent.parent / "models" / "trident"
)

try:
    from trident.segmentation_models.load import (
        GrandQCSegmenter as LIBTRIDENTGrandQCSegmenter,
    )
    from trident.segmentation_models.load import (
        HESTSegmenter as LIBTRIDENTHESTSegmenter,
    )

    from segmenteer.core.base import TRIDENTSegmenter
    from segmenteer.methods.dl.pathprofiler import LIBTRIDENTPathProfilerSegmenter
    from segmenteer.methods.dl.cpg import LIBTRIDENTCPGSegmenter

    TRIDENTGrandQCSegmenter = TRIDENTSegmenter(LIBTRIDENTGrandQCSegmenter())
    TRIDENTHESTSegmenter = TRIDENTSegmenter(LIBTRIDENTHESTSegmenter())
    TRIDENTPathProfilerSegmenter = TRIDENTSegmenter(LIBTRIDENTPathProfilerSegmenter())
    TRIDENTCPGSegmenter = TRIDENTSegmenter(LIBTRIDENTCPGSegmenter())
except ImportError:
    TRIDENTGrandQCSegmenter = None  # type: ignore[assignment]
    TRIDENTHESTSegmenter = None  # type: ignore[assignment]
    TRIDENTPathProfilerSegmenter = None  # type: ignore[assignment]
    TRIDENTCPGSegmenter = None  # type: ignore[assignment]

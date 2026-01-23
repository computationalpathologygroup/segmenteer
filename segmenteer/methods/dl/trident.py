import os
from pathlib import Path

from segmenteer.core.base import TRIDENTSegmenter
from trident.segmentation_models.load import GrandQCSegmenter as LIBTRIDENTGrandQCSegmenter
from trident.segmentation_models.load import HESTSegmenter as LIBTRIDENTHESTSegmenter

os.environ["TRIDENT_HOME"] = str(Path(__file__).parent.parent.parent.parent / "models" / "trident")

TRIDENTGrandQCSegmenter = TRIDENTSegmenter(LIBTRIDENTGrandQCSegmenter())
TRIDENTHESTSegmenter = TRIDENTSegmenter(LIBTRIDENTHESTSegmenter())

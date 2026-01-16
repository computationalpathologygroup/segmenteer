from typing import Protocol
from dataclasses import dataclass
from pathlib import Path
import geojson


@dataclass
class SegmentationResult:
    geojson: dict
    execution_time: float
    method_name: str
    metadata: dict


class Segmenter(Protocol):
    @property
    def name(self) -> str: ...

    def segment(self, image: Path) -> geojson.FeatureCollection: ...

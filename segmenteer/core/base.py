from typing import Protocol
from dataclasses import dataclass
import numpy as np


@dataclass
class SegmentationResult:
    geojson: dict
    execution_time: float
    method_name: str
    metadata: dict


class Segmenter(Protocol):
    @property
    def name(self) -> str: ...

    def segment(self, image: np.ndarray) -> dict: ...

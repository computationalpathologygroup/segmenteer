import numpy as np
from histomicstk.segmentation import simple_mask as histomicstk_simple_mask
from segmenteer.core.utils import mask_to_geojson


class HistomicsTKSegmenter:
    def __init__(self, min_area: int = 10):
        self.min_area = min_area

    @property
    def name(self) -> str:
        return "histomicstk"

    def segment(self, image: np.ndarray) -> dict:
        mask = histomicstk_simple_mask(image)
        return mask_to_geojson(mask, self.min_area)

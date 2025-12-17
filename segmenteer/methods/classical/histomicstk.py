import numpy as np
from histomicstk.segmentation import simple_mask as histomicstk_simple_tissue_mask
from histomicstk.saliency.tissue_detection import get_tissue_mask as histomicstk_saliency_tissue_mask
from segmenteer.core.utils import mask_to_geojson
from enum import StrEnum

class HistomicsTKMaskType(StrEnum):
    SIMPLE = "simple"
    SALIENCY = "saliency"


class HistomicsTKSegmenter:
    def __init__(self, mask_type: HistomicsTKMaskType = HistomicsTKMaskType.SALIENCY, min_area: int = 10):
        self.mask_type = mask_type
        self.min_area = min_area

    @property
    def name(self) -> str:
        return f"histomicstk_{self.mask_type.value}"

    def segment(self, image: np.ndarray) -> dict:
        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            mask = histomicstk_simple_tissue_mask(image)
        elif self.mask_type == HistomicsTKMaskType.SALIENCY:
            tissue_regions, _ = histomicstk_saliency_tissue_mask(image)
            mask = tissue_regions.astype(bool).astype(np.uint8)  # Convert to zero-one mask
        return mask_to_geojson(mask, self.min_area)

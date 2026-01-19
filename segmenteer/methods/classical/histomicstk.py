import numpy as np
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import Image
from enum import StrEnum
import geojson

class HistomicsTKMaskType(StrEnum):
    SIMPLE = "simple"
    SALIENCY = "saliency"


class HistomicsTKSegmenter:
    def __init__(self, level: int = 1, mask_type: HistomicsTKMaskType = HistomicsTKMaskType.SALIENCY, min_area: int = 10):
        self.level = level
        self.mask_type = mask_type
        self.min_area = min_area
        self._validate_histomicstk_available()

    def _validate_histomicstk_available(self):
        """Check if histomicstk is available, raise helpful error if not."""
        try:
            import histomicstk
        except ImportError:
            raise ImportError(
                "histomicstk is required for HistomicsTKSegmenter but not installed. "
                "Install it with: pip install 'segmenteer[histomicstk]'"
            )

    @property
    def name(self) -> str:
        return f"histomicstk_{self.mask_type.value}"

    def segment(self, image: Image) -> geojson.FeatureCollection:

        image = image.get_numpy_image(level=self.level)
        # Lazy import - only import when actually used
        from histomicstk.segmentation import simple_mask as histomicstk_simple_tissue_mask
        from histomicstk.saliency.tissue_detection import get_tissue_mask as histomicstk_saliency_tissue_mask
        
        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            mask = histomicstk_simple_tissue_mask(image)
        elif self.mask_type == HistomicsTKMaskType.SALIENCY:
            tissue_regions, _ = histomicstk_saliency_tissue_mask(image)
            mask = tissue_regions.astype(bool).astype(np.uint8)  # Convert to zero-one mask
        return mask_to_geojson(mask, self.min_area)

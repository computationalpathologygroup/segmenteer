import numpy as np
from segmenteer.core.base import NumpySegmenter
from enum import Enum

class HistomicsTKMaskType(str, Enum):
    SIMPLE = "simple"
    SALIENCY = "saliency"


class HistomicsTKSegmenter(NumpySegmenter):
    def __init__(
            self,
            mpp: float = 20,
            mask_type: HistomicsTKMaskType = HistomicsTKMaskType.SALIENCY, 
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.mask_type = mask_type
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


    def _segment_numpy(self, image):
        from histomicstk.segmentation import simple_mask as histomicstk_simple_tissue_mask
        from histomicstk.saliency.tissue_detection import get_tissue_mask as histomicstk_saliency_tissue_mask
        
        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            return histomicstk_simple_tissue_mask(image)
        elif self.mask_type == HistomicsTKMaskType.SALIENCY:
            tissue_regions, _ = histomicstk_saliency_tissue_mask(image)
            return tissue_regions.astype(bool).astype(np.uint8)  # Convert to zero-one mask

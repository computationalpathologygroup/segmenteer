from enum import Enum
import numpy as np
import skimage
from segmenteer.core.base import NumpySegmenter


class HistomicsTKMaskType(str, Enum):
    SIMPLE = "simple"
    SALIENCY = "saliency"


class HistomicsTKTissueSegmenter(NumpySegmenter):
    def __init__(
        self,
        mpp: float | None = 20,
        mask_type: HistomicsTKMaskType = HistomicsTKMaskType.SALIENCY,
        *args,
        **kwargs,
    ):
        if mpp is None:
            mpp = 20
        super().__init__(*args, mpp=mpp, **kwargs)
        self.mask_type = HistomicsTKMaskType(mask_type)
        self._validate_histomicstk_available()

    def _validate_histomicstk_available(self) -> None:
        try:
            import histomicstk
        except ImportError as exc:
            raise ImportError(
                "histomicstk is required for HistomicsTKTissueSegmenter but is not installed. Install it with: pip install 'segmenteer[histomicstk]'"
            ) from exc

    @property
    def name(self) -> str:
        return f"histomicstk_tissue_{self.mask_type.value}"

    def _preprocess(self, image):
        # HistomicsTK saliency requires RGB uint8 input for correct inversion.
        rgb = self._rgba_to_rgb(image)
        return skimage.util.img_as_ubyte(rgb)

    def _segment_numpy(self, image):
        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            from histomicstk.segmentation import (
                simple_mask as histomicstk_simple_tissue_mask,
            )

            return np.asarray(histomicstk_simple_tissue_mask(image), dtype=bool)

        # Avoid upstream component selection on all-white images.
        if np.all(image == 255):
            return np.zeros(image.shape[:2], dtype=bool)

        from histomicstk.saliency.tissue_detection import (
            get_tissue_mask as histomicstk_saliency_tissue_mask,
        )

        labeled_regions, _ = histomicstk_saliency_tissue_mask(image)
        return labeled_regions > 0

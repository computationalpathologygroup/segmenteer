from enum import Enum

import numpy as np
import skimage

from segmenteer.core.base import NumpySegmenter

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
        # Config files store enum values as plain strings.  Normalize both the
        # public Enum API and the persisted string form to one runtime type.
        self.mask_type = HistomicsTKMaskType(mask_type)
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
        from histomicstk.saliency.tissue_detection import \
            get_tissue_mask as histomicstk_saliency_tissue_mask
        from histomicstk.segmentation import \
            simple_mask as histomicstk_simple_tissue_mask

        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            return histomicstk_simple_tissue_mask(image)
        elif self.mask_type == HistomicsTKMaskType.SALIENCY:
            tissue_regions, _ = histomicstk_saliency_tissue_mask(image)
            return tissue_regions.astype(bool).astype(
                np.uint8
            )  # Convert to zero-one mask


class HistomicsTKTissueSegmenter(NumpySegmenter):
    """Corrected HistomicsTK tissue segmenter.

    `simple_mask` receives uint8 RGB input, as required by HistomicsTK.
    The saliency route retains grayscale preprocessing because HistomicsTK
    accepts grayscale thumbnails for `get_tissue_mask`.
    """

    def __init__(
        self,
        mpp: float = 20,
        mask_type: HistomicsTKMaskType = HistomicsTKMaskType.SALIENCY,
        *args,
        **kwargs,
    ):
        super().__init__(mpp=mpp, *args, **kwargs)
        self.mask_type = HistomicsTKMaskType(mask_type)
        self._validate_histomicstk_available()

    def _validate_histomicstk_available(self) -> None:
        try:
            import histomicstk  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "histomicstk is required for HistomicsTKTissueSegmenter but "
                "is not installed. Install it with: "
                "pip install 'segmenteer[histomicstk]'"
            ) from exc

    @property
    def name(self) -> str:
        return f"histomicstk_tissue_{self.mask_type.value}"

    def _preprocess(self, image):
        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            # NumpySegmenter normally converts to grayscale. HistomicsTK
            # simple_mask instead requires an H x W x 3 uint8 RGB array.
            rgb = self._rgba_to_rgb(image)
            return skimage.util.img_as_ubyte(rgb)

        # Preserve legacy preprocessing for saliency: uint8 grayscale.
        return super()._preprocess(image)

    def _segment_numpy(self, image):
        from histomicstk.saliency.tissue_detection import (
            get_tissue_mask as histomicstk_saliency_tissue_mask,
        )
        from histomicstk.segmentation import (
            simple_mask as histomicstk_simple_tissue_mask,
        )

        if self.mask_type == HistomicsTKMaskType.SIMPLE:
            return histomicstk_simple_tissue_mask(image).astype(bool)

        labeled_regions, _ = histomicstk_saliency_tissue_mask(image)

        # Preserve all retained tissue components, not only the largest one.
        return labeled_regions > 0
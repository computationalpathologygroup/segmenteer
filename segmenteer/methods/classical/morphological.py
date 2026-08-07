import geojson
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.segmentation import watershed

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.utils import mask_to_geojson


class MorphologicalSegmenter(NumpySegmenter):
    def __init__(
        self,
        mpp: int = 10,
        disk_size: int = 3,
        *args,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        self.mpp = mpp
        self.disk_size = disk_size

    @property
    def name(self) -> str:
        return f"morphological_disk{self.disk_size}"

    def _segment_numpy(self, image):
        threshold = threshold_otsu(image)
        binary = image > threshold
        selem = disk(self.disk_size)
        opened = binary_opening(binary, selem)
        return binary_closing(opened, selem)


class WatershedSegmenter(NumpySegmenter):
    def __init__(
        self,
        mpp: int = 10,
        min_distance: int = 10,
        *args,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        self.mpp = mpp
        self.min_distance = min_distance

    @property
    def name(self) -> str:
        return f"watershed_mindist{self.min_distance}"

    def _segment_numpy(self, image):
        threshold = threshold_otsu(image)
        binary = image > threshold
        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(distance, min_distance=self.min_distance, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = ndi.label(mask)[0]
        return watershed(-distance, markers, mask=binary)


class WatershedTissueSegmenter(NumpySegmenter):

    def __init__(
        self,
        mpp: float = 10,
        min_distance_um: float = 100.0,
        *args,
        **kwargs,
    ):
        mpp = float(mpp)
        min_distance_um = float(min_distance_um)

        if not np.isfinite(mpp) or mpp <= 0:
            raise ValueError(f"mpp must be a positive finite number; got {mpp!r}.")
        if not np.isfinite(min_distance_um) or min_distance_um <= 0:
            raise ValueError(
                "min_distance_um must be a positive finite number; "
                f"got {min_distance_um!r}."
            )

        super().__init__(mpp=mpp, *args, **kwargs)
        self.min_distance_um = min_distance_um

    @property
    def name(self) -> str:
        return f"watershed_tissue_mindist{self.min_distance_um:g}um"

    @property
    def min_distance_px(self) -> int:
        """Convert the requested physical marker spacing into analysis pixels."""
        return max(1, round(self.min_distance_um / self.mpp))

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=bool)

        threshold = threshold_otsu(image)
        tissue = image <= threshold

        if not tissue.any():
            return tissue

        distance = ndi.distance_transform_edt(tissue)

        coords = peak_local_max(
            distance,
            min_distance=self.min_distance_px,
            labels=tissue,
            exclude_border=False,
        )

        if coords.size == 0:
            return tissue

        markers = np.zeros(distance.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(
            1,
            len(coords) + 1,
            dtype=np.int32,
        )

        labels = watershed(-distance, markers, mask=tissue)

        return labels > 0
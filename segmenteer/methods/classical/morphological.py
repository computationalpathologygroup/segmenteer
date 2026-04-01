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

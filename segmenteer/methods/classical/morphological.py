import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, disk
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from segmenteer.core.utils import mask_to_geojson


class MorphologicalSegmenter:
    def __init__(self, disk_size: int = 3, min_area: int = 10):
        self.disk_size = disk_size
        self.min_area = min_area

    @property
    def name(self) -> str:
        return f"morphological_disk{self.disk_size}"

    def segment(self, image: np.ndarray) -> dict:
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image

        threshold = threshold_otsu(gray)
        binary = gray > threshold

        selem = disk(self.disk_size)
        opened = binary_opening(binary, selem)
        closed = binary_closing(opened, selem)

        return mask_to_geojson(closed, self.min_area)


class WatershedSegmenter:
    def __init__(self, min_distance: int = 10, min_area: int = 10):
        self.min_distance = min_distance
        self.min_area = min_area

    @property
    def name(self) -> str:
        return f"watershed_mindist{self.min_distance}"

    def segment(self, image: np.ndarray) -> dict:
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image

        threshold = threshold_otsu(gray)
        binary = gray > threshold

        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(distance, min_distance=self.min_distance, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = ndi.label(mask)[0]

        labels = watershed(-distance, markers, mask=binary)

        return mask_to_geojson(labels > 0, self.min_area)

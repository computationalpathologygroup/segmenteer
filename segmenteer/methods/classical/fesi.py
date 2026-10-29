import numpy as np
import scipy.ndimage as ndi
import skimage.filters
import skimage.morphology
import skimage.segmentation
from segmenteer.core.base import NumpySegmenter


class FESISegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(self, mpp: float | None = 20, improved: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mpp is None:
            mpp = 20
        self.mpp = mpp
        self.improved = improved

    @property
    def name(self) -> str:
        if self.improved:
            return "improved_fesi"
        else:
            return "fesi"

    def _segment_numpy(self, image):
        if self.improved:
            return improved_fesi(image)
        return fesi(image)


def _is_close(_seeds, _start) -> bool:
    _start = np.asarray(_start)
    max_dist = 500
    for seed in _seeds:
        _seed = np.asarray(seed[0])
        if np.sqrt(((_start - _seed) ** 2).sum()) <= max_dist:
            return True
    return False


def _fesi_common(image: np.ndarray) -> np.ndarray:
    blurred = skimage.filters.gaussian(image, sigma=1, truncate=5)
    mask = ((blurred >= np.mean(blurred)) * 150).astype("uint8")
    inverse_mask = ((blurred < np.mean(blurred)) * 255).astype("uint8")
    dseed = ndi.distance_transform_edt(inverse_mask)
    mask = skimage.filters.median(mask, np.ones((7, 7))) + 100
    kernel = np.ones((5, 5), np.uint8)
    mask = skimage.morphology.opening(mask, footprint=kernel)
    max_point = np.unravel_index(np.argmax(dseed, axis=None), dseed.shape)
    skimage.segmentation.flood_fill(
        mask, seed_point=max_point, new_value=0, in_place=True
    )
    mask[mask > 0] = 255
    distance = ndi.distance_transform_edt(mask)
    final_mask = mask.copy()
    maximal_distance = distance.max()
    global_max = distance.max()
    seeds: list = []
    while maximal_distance > 0:
        start = np.unravel_index(distance.argmax(), distance.shape)
        if maximal_distance > 0.6 * global_max or _is_close(seeds, start[::-1]):
            skimage.segmentation.flood_fill(
                final_mask, seed_point=start, new_value=200, in_place=True
            )
            seeds.append((start, maximal_distance))
        skimage.segmentation.flood_fill(
            mask, seed_point=start, new_value=0, in_place=True
        )
        distance[mask == 0] = 0
        maximal_distance = distance.max()
    final_mask[final_mask != 200] = 0
    final_mask[final_mask == 200] = 255
    return final_mask.astype(bool)


def improved_fesi(image: np.ndarray) -> np.ndarray:
    chnl = [0, 1]
    img_bgr = image[..., [2, 1, 0]]
    img_hsv = skimage.color.rgb2hsv(img_bgr)
    lab_bgr = skimage.color.rgb2lab(img_bgr)
    lab_bgr[..., chnl] = 100
    gray_bgr = skimage.color.rgb2gray(skimage.color.lab2rgb(lab_bgr))
    tissue_bgr = np.abs(
        skimage.filters.laplace(skimage.color.rgb2gray(img_bgr), ksize=3)
    )
    gray_bgr = (gray_bgr >= np.mean(gray_bgr)) * 1.0
    tissue_rgb_hsv_1 = tissue_bgr * img_hsv[..., 1] * gray_bgr
    return _fesi_common(tissue_rgb_hsv_1)


def fesi(image: np.ndarray) -> np.ndarray:
    gray = skimage.color.rgb2gray(image)
    gray = np.abs(skimage.filters.laplace(gray, ksize=3))
    return _fesi_common(gray)

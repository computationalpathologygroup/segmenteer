import warnings
from functools import partial
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.filters.rank import entropy
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.util import apply_parallel
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import Image
from pathlib import Path
import pyvips
import geojson


class DaskWarning(UserWarning):
    pass


warnings.simplefilter("once", DaskWarning)


class OtsuSegmenter:
    def __init__(self, level: int = 1, min_area: int = 10):
        self.level = level
        self.min_area = min_area

    @property
    def name(self) -> str:
        return "otsu"

    def segment(self, image: Image) -> geojson.FeatureCollection:
        image_np = image.get_numpy_image(level=self.level)
        if image_np.ndim == 3:
            gray = rgb2gray(image_np)
        else:
            gray = image_np

        threshold = threshold_otsu(gray)
        mask = gray > threshold

        return mask_to_geojson(mask, self.min_area, scaling_factor=image.get_scaling(self.level))


class LiSegmenter:
    def __init__(self, level: int = 1, min_area: int = 10):
        self.level = level
        self.min_area = min_area

    @property
    def name(self) -> str:
        return "li"

    def segment(self, image: Image) -> geojson.FeatureCollection:
        image_np = image.get_numpy_image(level=self.level)
        if image_np.ndim == 3:
            gray = rgb2gray(image_np)
        else:
            gray = image_np

        threshold = threshold_li(gray)
        mask = gray > threshold

        return mask_to_geojson(mask, self.min_area, scaling_factor=image.get_scaling(self.level))


class YenSegmenter:
    def __init__(self, level: int = 1, min_area: int = 10):
        self.level = level
        self.min_area = min_area

    @property
    def name(self) -> str:
        return "yen"

    def segment(self, image: Image) -> geojson.FeatureCollection:
        image_np = image.get_numpy_image(level=self.level)
        if image_np.ndim == 3:
            gray = rgb2gray(image_np)
        else:
            gray = image_np

        threshold = threshold_yen(gray)
        mask = gray > threshold

        return mask_to_geojson(mask, self.min_area, scaling_factor=image.get_scaling(self.level))


class EntropyMaskerSegmenter:
    """Unofficial implementation of the EntropyMasker algorithm [1] to extract foreground from histopathology images.

    Parameters
    ----------
    min_area : int, default=10
        Minimum area of a polygon to be included in the output.
    footprint : np.ndarray, default=`skimage.morphology.disk(9)`
        Footprint to use with `skimage.filters.rank.entropy`.
    to_gray_func : Callable, default=`np.max(..., axis=2)`
        Function to convert a RGB image to grayscale.

    References
    ----------
    [1] https://doi.org/10.1038/s41598-023-29638-1
    """

    def __init__(
        self,
        level: int = 1,
        min_area: int = 10,
        footprint: Optional[npt.NDArray] = None,
        to_gray_func: Callable = partial(np.max, axis=2),
    ):
        self.level = level
        self.min_area = min_area
        self.footprint = footprint
        self.to_gray_func = to_gray_func

    @property
    def name(self) -> str:
        return "entropy_masker"

    def segment(self, image: Image) -> geojson.FeatureCollection:
        image_np = image.get_numpy_image(level=self.level)
        if image_np.ndim == 3:
            gray = self.to_gray_func(image_np)
        else:
            gray = image_np

        mask = entropy_masker(gray, self.footprint)

        return mask_to_geojson(mask, self.min_area, scaling_factor=image.get_scaling(self.level))


def entropy_masker(
    image: npt.NDArray,
    footprint: Optional[npt.NDArray[np.int_]] = None,
) -> npt.NDArray[np.bool_]:
    """
    Extract foreground from background in histopathological images using an Otsu threshold on local entropy.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    footprint : np.ndarray, default=`skimage.morphology.disk(9)`
        Footprint to use with `skimage.filters.rank.entropy`.
    Returns
    -------
    np.ndarray
        Tissue mask

    References
    ----------
    .. [1] https://doi.org/10.1038/s41598-023-29638-1
    """
    if footprint is None:
        footprint = disk(9)

    try:
        ent = apply_parallel(
            entropy,
            image,
            dtype=image.dtype,
            extra_arguments=(footprint,),
        )
    except RuntimeError as e:
        warnings.warn(
            f"skimage.filters.rank.entropy in using skimage.util.apply_parallel failed with RuntimeError: {e}. "
            "Falling back to sequential processing. This will be slower. Run pip install segmenteer[entropymasker] for faster processing.",
            DaskWarning,
        )
        ent = entropy(image, footprint)
    threshold: float = threshold_otsu(ent)
    mask: npt.NDArray[np.bool_] = ent >= threshold
    return mask

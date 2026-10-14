import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from skimage.filters import threshold_li, threshold_otsu, threshold_yen
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import apply_parallel

from segmenteer.core.base import NumpySegmenter


class DaskWarning(UserWarning):
    pass


warnings.simplefilter("once", DaskWarning)


class OtsuSegmenter(NumpySegmenter):
    def __init__(self, mpp: float = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpp = mpp

    @property
    def name(self) -> str:
        return "otsu"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        return image > threshold_otsu(image)


class LiSegmenter(NumpySegmenter):
    def __init__(self, mpp: float = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpp = mpp

    @property
    def name(self) -> str:
        return "li"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        return image > threshold_li(image)


class YenSegmenter(NumpySegmenter):
    def __init__(self, mpp: float = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpp = mpp

    @property
    def name(self) -> str:
        return "yen"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        return image > threshold_yen(image)


class EntropyMaskerSegmenter(NumpySegmenter):
    """Unofficial implementation of the EntropyMasker algorithm [1] to extract foreground from histopathology images.

    Parameters
    ----------
    min_area : int, default=0
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
        footprint: npt.NDArray[np.int_] | None = None,
        to_gray_func: Union[callable, None] = None,
        *args,
        **kwargs,
    ) -> None:
        if to_gray_func is None:
            to_gray_func = partial(np.max, axis=2)
        super().__init__(to_gray_func=to_gray_func, *args, **kwargs)
        self.footprint = footprint

    @property
    def name(self) -> str:
        return "entropy_masker"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        return entropy_masker(image, self.footprint)


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


class OtsuTissueSegmenter(NumpySegmenter):
    """Otsu tissue mask for light-background histology images.

    Unlike the OtsuSegmenter, dark pixels are treated as tissue and
    bright pixels as slide background.
    """

    def __init__(self, mpp: float = 20, *args, **kwargs):
        super().__init__(mpp=mpp, *args, **kwargs)

    @property
    def name(self) -> str:
        return "otsu_tissue"

    def _segment_numpy(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.bool_]:
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.bool_)

        threshold = threshold_otsu(image)

        # Histology tissue is generally darker than the bright slide background.
        return image <= threshold
    

class LiTissueSegmenter(NumpySegmenter):
    """Li global thresholding with dark histology tissue as foreground."""

    def __init__(self, mpp: float = 20, *args, **kwargs):
        super().__init__(mpp=mpp, *args, **kwargs)

    @property
    def name(self) -> str:
        return "li_tissue"

    def _segment_numpy(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.bool_]:
        # Avoid labelling a uniform white or black image as tissue.
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.bool_)

        threshold = threshold_li(image)

        # Stained tissue is darker than the bright glass-slide background.
        return image <= threshold


class YenTissueSegmenter(NumpySegmenter):
    """Yen global thresholding with dark histology tissue as foreground."""

    def __init__(self, mpp: float = 20, *args, **kwargs):
        super().__init__(mpp=mpp, *args, **kwargs)

    @property
    def name(self) -> str:
        return "yen_tissue"

    def _segment_numpy(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.bool_]:
        # Avoid labelling a uniform white or black image as tissue.
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.bool_)

        threshold = threshold_yen(image)

        # Stained tissue is darker than the bright glass-slide background.
        return image <= threshold

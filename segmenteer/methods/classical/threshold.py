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


class EntropyMaskerSegmenter(NumpySegmenter):
    def __init__(
        self,
        footprint: npt.NDArray[np.int_] | None = None,
        to_gray_func: Union[callable, None] = None,
        mpp: float | None = 10,
        *args,
        **kwargs,
    ) -> None:
        if to_gray_func is None:
            to_gray_func = partial(np.max, axis=2)
        super().__init__(*args, to_gray_func=to_gray_func, **kwargs)
        self.footprint = footprint
        if mpp is None:
            mpp = 10
        self.mpp = mpp

    @property
    def name(self) -> str:
        return "entropy_masker"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        return entropy_masker(image, self.footprint)


def entropy_masker(
    image: npt.NDArray, footprint: Optional[npt.NDArray[np.int_]] = None
) -> npt.NDArray[np.bool_]:
    if footprint is None:
        footprint = disk(9)
    try:
        ent = apply_parallel(
            entropy, image, dtype=image.dtype, extra_arguments=(footprint,)
        )
    except RuntimeError as e:
        warnings.warn(
            f"skimage.filters.rank.entropy in using skimage.util.apply_parallel failed with RuntimeError: {e}. Falling back to sequential processing. This will be slower. Run pip install segmenteer[entropymasker] for faster processing.",
            DaskWarning,
        )
        ent = entropy(image, footprint)
    threshold: float = threshold_otsu(ent)
    mask: npt.NDArray[np.bool_] = ent >= threshold
    return mask


class OtsuTissueSegmenter(NumpySegmenter):
    def __init__(self, mpp: float | None = 20, *args, **kwargs):
        if mpp is None:
            mpp = 20
        super().__init__(*args, mpp=mpp, **kwargs)

    @property
    def name(self) -> str:
        return "otsu_tissue"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.bool_)
        threshold = threshold_otsu(image)
        return image <= threshold


class LiTissueSegmenter(NumpySegmenter):
    def __init__(self, mpp: float | None = 20, *args, **kwargs):
        if mpp is None:
            mpp = 20
        super().__init__(*args, mpp=mpp, **kwargs)

    @property
    def name(self) -> str:
        return "li_tissue"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.bool_)
        threshold = threshold_li(image)
        return image <= threshold


class YenTissueSegmenter(NumpySegmenter):
    def __init__(self, mpp: float | None = 20, *args, **kwargs):
        if mpp is None:
            mpp = 20
        super().__init__(*args, mpp=mpp, **kwargs)

    @property
    def name(self) -> str:
        return "yen_tissue"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.bool_)
        threshold = threshold_yen(image)
        return image <= threshold

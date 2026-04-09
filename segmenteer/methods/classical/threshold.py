import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.feature import canny
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
        # Tissue is darker than the near-white glass background, so foreground
        # pixels lie *below* the Otsu threshold (not above).
        return image < threshold_otsu(image)


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


class ConnectedComponentSegmenter(NumpySegmenter):
    """Extracts foreground by thresholding and connected component labeling.

    Applies Otsu thresholding to create a binary image, then identifies connected
    components and returns all components above the minimum area threshold.

    Parameters
    ----------
    connectivity : int, default=2
        Connectivity for labeling: 1 for 4-connectivity, 2 for 8-connectivity.
    """

    def __init__(
        self,
        mpp: float = 20,
        connectivity: int = 2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.connectivity = connectivity

    @property
    def name(self) -> str:
        return f"connected_components_c{self.connectivity}"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        from skimage.measure import label

        # Apply Otsu thresholding
        binary = image > threshold_otsu(image)

        # Label connected components
        labeled = label(binary, connectivity=self.connectivity)

        # Filter components by minimum area
        mask = np.zeros_like(binary)
        for component_id in np.unique(labeled):
            if component_id == 0:  # Skip background
                continue
            component = labeled == component_id
            if component.sum() >= self.min_area:
                mask |= component

        return mask.astype(bool)


class EdgeBasedSegmenter(NumpySegmenter):
    """Extracts foreground using edge detection (Canny or Sobel).

    Detects edges in the image and identifies regions between edges via
    connected component labeling. Edges act as boundaries between foreground
    and background regions.

    Parameters
    ----------
    method : str, default='canny'
        Edge detection method: 'canny' or 'sobel'.
    sigma : float, default=1.0
        Standard deviation for Gaussian blur (Canny only). Controls edge smoothness.
    sobel_threshold : float or None, default=None
        Manual threshold for Sobel magnitude. If None, uses Otsu on magnitude.
    """

    def __init__(
        self,
        mpp: float = 20,
        method: str = "canny",
        sigma: float = 1.0,
        sobel_threshold: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        if method not in ("canny", "sobel"):
            raise ValueError(f"method must be 'canny' or 'sobel', got '{method}'")
        self.method = method
        self.sigma = sigma
        self.sobel_threshold = sobel_threshold

    @property
    def name(self) -> str:
        return f"edge_based_{self.method}"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        from skimage.filters import sobel
        from skimage.measure import label

        if self.method == "canny":
            edges = canny(image, sigma=self.sigma)
        else:  # sobel
            sobel_mag = sobel(image)
            if self.sobel_threshold is not None:
                edges = sobel_mag > self.sobel_threshold
            else:
                edges = sobel_mag > threshold_otsu(sobel_mag)

        # Invert edges: non-edges become 255, edges become 0
        inverted_edges = ~edges

        # Label connected components in inverted edge space
        labeled = label(inverted_edges)

        # Build mask from components, filtering by minimum area
        mask = np.zeros_like(inverted_edges)
        for component_id in np.unique(labeled):
            if component_id == 0:  # Skip background (edges)
                continue
            component = labeled == component_id
            if component.sum() >= self.min_area:
                mask |= component

        return mask.astype(bool)

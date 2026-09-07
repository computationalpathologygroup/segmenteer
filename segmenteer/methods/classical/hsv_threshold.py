"""HSV colour-range tissue segmenter.

Converts an RGB thumbnail to HSV and applies a fixed hue/saturation/value
range mask to detect H&E-stained tissue.  The default bounds:

    lower = [90,   8, 103]  (H, S, V)
    upper = [180, 255, 255]

target the purple-to-pink hue range typical of haematoxylin and eosin stains
while rejecting white/grey background (low saturation) and very dark artefacts
(low value).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from segmenteer.core.base import NumpySegmenter

_DEFAULT_LOWER = np.array([90, 8, 103], dtype=np.uint8)
_DEFAULT_UPPER = np.array([180, 255, 255], dtype=np.uint8)


class HSVThresholdSegmenter(NumpySegmenter):
    """Tissue segmenter based on HSV colour-range thresholding.

    Parameters
    ----------
    lower : array-like of 3 uint8, default ``[90, 8, 103]``
        Lower bound of the HSV range (H, S, V) passed to ``cv2.inRange``.
    upper : array-like of 3 uint8, default ``[180, 255, 255]``
        Upper bound of the HSV range (H, S, V).
    mpp : float, default 0
        Microns-per-pixel at which the WSI is downsampled before segmentation.
    min_area : int, default 0
        Minimum polygon area (in pixels at *mpp* resolution) to retain.
    """

    # We need the original RGB image — do not convert to grayscale.
    APPLY_TO_GRAYSCALE: bool = False

    def __init__(
        self,
        lower: list[int] | npt.NDArray[np.uint8] | None = None,
        upper: list[int] | npt.NDArray[np.uint8] | None = None,
        mpp: float = 10,
        min_area: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(mpp=mpp, min_area=min_area, **kwargs)
        self.lower = list(map(int, lower if lower is not None else _DEFAULT_LOWER))
        self.upper = list(map(int, upper if upper is not None else _DEFAULT_UPPER))

    @property
    def name(self) -> str:
        return "hsv-threshold"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        """Apply HSV range threshold to an RGB *image* and return a binary mask."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "HSVThresholdSegmenter requires opencv-python. "
                "Install it with:  pip install opencv-python-headless"
            ) from exc

        lower = np.array(self.lower, dtype=np.uint8)
        upper = np.array(self.upper, dtype=np.uint8)

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, lower, upper) > 0
        return mask.astype(bool)

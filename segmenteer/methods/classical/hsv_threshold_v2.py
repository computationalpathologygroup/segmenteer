from __future__ import annotations

from numbers import Real

import numpy as np
import numpy.typing as npt

from segmenteer.core.base import NumpySegmenter


_DEFAULT_LOWER = (0, 15, 0)
_DEFAULT_UPPER = (179, 255, 255)


def _parse_hsv_bound(values, name: str) -> list[int]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain exactly three HSV channel values."
        ) from exc
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three HSV channel values.")
    result = []
    for index, (value, maximum) in enumerate(zip(items, (179, 255, 255))):
        if name == "upper" and index == 0 and value == 180:
            value = 179
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value != int(value)
            or not 0 <= value <= maximum
        ):
            raise ValueError(f"{name}[{index}] must be an integer from 0 to {maximum}.")
        result.append(int(value))
    return result


class HSVThreshold_v2(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        lower: list[int] | npt.NDArray[np.uint8] | None = None,
        upper: list[int] | npt.NDArray[np.uint8] | None = None,
        mpp: float | None = 10,
        min_area: int = 0,
        algorithm_revision: int = 2,
        **kwargs,
    ) -> None:
        if mpp is None:
            mpp = 10
        if algorithm_revision != 2:
            raise ValueError("Unsupported HSVThreshold_v2 algorithm revision.")
        parsed_lower = _parse_hsv_bound(
            _DEFAULT_LOWER if lower is None else lower, "lower"
        )
        parsed_upper = _parse_hsv_bound(
            _DEFAULT_UPPER if upper is None else upper, "upper"
        )
        for index in (1, 2):
            if parsed_lower[index] > parsed_upper[index]:
                raise ValueError(f"lower[{index}] must not exceed upper[{index}].")
        super().__init__(mpp=mpp, min_area=min_area, **kwargs)
        self.lower = parsed_lower
        self.upper = parsed_upper
        self.algorithm_revision = 2

    @property
    def name(self) -> str:
        return "hsv-threshold-v2"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "HSVThreshold_v2 requires opencv-python. Install it with: pip install opencv-python-headless"
            ) from exc
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "HSVThreshold_v2 requires an RGB image with three channels."
            )
        if not image.shape[0] or not image.shape[1]:
            return np.zeros(image.shape[:2], dtype=bool)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        low = self.lower
        high = self.upper
        saturation_and_value = (
            cv2.inRange(
                hsv,
                np.array((0, low[1], low[2]), dtype=np.uint8),
                np.array((179, high[1], high[2]), dtype=np.uint8),
            )
            > 0
        )
        hue = hsv[:, :, 0]
        if low[0] <= high[0]:
            hue_in_range = (hue >= low[0]) & (hue <= high[0])
        else:
            hue_in_range = (hue >= low[0]) | (hue <= high[0])
        return saturation_and_value & hue_in_range

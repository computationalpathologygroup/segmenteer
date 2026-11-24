"""Classical foreground preprocessing described by Skrede et al. (2026).

DOI: https://doi.org/10.1038/s41698-026-01311-6
Source: https://github.com/icgi/automatic-tumour-segmentation-in-WSIs

Implements the published 5 µm/px Canny and morphology procedure, not the
paper's neural tumour segmenter. No upstream code or weights are bundled.
The research source is Apache-2.0 licensed; the article is CC BY 4.0.
"""

from __future__ import annotations

import inspect

import numpy as np

from segmenteer.core.base import NumpySegmenter


def _remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """Preserve strict hole-area thresholds across scikit-image versions."""
    if area_threshold <= 1:
        return mask
    from skimage.morphology import remove_small_holes

    if "max_size" in inspect.signature(remove_small_holes).parameters:
        return remove_small_holes(mask, max_size=area_threshold - 1)
    return remove_small_holes(mask, area_threshold=area_threshold)


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Preserve strict object-area thresholds across scikit-image versions."""
    if min_size <= 1:
        return mask
    from skimage.morphology import remove_small_objects

    if "max_size" in inspect.signature(remove_small_objects).parameters:
        return remove_small_objects(mask, max_size=min_size - 1)
    return remove_small_objects(mask, min_size=min_size)


class CannyForegroundSegmenter(NumpySegmenter):
    """Classical Canny foreground detector at 5 µm/px (Skrede et al., 2026)."""

    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 5.0,
        *,
        low_threshold: int = 10,
        high_threshold: int = 50,
        kernel_size: int = 9,
        hole_area_px: int = 10_000,
        min_object_area_px: int = 1_600,
        algorithm_revision: int = 1,
        **kwargs,
    ) -> None:
        # Area thresholds are calibrated at the published 5 µm/px resolution.
        if not np.isfinite(mpp) or abs(float(mpp) - 5.0) > 1e-6:
            raise ValueError("The published Canny foreground method requires mpp=5.")
        if algorithm_revision != 1:
            raise ValueError("Unsupported CannyForegroundSegmenter algorithm revision.")
        for label, value in (
            ("low_threshold", low_threshold),
            ("high_threshold", high_threshold),
            ("hole_area_px", hole_area_px),
            ("min_object_area_px", min_object_area_px),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{label} must be a nonnegative integer.")
        if high_threshold <= low_threshold:
            raise ValueError("high_threshold must exceed low_threshold.")
        if (
            isinstance(kernel_size, (bool, np.bool_))
            or not isinstance(kernel_size, (int, np.integer))
            or kernel_size < 1
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer.")
        super().__init__(mpp=5.0, **kwargs)
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.kernel_size = int(kernel_size)
        self.hole_area_px = int(hole_area_px)
        self.min_object_area_px = int(min_object_area_px)
        self.algorithm_revision = 1

    @property
    def name(self) -> str:
        return "canny_foreground_mpp5"

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        """Return a boolean tissue/foreground mask at the input thumbnail size."""
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Canny foreground requires an RGB uint8 image (H, W, 3).")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Canny foreground requires a nonempty image.")
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Canny foreground requires OpenCV: pip install opencv-python-headless"
            ) from exc

        # The reference method applies Canny to RGB without grayscale conversion.
        edges = cv2.Canny(
            image, self.low_threshold, self.high_threshold, apertureSize=3
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.kernel_size, self.kernel_size),
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) > 0
        filled = _remove_small_holes(closed, self.hole_area_px)
        opened = (
            cv2.morphologyEx(
                np.asarray(filled, dtype=np.uint8),
                cv2.MORPH_OPEN,
                kernel,
            )
            > 0
        )
        cleaned = _remove_small_objects(opened, self.min_object_area_px)
        return np.asarray(cleaned, dtype=bool)

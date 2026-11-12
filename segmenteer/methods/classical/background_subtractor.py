from __future__ import annotations

import numpy as np
import numpy.typing as npt

from segmenteer.core.base import NumpySegmenter


class BackgroundSubtractorMOG2Segmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float | None = 20,
        var_threshold: float = 16.0,
        history: int = 32,
        training_frames: int = 32,
        min_background_value: int = 215,
        max_background_saturation: int = 25,
        min_background_fraction: float = 0.005,
        initial_variance: float = 64.0,
        tile_size: int = 1024,
        random_state: int = 0,
        algorithm_revision: int = 2,
        **kwargs,
    ) -> None:
        if mpp is None:
            mpp = 20
        if not np.isfinite(mpp) or mpp <= 0:
            raise ValueError("mpp must be positive and finite.")
        if not np.isfinite(var_threshold) or var_threshold <= 0:
            raise ValueError("var_threshold must be positive and finite.")
        if not isinstance(history, int) or isinstance(history, bool) or history < 2:
            raise ValueError("history must be an integer greater than 1.")
        if (
            not isinstance(training_frames, int)
            or isinstance(training_frames, bool)
            or training_frames < 2
        ):
            raise ValueError("training_frames must be an integer greater than 1.")
        if (
            not isinstance(min_background_value, int)
            or not 0 <= min_background_value <= 255
        ):
            raise ValueError("min_background_value must be an integer from 0 to 255.")
        if (
            not isinstance(max_background_saturation, int)
            or not 0 <= max_background_saturation <= 255
        ):
            raise ValueError(
                "max_background_saturation must be an integer from 0 to 255."
            )
        if (
            not np.isfinite(min_background_fraction)
            or not 0 < min_background_fraction <= 1
        ):
            raise ValueError("min_background_fraction must be in (0, 1].")
        if not np.isfinite(initial_variance) or initial_variance <= 0:
            raise ValueError("initial_variance must be positive and finite.")
        if (
            not isinstance(tile_size, int)
            or isinstance(tile_size, bool)
            or tile_size < 1
        ):
            raise ValueError("tile_size must be a positive integer.")
        if (
            not isinstance(random_state, int)
            or isinstance(random_state, bool)
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer.")
        if algorithm_revision != 2:
            raise ValueError(
                "Unsupported BackgroundSubtractorMOG2Segmenter algorithm revision."
            )
        super().__init__(mpp=mpp, **kwargs)
        self.var_threshold = float(var_threshold)
        self.history = history
        self.training_frames = training_frames
        self.min_background_value = min_background_value
        self.max_background_saturation = max_background_saturation
        self.min_background_fraction = float(min_background_fraction)
        self.initial_variance = float(initial_variance)
        self.tile_size = tile_size
        self.random_state = random_state
        self.algorithm_revision = 2

    @property
    def name(self) -> str:
        return "BackgroundsubtractorMOG2"

    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "BackgroundSubtractorMOG2Segmenter requires OpenCV. Install opencv-python-headless."
            ) from exc
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                "BackgroundSubtractorMOG2Segmenter expects an RGB uint8 image with shape (H, W, 3)."
            )
        height, width, _ = image.shape
        if not height or not width:
            return np.zeros((height, width), dtype=bool)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        candidate = (hsv[:, :, 1] <= self.max_background_saturation) & (
            hsv[:, :, 2] >= self.min_background_value
        )
        count = int(np.count_nonzero(candidate))
        minimum = max(8, int(np.ceil(height * width * self.min_background_fraction)))
        if count < minimum:
            raise ValueError(
                f"Cannot calibrate MOG2: only {count} high-confidence background pixels found; "
                f"at least {minimum} required. Provide slides with visible glass background or "
                f"adjust min_background_value, max_background_saturation and min_background_fraction."
            )
        background_rgb = image[candidate]
        rng = np.random.default_rng(self.random_state)
        reference = background_rgb[rng.integers(0, count, size=self.training_frames)]
        reference_bgr = np.ascontiguousarray(reference[:, ::-1])
        del background_rgb
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output = np.empty((height, width), dtype=bool)
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                tile = image_bgr[y : y + self.tile_size, x : x + self.tile_size]
                model = cv2.createBackgroundSubtractorMOG2(
                    history=self.history,
                    varThreshold=self.var_threshold,
                    detectShadows=False,
                )
                model.setVarInit(self.initial_variance)
                calibration_frame = np.empty_like(tile)
                for frame_index, color in enumerate(reference_bgr):
                    calibration_frame[:] = color
                    model.apply(calibration_frame, learningRate=1.0 / (frame_index + 1))
                foreground = model.apply(tile, learningRate=0.0) == 255
                tile_background = candidate[
                    y : y + tile.shape[0], x : x + tile.shape[1]
                ]
                background_mismatch = (
                    float(np.mean(foreground[tile_background]))
                    if np.any(tile_background)
                    else 0.0
                )
                if background_mismatch > 0.5:
                    raise RuntimeError(
                        "MOG2 background calibration was unstable: more than half of calibrated glass pixels were classified as foreground."
                    )
                output[y : y + tile.shape[0], x : x + tile.shape[1]] = foreground
        return output

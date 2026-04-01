import numpy as np

from segmenteer.core.base import NumpySegmenter


class BackgroundSubtractorMOG2Segmenter(NumpySegmenter):
    def __init__(
        self,
        mpp: float = 20,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        *args,
        **kwargs,
    ):
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ImportError(
                "opencv-python is required for BackgroundSubtractorMOG2Segmenter.\n"
                "Install with: pip install 'segmenteer[background-subtractor]'"
            ) from None
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows

    @property
    def name(self) -> str:
        return f"bg_subtractor_mog2_h{self.history}_v{int(self.var_threshold)}"

    def _segment_numpy(self, image):
        import cv2
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image

        if len(image_uint8.shape) == 2:
            import cv2
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

        # Create white background - use zeros then add 255 to avoid issues with large arrays
        blank_background = np.zeros(image_uint8.shape, dtype=np.uint8)
        blank_background[:] = 255

        for _ in range(self.history):
            bg_subtractor.apply(blank_background, learningRate=1.0)

        fg_mask = bg_subtractor.apply(image_uint8, learningRate=0)

        if self.detect_shadows:
            fg_mask[fg_mask == 127] = 0

        return fg_mask > 0

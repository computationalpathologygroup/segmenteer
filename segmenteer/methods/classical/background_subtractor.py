import numpy as np

from segmenteer.core.base import NumpySegmenter


class BackgroundSubtractorMOG2Segmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 20,
        var_threshold: float = 16.0,
        *args,
        **kwargs,
    ):
        super().__init__(mpp=mpp, *args, **kwargs)
        self.var_threshold = var_threshold

    def _segment_numpy(self, image):
        import cv2
        import numpy as np

        image_uint8 = image.astype(np.uint8, copy=False)

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=1,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )

        white_reference = np.full_like(image_uint8, 255)
        bg_subtractor.apply(white_reference, learningRate=1.0)

        fg_mask = bg_subtractor.apply(image_uint8, learningRate=0.0)
        return fg_mask > 0

    @property
    def name(self):
        return "BackgroundsubtractorMOG2"
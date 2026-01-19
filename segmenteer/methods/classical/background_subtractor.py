import numpy as np
import cv2
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import Image
import geojson


class BackgroundSubtractorMOG2Segmenter:
    def __init__(
        self,
        level: int = 1,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_area: int = 10,
    ):
        self.level = level
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_area = min_area

    @property
    def name(self) -> str:
        return f"bg_subtractor_mog2_h{self.history}_v{int(self.var_threshold)}"

    def segment(self, image: Image) -> geojson.FeatureCollection:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

        image_np = image.get_numpy_image(level=self.level)

        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_uint8 = (image_np * 255).astype(np.uint8)
            else:
                image_uint8 = image_np.astype(np.uint8)
        else:
            image_uint8 = image_np

        if len(image_uint8.shape) == 2:
            image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

        # Create white background - use zeros then add 255 to avoid issues with large arrays
        blank_background = np.zeros(image_uint8.shape, dtype=np.uint8)
        blank_background[:] = 255
        
        for _ in range(self.history):
            bg_subtractor.apply(blank_background, learningRate=1.0)
        
        fg_mask = bg_subtractor.apply(image_uint8, learningRate=0)

        if self.detect_shadows:
            fg_mask[fg_mask == 127] = 0

        mask = fg_mask > 0

        return mask_to_geojson(mask, self.min_area, scaling_factor=image.get_scaling(self.level))

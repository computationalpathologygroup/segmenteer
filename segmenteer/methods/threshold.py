import numpy as np
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.color import rgb2gray
from segmenteer.core.utils import mask_to_geojson


class OtsuSegmenter:
    def __init__(self, min_area: int = 10):
        self.min_area = min_area
    
    @property
    def name(self) -> str:
        return "otsu"
    
    def segment(self, image: np.ndarray) -> dict:
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        threshold = threshold_otsu(gray)
        mask = gray > threshold
        return mask_to_geojson(mask, self.min_area)


class LiSegmenter:
    def __init__(self, min_area: int = 10):
        self.min_area = min_area
    
    @property
    def name(self) -> str:
        return "li"
    
    def segment(self, image: np.ndarray) -> dict:
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        threshold = threshold_li(gray)
        mask = gray > threshold
        return mask_to_geojson(mask, self.min_area)


class YenSegmenter:
    def __init__(self, min_area: int = 10):
        self.min_area = min_area
    
    @property
    def name(self) -> str:
        return "yen"
    
    def segment(self, image: np.ndarray) -> dict:
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        threshold = threshold_yen(gray)
        mask = gray > threshold
        return mask_to_geojson(mask, self.min_area)
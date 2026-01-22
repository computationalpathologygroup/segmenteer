from abc import ABC, abstractmethod
from typing import Protocol, Union
from dataclasses import dataclass
from pathlib import Path
import geojson
import numpy.typing as npt
from skimage.color import rgb2gray
import numpy as np
from segmenteer.core.utils import mask_to_geojson
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
import skimage
import tempfile
import pyvips


@dataclass
class SegmentationResult:
    geojson: dict
    execution_time: float
    method_name: str
    metadata: dict


class Segmenter(Protocol):
    """Protocol defining the interface for all segmenters."""
    
    @property
    def name(self) -> str: ...

    def segment(self, path: Path) -> geojson.FeatureCollection: ...


class NumpySegmenter(ABC):
    """Base class for segmenters that work on numpy arrays."""

    APPLY_TO_GRAYSCALE: bool = True

    def __init__(
        self,
        mpp: float = 10,
        min_area: int = 10,
        to_gray_func: Union[callable, None] = rgb2gray,
        reader: WSIReader | None = None,
    ):
        self.mpp = mpp
        self.min_area = min_area
        self.to_gray_func = to_gray_func
        self.reader = reader if reader is not None else WSIReader("openslide")
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Satisfies Segmenter protocol."""
        wsi = self.reader.read(path)
        image = self.load_numpy(wsi)
        preprocessed = self._preprocess(image)
        mask = self._segment_numpy(preprocessed)
        return self._convert_to_geojson(wsi, mask)

    def load_numpy(self, wsi):
        return self.reader.get_wsi_at_mpp(wsi, (self.mpp, self.mpp))

    def _preprocess(self, image: npt.NDArray[np.int_]) -> npt.NDArray[np.uint8]:
        rgb = self._rgba_to_rgb(image)

        if self.APPLY_TO_GRAYSCALE and rgb.ndim == 3:
            preprocessed = self.to_gray_func(rgb)
        else:
            preprocessed = rgb

        return skimage.util.img_as_ubyte(preprocessed)

    def _rgba_to_rgb(self, image):
        return np.take(image, [0, 1, 2], 2)

    def _convert_to_geojson(self, wsi: OpenSlide, mask: npt.NDArray[np.bool]) -> geojson.FeatureCollection:
        scaling_factor = mask.shape[0] / self.reader.get_size(wsi, 0)[0]
        return mask_to_geojson(mask, self.min_area, scaling_factor=scaling_factor)

    @abstractmethod
    def _segment_numpy(
        self, 
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.bool_]:
        pass


class PathSegmenter(ABC):
    """Base class for segmenters that need the image file path and output a mask file path."""

    def __init__(
        self,
        min_area: int = 10,
        reader: WSIReader | None = None,
    ):
        self.min_area = min_area
        self.reader = reader if reader is not None else WSIReader("openslide")

    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Satisfies Segmenter protocol."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / "tissuemask.tif"
            self._segment_path(path, output_path)
            return self._convert_to_geojson(path, output_path)
    
    @abstractmethod
    def _segment_path(self, image_path: Path, output_path: Path) -> None:
        pass

    def _convert_to_geojson(self, output_path: Path) -> geojson.FeatureCollection:
        mask = pyvips.Image.new_from_file(output_path).numpy()
        return mask_to_geojson(mask, self.min_area)

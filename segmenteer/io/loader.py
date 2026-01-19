from pathlib import Path
from typing import Union
import numpy as np
import tifffile
import json
import geojson
from segmenteer.core.utils import scale_geojson_coordinates
from dataclasses import dataclass
import pyvips

@dataclass
class Image:
    path: Path

    def get_vips_image(self, level: int = 0):
        return pyvips.Image.tiffload(self.path, access="sequential", page=level, n=1, revalidate=True)
    
    def get_numpy_image(self, level: int = 0):
        return self.get_vips_image(level=level).numpy()
    
    @property
    def width(self):
        return self.get_vips_image().width
    
    @property
    def height(self):
        return self.get_vips_image().height

    @property
    def shape(self):
        return (self.width, self.height)

    @property
    def area(self):
        return self.shape[0] * self.shape[1]


def is_dicom_directory(path: Path) -> bool:
    if not path.is_dir():
        return False

    dicom_indicators = ["1_Map", "2", "3_0", "DICOMDIR"]

    dir_contents = [f.name for f in path.iterdir()]

    return any(indicator in dir_contents for indicator in dicom_indicators)


def load_dicom_wsi(path: Path, level: int = 0, verbose: bool = True) -> np.ndarray:
    from wsidicom import WsiDicom

    wsi = WsiDicom.open(path)

    num_levels = len(wsi.levels)

    if verbose:
        print(f"DICOM WSI Info:")
        print(f"  Available pyramid levels: {num_levels}")
        for i, lvl in enumerate(wsi.levels):
            print(f"    Level {i}: {lvl.size.width}x{lvl.size.height}")

    if level >= num_levels:
        original_level = level
        level = num_levels - 1
        if verbose:
            print(
                f"  Warning: Requested level {original_level} not available, using level {level}"
            )

    if verbose:
        print(
            f"  Loading level {level}: {wsi.levels[level].size.width}x{wsi.levels[level].size.height}"
        )

    level_size = wsi.levels[level].size
    size_tuple = (level_size.width, level_size.height)

    region = wsi.read_region((0, 0), level, size_tuple)

    image_array = np.array(region)

    wsi.close()

    return image_array


def load_openslide_wsi(path: Path, level: int = 0, verbose: bool = True) -> np.ndarray:
    try:
        import openslide
    except ImportError:
        raise ImportError(
            f"OpenSlide is required to read {path.suffix} files. "
            "Install with: pip install openslide-python openslide-bin"
        )

    slide = openslide.OpenSlide(str(path))

    num_levels = slide.level_count

    if verbose:
        print(f"OpenSlide WSI Info:")
        print(f"  Format: {slide.detect_format(str(path))}")
        print(f"  Available pyramid levels: {num_levels}")
        for i in range(num_levels):
            w, h = slide.level_dimensions[i]
            print(f"    Level {i}: {w}x{h}")

    if level >= num_levels:
        original_level = level
        level = num_levels - 1
        if verbose:
            print(
                f"  Warning: Requested level {original_level} not available, using level {level}"
            )

    if verbose:
        w, h = slide.level_dimensions[level]
        print(f"  Loading level {level}: {w}x{h}")

    level_dimensions = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, level_dimensions)
    image_array = np.array(region.convert("RGB"))

    slide.close()

    return image_array


def load_image(path: Union[str, Path], level: int = 0) -> np.ndarray:
    path = Path(path)

    if path.is_dir() and is_dicom_directory(path):
        return load_dicom_wsi(path, level=level, verbose=True)

    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        return tifffile.imread(path)
    elif suffix == ".mrxs":
        try:
            return load_openslide_wsi(path, level=level, verbose=True)
        except ImportError:
            raise ImportError(
                "OpenSlide is required for MRXS files. "
                "Install with: pip install openslide-python openslide-bin"
            )
    else:
        try:
            import openslide

            if openslide.OpenSlide.detect_format(str(path)):
                return load_openslide_wsi(path, level=level, verbose=True)
        except Exception:
            pass

        return np.array(pyvips.Image.new_from_file(path))


def save_geojson(geojson_data: dict, path: Union[str, Path], scale_factor: float = 1.0):
    path = Path(path)

    if scale_factor != 1.0:
        geojson_data = scale_geojson_coordinates(geojson_data, scale_factor)

    with open(path, "w") as f:
        json.dump(geojson_data, f, indent=2)


def load_geojson(path: Union[str, Path]) -> dict:
    path = Path(path)

    with open(path, "r") as f:
        return geojson.load(f)

from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
import json
import geojson
from segmenteer.core.utils import scale_geojson_coordinates


def is_dicom_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    
    dicom_indicators = ['1_Map', '2', '3_0', 'DICOMDIR']
    
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
            print(f"  Warning: Requested level {original_level} not available, using level {level}")
    
    if verbose:
        print(f"  Loading level {level}: {wsi.levels[level].size.width}x{wsi.levels[level].size.height}")
    
    level_size = wsi.levels[level].size
    size_tuple = (level_size.width, level_size.height)
    
    region = wsi.read_region((0, 0), level, size_tuple)
    
    image_array = np.array(region)
    
    wsi.close()
    
    return image_array


def load_image(path: str | Path, dicom_level: int = 0) -> np.ndarray:
    path = Path(path)
    
    if path.is_dir() and is_dicom_directory(path):
        return load_dicom_wsi(path, level=dicom_level, verbose=True)
    
    if path.suffix.lower() in ['.tif', '.tiff']:
        return tifffile.imread(path)
    else:
        return np.array(Image.open(path))


def save_geojson(geojson_data: dict, path: str | Path, scale_factor: float = 1.0):
    path = Path(path)
    
    if scale_factor != 1.0:
        geojson_data = scale_geojson_coordinates(geojson_data, scale_factor)
    
    with open(path, 'w') as f:
        json.dump(geojson_data, f, indent=2)


def load_geojson(path: str | Path) -> dict:
    path = Path(path)
    
    with open(path, 'r') as f:
        return geojson.load(f)
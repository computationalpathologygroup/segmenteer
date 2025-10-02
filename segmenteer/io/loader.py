from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
import json
import geojson
from segmenteer.core.utils import scale_geojson_coordinates


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    
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
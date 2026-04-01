"""Plain thumbnail creation and saving (no segmentation overlay)."""

from pathlib import Path
from typing import Union

import numpy as np
import pyvips
from PIL import Image


def create_thumbnail(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]

    scale = min(max_size / width, max_size / height)

    if scale >= 1:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)

    img_pil = Image.fromarray(image)
    img_pil_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.array(img_pil_resized)


def save_thumbnail(image: Path, output_path: Union[str, Path], max_size: int = 1024):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = pyvips.Image.thumbnail(str(image), max_size)
    thumbnail.write_to_file(str(output_path))

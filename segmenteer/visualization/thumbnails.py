"""Plain thumbnail creation and saving without pyvips."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def create_thumbnail(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize an array so that its longest edge is at most *max_size*."""
    height, width = image.shape[:2]
    scale = min(max_size / width, max_size / height)
    if scale >= 1:
        return image
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return np.asarray(
        Image.fromarray(image[..., :3]).resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
    )


def save_thumbnail(
    image: Path, output_path: Union[str, Path], max_size: int = 1024
) -> None:
    """Read only a low-resolution WSI level and save it as a PNG thumbnail."""
    from segmenteer.core.base import get_wsi_reader

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = get_wsi_reader()
    wsi = reader.read(str(image))
    width, height = reader.get_size(wsi, 0)
    native_mpp_x, native_mpp_y = reader.get_mpp(wsi, 0)
    requested_scale = max(width / max_size, height / max_size, 1.0)
    target_mpp = (native_mpp_x * requested_scale, native_mpp_y * requested_scale)
    array = reader.get_wsi_at_mpp(wsi, target_mpp)
    array = create_thumbnail(array, max_size=max_size)
    Image.fromarray(array[..., :3]).save(output_path, format="PNG")

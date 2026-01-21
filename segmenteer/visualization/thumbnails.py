from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from segmenteer.core.utils import geojson_to_mask, scale_geojson_coordinates
from PIL import Image as PILImage
import pyvips


def create_thumbnail(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]

    scale = min(max_size / width, max_size / height)

    if scale >= 1:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)

    img_pil = PILImage.fromarray(image)
    img_pil_resized = img_pil.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

    return np.array(img_pil_resized)


def create_heatmap_overlay(
    image: np.ndarray, geojson_data: dict, alpha: float = 0.4
) -> np.ndarray:
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image.copy()

    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)

    mask = geojson_to_mask(geojson_data, image.shape[:2])

    overlay = image_rgb.copy()

    green_overlay = np.zeros_like(overlay)
    green_overlay[:, :] = [0, 255, 0]

    overlay[mask] = (overlay[mask] * (1 - alpha) + green_overlay[mask] * alpha).astype(
        np.uint8
    )

    return overlay


def save_thumbnail(
    image: Path, output_path: Union[str, Path], max_size: int = 1024
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = pyvips.Image.thumbnail(image, max_size)
    thumbnail.write_to_file(output_path)


def save_heatmap_thumbnail(
    image: Image,
    geojson_data: dict,
    output_path: Union[str, Path],
    mpp: int = 0,
    max_size: int = 1024,
    alpha: float = 0.4,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: this isn't very clean, refactor?
    from monai.data.wsi_reader import WSIReader
    reader = WSIReader("openslide")
    image = reader.read(image)

    geojson_data = scale_geojson_coordinates(geojson_data, scale_factor=reader.get_mpp(image, 0)[0] / mpp)
    image =  reader.get_wsi_at_mpp(image, (mpp, mpp))[..., :3]
    overlay = create_heatmap_overlay(image, geojson_data, alpha)
    thumbnail = create_thumbnail(overlay, max_size)
    pyvips.Image.new_from_array(thumbnail).write_to_file(output_path)

from pathlib import Path
import numpy as np
from PIL import Image
from segmenteer.core.utils import geojson_to_mask


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


def save_thumbnail(image: np.ndarray, output_path: str | Path, max_size: int = 1024):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thumbnail = create_thumbnail(image, max_size)

    if thumbnail.dtype != np.uint8:
        if thumbnail.max() <= 1.0:
            thumbnail = (thumbnail * 255).astype(np.uint8)
        else:
            thumbnail = thumbnail.astype(np.uint8)

    Image.fromarray(thumbnail).save(output_path)


def save_heatmap_thumbnail(
    image: np.ndarray,
    geojson_data: dict,
    output_path: str | Path,
    max_size: int = 1024,
    alpha: float = 0.4,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = create_heatmap_overlay(image, geojson_data, alpha)
    thumbnail = create_thumbnail(overlay, max_size)

    Image.fromarray(thumbnail).save(output_path)

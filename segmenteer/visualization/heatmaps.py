from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image as PILImage


def create_heatmap_overlay(
    image: np.ndarray, geojson_data: dict, alpha: float = 0.4
) -> np.ndarray:
    from segmenteer.core.utils import geojson_to_mask

    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image[..., :3].copy()
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)
    mask = geojson_to_mask(geojson_data, image_rgb.shape[:2])
    overlay = image_rgb.copy()
    green = np.zeros_like(overlay)
    green[:, :] = [0, 255, 0]
    overlay[mask] = (overlay[mask] * (1 - alpha) + green[mask] * alpha).astype(np.uint8)
    return overlay


def save_heatmap_thumbnail(
    image: Path,
    geojson_data: dict,
    output_path: Union[str, Path],
    mpp: float = 10.0,
    max_size: int = 1024,
    alpha: float = 0.4,
) -> None:
    from segmenteer.core.base import get_wsi_reader
    from segmenteer.core.utils import scale_geojson_coordinates

    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError("Heatmap MPP must be positive.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = get_wsi_reader()
    wsi = reader.read(str(image))
    image_array = reader.get_wsi_at_mpp(wsi, (mpp, mpp))[..., :3]
    level0_width, level0_height = reader.get_size(wsi, 0)
    scale = (image_array.shape[1] / level0_width, image_array.shape[0] / level0_height)
    scaled_geojson = scale_geojson_coordinates(geojson_data, scale_factor=scale)
    overlay = create_heatmap_overlay(image_array, scaled_geojson, alpha)
    thumbnail = PILImage.fromarray(overlay).resize(
        _thumbnail_size(overlay.shape[1], overlay.shape[0], max_size),
        PILImage.Resampling.LANCZOS,
    )
    thumbnail.save(output_path, format="PNG")


def _thumbnail_size(w: int, h: int, max_size: int) -> tuple[int, int]:
    scale = min(max_size / w, max_size / h)
    return (int(w * scale), int(h * scale)) if scale < 1 else (w, h)


_EVALUATION_COLORS: dict[str, tuple[int, int, int]] = {
    "true_positive": (22, 163, 74),
    "false_positive": (220, 38, 38),
    "false_negative": (245, 158, 11),
}


def create_supervised_error_overlay(
    image: np.ndarray, evaluation_layers: dict[str, dict], alpha: float = 0.48
) -> np.ndarray:
    from segmenteer.core.utils import geojson_to_mask

    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1)
    else:
        base = image[..., :3].copy()
    if base.dtype != np.uint8:
        base = (
            (base * 255).astype(np.uint8)
            if base.max() <= 1.0
            else base.astype(np.uint8)
        )
    overlay = base.copy()
    for label, color in _EVALUATION_COLORS.items():
        layer = evaluation_layers.get(label)
        if not layer:
            continue
        mask = geojson_to_mask(layer, overlay.shape[:2])
        if not np.any(mask):
            continue
        color_array = np.asarray(color, dtype=np.float32)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha
        ).astype(np.uint8)
    return overlay


def save_supervised_error_map(
    image: Path,
    evaluation_layers: dict[str, dict],
    output_path: Union[str, Path],
    mpp: float = 10.0,
    max_size: int = 1024,
    alpha: float = 0.48,
) -> None:
    from segmenteer.core.base import get_wsi_reader
    from segmenteer.core.utils import scale_geojson_coordinates

    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError("Error-map MPP must be positive.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = get_wsi_reader()
    wsi = reader.read(str(image))
    image_array = reader.get_wsi_at_mpp(wsi, (mpp, mpp))[..., :3]
    level0_width, level0_height = reader.get_size(wsi, 0)
    factor = (image_array.shape[1] / level0_width, image_array.shape[0] / level0_height)
    scaled_layers = {
        label: scale_geojson_coordinates(layer, scale_factor=factor)
        for label, layer in evaluation_layers.items()
    }
    overlay = create_supervised_error_overlay(image_array, scaled_layers, alpha=alpha)
    thumbnail = PILImage.fromarray(overlay).resize(
        _thumbnail_size(overlay.shape[1], overlay.shape[0], max_size),
        PILImage.Resampling.LANCZOS,
    )
    thumbnail.save(output_path, format="PNG")

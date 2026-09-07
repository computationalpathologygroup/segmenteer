"""Heatmap rendering for individual predictions and ensemble vote maps.

This module uses only core dependencies for normal prediction overlays.  It
never imports MONAI or pyvips, so saving outputs cannot introduce a hidden WSI
backend dependency after segmentation has succeeded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image as PILImage


def create_heatmap_overlay(
    image: np.ndarray, geojson_data: dict, alpha: float = 0.4
) -> np.ndarray:
    """Return *image* with a green overlay drawn over the predicted tissue mask."""
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
    overlay[mask] = (
        overlay[mask] * (1 - alpha) + green[mask] * alpha
    ).astype(np.uint8)
    return overlay


def save_heatmap_thumbnail(
    image: Path,
    geojson_data: dict,
    output_path: Union[str, Path],
    mpp: float = 10.0,
    max_size: int = 1024,
    alpha: float = 0.4,
) -> None:
    """Render an image-space prediction overlay without MONAI/pyvips."""
    from segmenteer.core.base import get_wsi_reader
    from segmenteer.core.utils import scale_geojson_coordinates

    if mpp <= 0:
        raise ValueError("Heatmap MPP must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = get_wsi_reader()
    wsi = reader.read(str(image))
    native_mpp_x, _ = reader.get_mpp(wsi, 0)
    scaled_geojson = scale_geojson_coordinates(
        geojson_data, scale_factor=native_mpp_x / mpp
    )
    image_array = reader.get_wsi_at_mpp(wsi, (mpp, mpp))[..., :3]
    overlay = create_heatmap_overlay(image_array, scaled_geojson, alpha)
    thumbnail = PILImage.fromarray(overlay).resize(
        _thumbnail_size(overlay.shape[1], overlay.shape[0], max_size),
        PILImage.Resampling.LANCZOS,
    )
    thumbnail.save(output_path, format="PNG")


def _thumbnail_size(w: int, h: int, max_size: int) -> tuple[int, int]:
    scale = min(max_size / w, max_size / h)
    return (int(w * scale), int(h * scale)) if scale < 1 else (w, h)


# ---------------------------------------------------------------------------
# Ensemble vote-map renderer
# ---------------------------------------------------------------------------


def save_vote_heatmap(
    vote_ratio: np.ndarray,
    path: Path,
    bg_path: Path | None = None,
    alpha: float = 0.40,
) -> None:
    """Save a colour heatmap of the per-pixel vote fraction (0–1)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = vote_ratio.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8 * h / max(w, 1)))

    if bg_path is not None and Path(bg_path).exists():
        bg = plt.imread(str(bg_path))
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=-1)
        elif bg.shape[2] == 4:
            bg = bg[..., :3]
        ax.imshow(bg, extent=[0, w, h, 0], aspect="auto")
        overlay_alpha = alpha
    else:
        overlay_alpha = 1.0

    im = ax.imshow(
        vote_ratio,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        alpha=overlay_alpha,
    )
    plt.colorbar(im, ax=ax, label="weighted vote fraction")
    ax.set_title("Ensemble vote map")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supervised visual-QA map
# ---------------------------------------------------------------------------


_EVALUATION_COLORS: dict[str, tuple[int, int, int]] = {
    "true_positive": (22, 163, 74),
    "false_positive": (220, 38, 38),
    "false_negative": (245, 158, 11),
}


def create_supervised_error_overlay(
    image: np.ndarray,
    evaluation_layers: dict[str, dict],
    alpha: float = 0.48,
) -> np.ndarray:
    """Overlay exact TP/FP/FN regions on an RGB image.

    Green = true-positive tissue, red = false-positive tissue, and orange =
    false-negative tissue.  The layer geometry is produced in level-0 pixels
    and must be scaled to the supplied image before calling this function.
    """
    from segmenteer.core.utils import geojson_to_mask

    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1)
    else:
        base = image[..., :3].copy()

    if base.dtype != np.uint8:
        base = (base * 255).astype(np.uint8) if base.max() <= 1.0 else base.astype(np.uint8)

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
    """Save a low-resolution TP/FP/FN visual-QA map beside method outputs."""
    from segmenteer.core.base import get_wsi_reader
    from segmenteer.core.utils import scale_geojson_coordinates

    if mpp <= 0:
        raise ValueError("Error-map MPP must be positive.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = get_wsi_reader()
    wsi = reader.read(str(image))
    native_mpp_x, _ = reader.get_mpp(wsi, 0)
    factor = native_mpp_x / mpp
    scaled_layers = {
        label: scale_geojson_coordinates(layer, scale_factor=factor)
        for label, layer in evaluation_layers.items()
    }
    image_array = reader.get_wsi_at_mpp(wsi, (mpp, mpp))[..., :3]
    overlay = create_supervised_error_overlay(image_array, scaled_layers, alpha=alpha)
    thumbnail = PILImage.fromarray(overlay).resize(
        _thumbnail_size(overlay.shape[1], overlay.shape[0], max_size),
        PILImage.Resampling.LANCZOS,
    )
    thumbnail.save(output_path, format="PNG")

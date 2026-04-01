"""Heatmap rendering for individual method predictions and ensemble vote maps."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pyvips
from PIL import Image as _PILImage


def create_heatmap_overlay(
    image: np.ndarray, geojson_data: dict, alpha: float = 0.4
) -> np.ndarray:
    """Return *image* with a green overlay drawn over the predicted tissue mask.

    Parameters
    ----------
    image:
        H×W or H×W×3 uint8 / float array.
    geojson_data:
        GeoJSON FeatureCollection with predicted polygons.
    alpha:
        Overlay opacity (0 = invisible, 1 = fully green).
    """
    from segmenteer.core.utils import geojson_to_mask

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
    green = np.zeros_like(overlay)
    green[:, :] = [0, 255, 0]
    overlay[mask] = (overlay[mask] * (1 - alpha) + green[mask] * alpha).astype(np.uint8)
    return overlay


def save_heatmap_thumbnail(
    image: Path,
    geojson_data: dict,
    output_path: Union[str, Path],
    mpp: int = 0,
    max_size: int = 1024,
    alpha: float = 0.4,
) -> None:
    """Render prediction polygons as a coloured overlay and save to *output_path*.

    Parameters
    ----------
    image:
        Path to the source WSI file.
    geojson_data:
        GeoJSON FeatureCollection with predicted polygons.
    output_path:
        Destination PNG path.
    mpp:
        Microns-per-pixel resolution at which to read the WSI.
    max_size:
        Longest edge (pixels) of the saved thumbnail.
    alpha:
        Overlay opacity.
    """
    from monai.data.wsi_reader import WSIReader

    from segmenteer.core.base import WSI_READER
    from segmenteer.core.utils import scale_geojson_coordinates

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = WSIReader(WSI_READER)
    wsi = reader.read(str(image))
    geojson_data = scale_geojson_coordinates(
        geojson_data, scale_factor=reader.get_mpp(wsi, 0)[0] / mpp
    )
    img_arr = reader.get_wsi_at_mpp(wsi, (mpp, mpp))[..., :3]
    overlay = create_heatmap_overlay(img_arr, geojson_data, alpha)
    thumbnail = _PILImage.fromarray(overlay).resize(
        _thumbnail_size(overlay.shape[1], overlay.shape[0], max_size),
        _PILImage.Resampling.LANCZOS,
    )
    pyvips.Image.new_from_array(np.array(thumbnail)).write_to_file(str(output_path))


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
    """Save a colour heatmap of the per-pixel vote fraction (0–1).

    Parameters
    ----------
    vote_ratio:
        2-D float array with values in [0, 1] representing the weighted
        fraction of ensemble members that voted *tissue* for each pixel.
    path:
        Destination PNG path. Parent directory is created if needed.
    bg_path:
        Optional path to a background image (e.g. a WSI thumbnail).  When
        provided it is displayed in grayscale so the RdYlGn overlay remains
        readable.
    alpha:
        Opacity of the vote-map overlay when *bg_path* is given.  ``1.0``
        means fully opaque (background not visible).
    """
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

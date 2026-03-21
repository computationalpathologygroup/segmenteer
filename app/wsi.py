"""DZI-compatible tile server.

Uses openslide as the primary backend (handles all pyramid TIFF formats and
gives access to level metadata).  Falls back to tifffile for formats openslide
cannot open.

DZI protocol
------------
  GET /api/wsi/{stem}/dzi          → XML descriptor
  GET /api/wsi/{stem}/{level}/{x}_{y}.jpg  → JPEG tile
"""

from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image

TILE_SIZE = 256


class WSITileServer:
    """Serve Deep Zoom tiles for a single WSI file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._slide = None
        self._tif = None
        self.mpp: float | None = None

        try:
            import openslide

            self._slide = openslide.OpenSlide(str(self.path))
            self.width, self.height = self._slide.dimensions
            raw_mpp = self._slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            self.mpp = float(raw_mpp) if raw_mpp else None
        except Exception:
            import tifffile

            self._tif = tifffile.TiffFile(str(self.path))
            page = self._tif.series[0].levels[0].pages[0]
            self.width = page.imagewidth
            self.height = page.imagelength

        self.n_levels = math.ceil(math.log2(max(self.width, self.height))) + 1

    # ------------------------------------------------------------------
    # DZI descriptor
    # ------------------------------------------------------------------

    def dzi_descriptor(self) -> str:
        return (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"'
            f' Format="jpeg" Overlap="0" TileSize="{TILE_SIZE}">'
            f'<Size Width="{self.width}" Height="{self.height}"/>'
            "</Image>"
        )

    # ------------------------------------------------------------------
    # Tile rendering
    # ------------------------------------------------------------------

    def get_tile(self, dzi_level: int, x: int, y: int) -> bytes:
        """Return JPEG bytes for tile (x, y) at DZI level *dzi_level*."""
        scale = 2 ** (self.n_levels - 1 - dzi_level)
        lw = max(1, math.ceil(self.width / scale))
        lh = max(1, math.ceil(self.height / scale))

        x0, y0 = x * TILE_SIZE, y * TILE_SIZE
        tw = min(TILE_SIZE, lw - x0)
        th = min(TILE_SIZE, lh - y0)
        if tw <= 0 or th <= 0:
            return _blank_jpeg()

        if self._slide is not None:
            img = self._tile_openslide(scale, x0, y0, tw, th)
        else:
            img = self._tile_tifffile(scale, x0, y0, tw, th)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _tile_openslide(
        self, scale: float, x0: int, y0: int, tw: int, th: int
    ) -> Image.Image:
        best = self._slide.get_best_level_for_downsample(scale)
        best_ds = self._slide.level_downsamples[best]

        l0_x = int(x0 * scale)
        l0_y = int(y0 * scale)
        read_w = max(1, math.ceil(tw * scale / best_ds))
        read_h = max(1, math.ceil(th * scale / best_ds))

        region = self._slide.read_region((l0_x, l0_y), best, (read_w, read_h))
        img = region.convert("RGB")
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
        return img

    def _tile_tifffile(
        self, scale: float, x0: int, y0: int, tw: int, th: int
    ) -> Image.Image:
        levels = self._tif.series[0].levels
        best_idx, best_ds = 0, 1.0
        for i, lvl in enumerate(levels):
            ds = self.width / lvl.pages[0].imagewidth
            if ds <= scale:
                best_idx, best_ds = i, ds

        lvl_w = levels[best_idx].pages[0].imagewidth
        lvl_h = levels[best_idx].pages[0].imagelength
        lx0 = min(int(x0 * scale / best_ds), lvl_w)
        ly0 = min(int(y0 * scale / best_ds), lvl_h)
        lx1 = min(lx0 + math.ceil(tw * scale / best_ds) + 1, lvl_w)
        ly1 = min(ly0 + math.ceil(th * scale / best_ds) + 1, lvl_h)

        arr = levels[best_idx].asarray()[ly0:ly1, lx0:lx1]
        img = Image.fromarray(
            arr[..., :3] if arr.ndim == 3 and arr.shape[2] >= 3 else arr
        )
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
        return img

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
        if self._tif is not None:
            self._tif.close()


def _blank_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (248, 250, 252)).save(buf, format="JPEG")
    return buf.getvalue()

from __future__ import annotations

from io import BytesIO
from pathlib import Path


class WSITileServer:
    """Serve a WSI as Deep Zoom tiles for the standalone viewer.

    The viewer intentionally keeps WSI reading separate from runner/evaluator
    logic. OpenSlide is used for vendor WSI formats and pyramidal TIFFs because
    it can read individual regions without decoding the entire slide.
    """

    TILE_SIZE = 256
    OVERLAP = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"WSI file does not exist: {self.path}")

        try:
            import openslide
            from openslide.deepzoom import DeepZoomGenerator
        except ImportError as exc:
            raise RuntimeError(
                "OpenSlide viewer support is not installed. Install the runtime extra with "
                "`uv sync --extra runtime` (or install openslide-python and openslide-bin)."
            ) from exc

        try:
            self._slide = openslide.OpenSlide(str(self.path))
        except Exception as exc:  # openslide raises backend-specific exceptions
            raise RuntimeError(f"OpenSlide could not open {self.path.name}: {exc}") from exc

        self._deepzoom = DeepZoomGenerator(
            self._slide,
            tile_size=self.TILE_SIZE,
            overlap=self.OVERLAP,
            limit_bounds=False,
        )
        self.width, self.height = (int(v) for v in self._slide.dimensions)
        self.n_levels = int(self._deepzoom.level_count)
        self.backend = "openslide"
        self.last_error: str | None = None
        self.mpp = self._read_mpp(openslide)

    def _read_mpp(self, openslide_module) -> float | None:
        values: list[float] = []
        for key in (
            openslide_module.PROPERTY_NAME_MPP_X,
            openslide_module.PROPERTY_NAME_MPP_Y,
        ):
            raw = self._slide.properties.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        if not values:
            return None
        return float(sum(values) / len(values))

    def dzi_descriptor(self) -> str:
        return self._deepzoom.get_dzi("jpeg")

    def get_thumbnail(self, max_size: int = 1024) -> bytes:
        """Return a compact JPEG thumbnail for self-contained HTML reports."""
        max_size = max(128, min(int(max_size), 2048))
        try:
            image = self._slide.get_thumbnail((max_size, max_size)).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
            self.last_error = None
            return buffer.getvalue()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def get_tile(self, level: int, x: int, y: int) -> bytes:
        if level < 0 or level >= self._deepzoom.level_count:
            raise ValueError(f"Deep Zoom level out of range: {level}")
        columns, rows = self._deepzoom.level_tiles[level]
        if x < 0 or y < 0 or x >= columns or y >= rows:
            raise ValueError(
                f"Tile ({x}, {y}) is outside Deep Zoom level {level} "
                f"({columns} x {rows} tiles)"
            )
        try:
            tile = self._deepzoom.get_tile(level, (x, y)).convert("RGB")
            buffer = BytesIO()
            tile.save(buffer, format="JPEG", quality=90)
            self.last_error = None
            return buffer.getvalue()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def close(self) -> None:
        slide = getattr(self, "_slide", None)
        if slide is not None:
            slide.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

"""Minimal, lazy whole-slide image reader used by classical segmenteer methods.

The reader deliberately avoids MONAI.  It uses OpenSlide when selected and
falls back to tifffile for TIFF slides.  Optional packages are imported only
when an image is actually opened, never when :mod:`segmenteer` is imported.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class WSIHandle:
    """Lightweight reference to a slide; no native file handle is retained."""

    path: Path
    backend: str


class NativeWSIReader:
    """Read TIFF/WSI images through OpenSlide or tifffile.

    Interface compatibility is intentionally limited to the methods used by
    segmenteer's classical segmenters and reporting pipeline:
    ``read``, ``get_size``, ``get_mpp`` and ``get_wsi_at_mpp``.  The ``auto`` backend tries OpenSlide first and resolves to tifffile only when needed.
    """

    VALID_BACKENDS = {"auto", "openslide", "tifffile"}

    def __init__(self, backend: str = "auto", fallback_mpp: float | None = None):
        backend = str(backend).casefold().strip()
        if backend not in self.VALID_BACKENDS:
            valid = ", ".join(sorted(self.VALID_BACKENDS))
            raise ValueError(f"Unsupported WSI_READER={backend!r}. Use one of: {valid}.")
        self.backend = backend
        self.fallback_mpp = fallback_mpp

    def read(self, path: str | Path) -> WSIHandle:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.backend != "auto":
            return WSIHandle(path=path, backend=self.backend)

        # Resolve once per WSI so all subsequent calls use exactly one backend.
        # OpenSlide is safer for pyramidal WSI formats; tifffile remains the
        # fallback for ordinary TIFFs and environments without OpenSlide.
        try:
            slide = self._open_openslide(path)
            slide.close()
            return WSIHandle(path=path, backend="openslide")
        except Exception:
            with self._open_tiff(path) as tif:
                self._tiff_levels(tif)
            return WSIHandle(path=path, backend="tifffile")

    def get_size(self, wsi: WSIHandle, level: int = 0) -> tuple[int, int]:
        if wsi.backend == "openslide":
            slide = self._open_openslide(wsi.path)
            try:
                if not 0 <= level < slide.level_count:
                    raise IndexError(f"Level {level} is not available in {wsi.path.name}.")
                width, height = slide.level_dimensions[level]
                return int(width), int(height)
            finally:
                slide.close()

        with self._open_tiff(wsi.path) as tif:
            levels = self._tiff_levels(tif)
            if not 0 <= level < len(levels):
                raise IndexError(f"Level {level} is not available in {wsi.path.name}.")
            return self._level_size(levels[level])

    def get_mpp(self, wsi: WSIHandle, level: int = 0) -> tuple[float, float]:
        """Return the physical pixel size for *level* in micrometres/pixel."""
        if wsi.backend == "openslide":
            slide = self._open_openslide(wsi.path)
            try:
                native_x, native_y = self._openslide_native_mpp(slide, wsi.path)
                if not 0 <= level < slide.level_count:
                    raise IndexError(f"Level {level} is not available in {wsi.path.name}.")
                downsample = float(slide.level_downsamples[level])
                return native_x * downsample, native_y * downsample
            finally:
                slide.close()

        with self._open_tiff(wsi.path) as tif:
            levels = self._tiff_levels(tif)
            if not 0 <= level < len(levels):
                raise IndexError(f"Level {level} is not available in {wsi.path.name}.")
            native_x, native_y = self._tiff_native_mpp(tif, wsi.path)
            width0, height0 = self._level_size(levels[0])
            width, height = self._level_size(levels[level])
            return native_x * width0 / width, native_y * height0 / height

    def get_wsi_at_mpp(
        self, wsi: WSIHandle, mpp: tuple[float, float] | list[float] | float
    ) -> np.ndarray:
        """Read a full-slide RGB array close to the requested MPP.

        A suitable native pyramid level is selected first and is only further
        downsampled.  Therefore a requested coarse resolution does not force a
        level-0 WSI into memory when a pyramid is available.
        """
        target_x, target_y = self._normalise_target_mpp(mpp)
        if wsi.backend == "openslide":
            return self._read_openslide_at_mpp(wsi.path, target_x, target_y)
        return self._read_tiff_at_mpp(wsi.path, target_x, target_y)

    # ------------------------------------------------------------------
    # OpenSlide backend
    # ------------------------------------------------------------------

    def _open_openslide(self, path: Path):
        try:
            import openslide
        except ImportError as exc:
            raise ImportError(
                "OpenSlide is selected but not installed. Run:\n"
                "  uv sync --extra openslide\n"
                "or set WSI_READER = 'tifffile' in run.py."
            ) from exc
        try:
            return openslide.OpenSlide(str(path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"OpenSlide could not open {path.name}. "
                "Set WSI_READER = 'tifffile' for ordinary TIFF files."
            ) from exc

    def _openslide_native_mpp(self, slide: Any, path: Path) -> tuple[float, float]:
        props = slide.properties
        x = props.get("openslide.mpp-x") or props.get("aperio.MPP")
        y = props.get("openslide.mpp-y") or x
        if x is None:
            return self._fallback_mpp_or_raise(path)
        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return self._fallback_mpp_or_raise(path)

    def _read_openslide_at_mpp(
        self, path: Path, target_x: float, target_y: float
    ) -> np.ndarray:
        slide = self._open_openslide(path)
        try:
            native_x, native_y = self._openslide_native_mpp(slide, path)
            requested_downsample = max(target_x / native_x, target_y / native_y)
            level = int(slide.get_best_level_for_downsample(requested_downsample))
            current_downsample = float(slide.level_downsamples[level])
            width, height = slide.level_dimensions[level]
            image = np.asarray(slide.read_region((0, 0), level, (width, height)).convert("RGB"))
            current_x = native_x * current_downsample
            current_y = native_y * current_downsample
        finally:
            slide.close()

        return self._resample_from_current_mpp(
            image, current_x, current_y, target_x, target_y
        )

    # ------------------------------------------------------------------
    # tifffile backend
    # ------------------------------------------------------------------

    def _open_tiff(self, path: Path):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "tifffile is required for TIFF WSI reading. Run:\n"
                "  uv sync\n"
                "(tifffile is a core segmenteer dependency in this release)."
            ) from exc
        # Some valid-but-unusual WSI TIFFs carry a non-fatal shaped-series
        # warning.  The dimensions used below come from the primary page.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*shaped series shape does not match page shape.*",
                category=UserWarning,
            )
            return tifffile.TiffFile(str(path))

    @staticmethod
    def _tiff_levels(tif: Any) -> list[Any]:
        if not tif.series:
            raise ValueError("TIFF contains no image series.")
        levels = list(tif.series[0].levels)
        if not levels:
            raise ValueError("TIFF contains no readable image levels.")
        return levels

    @staticmethod
    def _level_size(level: Any) -> tuple[int, int]:
        page = level.pages[0]
        return int(page.imagewidth), int(page.imagelength)

    def _tiff_native_mpp(self, tif: Any, path: Path) -> tuple[float, float]:
        page = tif.series[0].levels[0].pages[0]
        description = page.description or ""

        # OME-TIFF physical sizes, in micrometres unless another unit is given.
        ome = getattr(tif, "ome_metadata", None) or description
        x_match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
        y_match = re.search(r'PhysicalSizeY="([0-9.eE+-]+)"', ome)
        if x_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1)) if y_match else x
            unit_match = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
            factor = self._unit_to_micrometres(unit_match.group(1)) if unit_match else 1.0
            return x * factor, y * factor

        # Aperio/ASAP-style text metadata sometimes records MPP directly.
        mpp_match = re.search(r"(?:^|[|;\s])MPP\s*=\s*([0-9.eE+-]+)", description, re.I)
        if mpp_match:
            value = float(mpp_match.group(1))
            return value, value

        xres = self._tag_ratio(page, "XResolution")
        yres = self._tag_ratio(page, "YResolution")
        unit = self._tag_value(page, "ResolutionUnit")
        unit_name = getattr(unit, "name", str(unit)).casefold()
        if xres and yres:
            if "inch" in unit_name or str(unit) == "2":
                return 25400.0 / xres, 25400.0 / yres
            if "centimeter" in unit_name or "centimet" in unit_name or str(unit) == "3":
                return 10000.0 / xres, 10000.0 / yres

        return self._fallback_mpp_or_raise(path)

    @staticmethod
    def _tag_ratio(page: Any, name: str) -> float | None:
        tag = page.tags.get(name)
        if tag is None:
            return None
        value = tag.value
        try:
            if isinstance(value, tuple) and len(value) == 2:
                numerator, denominator = value
                return float(numerator) / float(denominator)
            return float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def _tag_value(page: Any, name: str) -> Any:
        tag = page.tags.get(name)
        return tag.value if tag is not None else None

    def _read_tiff_at_mpp(self, path: Path, target_x: float, target_y: float) -> np.ndarray:
        with self._open_tiff(path) as tif:
            levels = self._tiff_levels(tif)
            native_x, native_y = self._tiff_native_mpp(tif, path)
            requested_downsample = max(target_x / native_x, target_y / native_y)
            level_idx = self._choose_tiff_level(levels, requested_downsample)
            level = levels[level_idx]
            width0, height0 = self._level_size(levels[0])
            width, height = self._level_size(level)
            current_x = native_x * width0 / width
            current_y = native_y * height0 / height
            array = self._normalise_tiff_array(level.asarray(), getattr(level, "axes", ""))

        return self._resample_from_current_mpp(
            array, current_x, current_y, target_x, target_y
        )

    def _choose_tiff_level(self, levels: list[Any], requested_downsample: float) -> int:
        width0, _ = self._level_size(levels[0])
        candidates: list[tuple[float, int]] = []
        for idx, level in enumerate(levels):
            width, _ = self._level_size(level)
            downsample = width0 / width
            if downsample <= requested_downsample * 1.000001:
                candidates.append((downsample, idx))
        if candidates:
            return max(candidates)[1]
        return 0

    # ------------------------------------------------------------------
    # Array and scaling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_target_mpp(
        mpp: tuple[float, float] | list[float] | float,
    ) -> tuple[float, float]:
        if isinstance(mpp, (int, float)):
            x = y = float(mpp)
        else:
            if len(mpp) != 2:
                raise ValueError("MPP must be a scalar or a two-item (x_mpp, y_mpp) tuple.")
            x, y = float(mpp[0]), float(mpp[1])
        if x <= 0 or y <= 0:
            raise ValueError(f"MPP must be positive, got ({x}, {y}).")
        return x, y

    @staticmethod
    def _normalise_tiff_array(array: np.ndarray, axes: str) -> np.ndarray:
        array = np.asarray(array)
        axes = (axes or "").upper()

        # Move Y and X to the front and an optional sample/channel axis last.
        if axes and len(axes) == array.ndim and "Y" in axes and "X" in axes:
            y_axis, x_axis = axes.index("Y"), axes.index("X")
            channel_axis = next((axes.index(a) for a in ("S", "C") if a in axes), None)
            keep = {y_axis, x_axis}
            if channel_axis is not None:
                keep.add(channel_axis)
            index = tuple(slice(None) if i in keep else 0 for i in range(array.ndim))
            array = array[index]
            remaining_axes = [axis for i, axis in enumerate(axes) if i in keep]
            y_axis, x_axis = remaining_axes.index("Y"), remaining_axes.index("X")
            channel_axis = next(
                (remaining_axes.index(a) for a in ("S", "C") if a in remaining_axes),
                None,
            )
            order = [y_axis, x_axis] + ([] if channel_axis is None else [channel_axis])
            array = np.moveaxis(array, order, range(len(order)))

        while array.ndim > 3:
            array = array[0]

        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        elif array.ndim == 3:
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            elif array.shape[2] >= 3:
                array = array[..., :3]
            elif array.shape[0] in (1, 3, 4):
                array = np.moveaxis(array, 0, -1)[..., :3]
            else:
                raise ValueError(f"Cannot interpret TIFF array shape {array.shape} as RGB.")
        else:
            raise ValueError(f"Cannot interpret TIFF array shape {array.shape} as an image.")

        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            elif np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize > 1:
                max_value = np.iinfo(array.dtype).max
                array = (array.astype(np.float32) * (255.0 / max_value)).clip(0, 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _resample_from_current_mpp(
        image: np.ndarray,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
    ) -> np.ndarray:
        scale_x = current_x / target_x
        scale_y = current_y / target_y
        new_width = max(1, round(image.shape[1] * scale_x))
        new_height = max(1, round(image.shape[0] * scale_y))
        if (new_width, new_height) == (image.shape[1], image.shape[0]):
            return image
        return np.asarray(
            Image.fromarray(image).resize((new_width, new_height), Image.Resampling.LANCZOS)
        )

    def _fallback_mpp_or_raise(self, path: Path) -> tuple[float, float]:
        raw = self.fallback_mpp
        if raw is None:
            env = os.environ.get("WSI_NATIVE_MPP", "").strip()
            raw = float(env) if env else None
        if raw is None:
            raise ValueError(
                f"Cannot determine level-0 MPP from {path.name}. "
                "Set WSI_NATIVE_MPP in run.py (for example 0.25 or 0.5), "
                "or use an OpenSlide-readable slide with MPP metadata."
            )
        raw = float(raw)
        if not math.isfinite(raw) or raw <= 0:
            raise ValueError(f"WSI_NATIVE_MPP must be a positive finite number, got {raw!r}.")
        return raw, raw

    @staticmethod
    def _unit_to_micrometres(unit: str) -> float:
        normalized = unit.casefold().strip()
        if normalized in {"µm", "um", "micrometer", "micrometre"}:
            return 1.0
        if normalized in {"mm", "millimeter", "millimetre"}:
            return 1000.0
        if normalized in {"cm", "centimeter", "centimetre"}:
            return 10000.0
        if normalized in {"m", "meter", "metre"}:
            return 1_000_000.0
        # OME defaults to micrometres, and an unrecognised unit should not
        # silently change the scale.
        return 1.0

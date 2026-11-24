from __future__ import annotations
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class WSIHandle:
    path: Path
    backend: str


class NativeWSIReader:
    VALID_BACKENDS = {"auto", "openslide", "tifffile"}

    def __init__(
        self,
        backend: str = "auto",
        fallback_mpp: float | tuple[float, float] | None = None,
    ):
        backend = str(backend).casefold().strip()
        if backend not in self.VALID_BACKENDS:
            valid = ", ".join(sorted(self.VALID_BACKENDS))
            raise ValueError(
                f"Unsupported WSI_READER={backend!r}. Use one of: {valid}."
            )
        self.backend = backend
        self.fallback_mpp = fallback_mpp

    def read(self, path: str | Path) -> WSIHandle:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.backend != "auto":
            return WSIHandle(path=path, backend=self.backend)
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
                    raise IndexError(
                        f"Level {level} is not available in {wsi.path.name}."
                    )
                width, height = slide.level_dimensions[level]
                return (int(width), int(height))
            finally:
                slide.close()
        with self._open_tiff(wsi.path) as tif:
            levels = self._tiff_levels(tif)
            if not 0 <= level < len(levels):
                raise IndexError(f"Level {level} is not available in {wsi.path.name}.")
            return self._level_size(levels[level])

    def get_mpp(self, wsi: WSIHandle, level: int = 0) -> tuple[float, float]:
        if wsi.backend == "openslide":
            slide = self._open_openslide(wsi.path)
            try:
                if not 0 <= level < slide.level_count:
                    raise IndexError(
                        f"Level {level} is not available in {wsi.path.name}."
                    )
                native_x, native_y = self._openslide_native_mpp(slide, wsi.path)
                width0, height0 = slide.level_dimensions[0]
                width, height = slide.level_dimensions[level]
                return (native_x * width0 / width, native_y * height0 / height)
            finally:
                slide.close()
        with self._open_tiff(wsi.path) as tif:
            levels = self._tiff_levels(tif)
            if not 0 <= level < len(levels):
                raise IndexError(f"Level {level} is not available in {wsi.path.name}.")
            native_x, native_y = self._tiff_native_mpp(tif, wsi.path)
            width0, height0 = self._level_size(levels[0])
            width, height = self._level_size(levels[level])
            return (native_x * width0 / width, native_y * height0 / height)

    def get_size_and_mpp(
        self, wsi: WSIHandle, *, read_mpp: bool = True
    ) -> tuple[tuple[int, int], tuple[float, float] | None]:
        if wsi.backend == "openslide":
            slide = self._open_openslide(wsi.path)
            try:
                size = tuple(int(value) for value in slide.level_dimensions[0])
                spacing = (
                    self._openslide_native_mpp(slide, wsi.path) if read_mpp else None
                )
                return size, spacing
            finally:
                slide.close()
        with self._open_tiff(wsi.path) as tif:
            levels = self._tiff_levels(tif)
            return self._level_size(levels[0]), self._tiff_native_mpp(
                tif, wsi.path
            ) if read_mpp else None

    def get_effective_mpp(
        self, wsi: WSIHandle, image_shape: tuple[int, ...]
    ) -> tuple[float, float]:
        if len(image_shape) < 2 or image_shape[0] < 1 or image_shape[1] < 1:
            raise ValueError("Image shape must contain positive height and width.")
        width0, height0 = self.get_size(wsi, 0)
        native_x, native_y = self.get_mpp(wsi, 0)
        return (native_x * width0 / image_shape[1], native_y * height0 / image_shape[0])

    def get_wsi_at_mpp(
        self, wsi: WSIHandle, mpp: tuple[float, float] | list[float] | float
    ) -> np.ndarray:
        target_x, target_y = self._normalise_target_mpp(mpp)
        if wsi.backend == "openslide":
            return self._read_openslide_at_mpp(wsi.path, target_x, target_y)
        return self._read_tiff_at_mpp(wsi.path, target_x, target_y)

    def _open_openslide(self, path: Path):
        try:
            import openslide
        except ImportError as exc:
            raise ImportError(
                "OpenSlide is selected but not installed. Run:\n  uv sync --extra openslide\nor set WSI_READER = 'tifffile' in run.py."
            ) from exc
        try:
            return openslide.OpenSlide(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"OpenSlide could not open {path.name}. Set WSI_READER = 'tifffile' for ordinary TIFF files."
            ) from exc

    def _openslide_native_mpp(self, slide: Any, path: Path) -> tuple[float, float]:
        props = slide.properties
        aperio = props.get("aperio.MPP")
        x = props.get("openslide.mpp-x", aperio)
        y = props.get("openslide.mpp-y", aperio)
        if x is None or y is None:
            if path.suffix.casefold() in {
                ".tif",
                ".tiff",
                ".svs",
                ".ndpi",
                ".scn",
                ".qptiff",
            }:
                try:
                    with self._open_tiff(path) as tif:
                        return self._tiff_native_mpp(tif, path)
                except (OSError, ImportError):
                    pass
                except ValueError as exc:
                    if "Cannot determine level-0 MPP" not in str(exc):
                        raise
            return self._fallback_mpp_or_raise(path)
        try:
            return (
                self._positive_mpp(x, "OpenSlide MPP X"),
                self._positive_mpp(y, "OpenSlide MPP Y"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid OpenSlide MPP metadata for {path.name}."
            ) from exc

    @staticmethod
    def _target_size(
        width: int,
        height: int,
        native_x: float,
        native_y: float,
        target_x: float,
        target_y: float,
    ) -> tuple[int, int]:
        output_x = width * native_x / target_x
        output_y = height * native_y / target_y
        if (
            not math.isfinite(output_x)
            or not math.isfinite(output_y)
            or output_x < 0.5
            or output_y < 0.5
        ):
            raise ValueError(
                "Requested MPP would produce a zero-sized or non-finite image."
            )
        if output_x > 2**31 - 1 or output_y > 2**31 - 1:
            raise ValueError("Requested MPP would produce an excessively large image.")
        return (max(1, round(output_x)), max(1, round(output_y)))

    @staticmethod
    def _choose_level(
        level_sizes: list[tuple[int, int]], target_size: tuple[int, int]
    ) -> int:
        target_width, target_height = target_size
        eligible = [
            (width * height, idx)
            for idx, (width, height) in enumerate(level_sizes)
            if width >= target_width and height >= target_height
        ]
        if not eligible:
            return 0
        return min(eligible)[1]

    def _read_openslide_at_mpp(
        self, path: Path, target_x: float, target_y: float
    ) -> np.ndarray:
        slide = self._open_openslide(path)
        try:
            native_x, native_y = self._openslide_native_mpp(slide, path)
            width0, height0 = slide.level_dimensions[0]
            requested_size = self._target_size(
                width0, height0, native_x, native_y, target_x, target_y
            )
            level = self._choose_level(list(slide.level_dimensions), requested_size)
            width, height = slide.level_dimensions[level]
            image = np.asarray(
                slide.read_region((0, 0), level, (width, height)).convert("RGB")
            )
        finally:
            slide.close()
        return self._resample_to_size(image, requested_size)

    def _open_tiff(self, path: Path):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "tifffile is required for TIFF WSI reading. Run:\n  uv sync\n(tifffile is a core segmenteer dependency in this release)."
            ) from exc
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
        return (int(page.imagewidth), int(page.imagelength))

    def _tiff_native_mpp(self, tif: Any, path: Path) -> tuple[float, float]:
        page = tif.series[0].levels[0].pages[0]
        description = page.description or ""
        ome = getattr(tif, "ome_metadata", None)
        if ome:
            try:
                root = ET.fromstring(ome)
            except ET.ParseError as exc:
                raise ValueError(f"Invalid OME XML metadata in {path.name}.") from exc
            pixels = next(
                (item for item in root.iter() if item.tag.split("}")[-1] == "Pixels"),
                None,
            )
            if pixels is not None:
                x_raw = pixels.get("PhysicalSizeX")
                y_raw = pixels.get("PhysicalSizeY")
                if x_raw is not None or y_raw is not None:
                    if x_raw is None or y_raw is None:
                        raise ValueError(
                            f"Incomplete OME PhysicalSizeX/PhysicalSizeY in {path.name}."
                        )
                    x_unit = pixels.get("PhysicalSizeXUnit", "µm")
                    y_unit = pixels.get("PhysicalSizeYUnit", "µm")
                    return (
                        self._positive_mpp(
                            float(x_raw) * self._unit_to_micrometres(x_unit),
                            "OME MPP X",
                        ),
                        self._positive_mpp(
                            float(y_raw) * self._unit_to_micrometres(y_unit),
                            "OME MPP Y",
                        ),
                    )
        mpp_match = re.search(r"(?:^|[|;\s])MPP\s*=\s*([0-9.eE+-]+)", description, re.I)
        if mpp_match:
            value = self._positive_mpp(float(mpp_match.group(1)), "TIFF MPP")
            return (value, value)
        xres = self._tag_ratio(page, "XResolution")
        yres = self._tag_ratio(page, "YResolution")
        unit = self._tag_value(page, "ResolutionUnit")
        unit_name = getattr(unit, "name", str(unit)).casefold()
        if xres is not None and yres is not None:
            if not all(math.isfinite(v) and v > 0 for v in (xres, yres)):
                raise ValueError(f"Invalid TIFF resolution tags in {path.name}.")
            if "inch" in unit_name or str(unit) == "2":
                return (
                    self._positive_mpp(25400.0 / xres, "TIFF MPP X"),
                    self._positive_mpp(25400.0 / yres, "TIFF MPP Y"),
                )
            if "centimeter" in unit_name or "centimet" in unit_name or str(unit) == "3":
                return (
                    self._positive_mpp(10000.0 / xres, "TIFF MPP X"),
                    self._positive_mpp(10000.0 / yres, "TIFF MPP Y"),
                )
        return self._fallback_mpp_or_raise(path)

    @staticmethod
    def _positive_mpp(value: Any, label: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(
                f"{label} must be a positive finite number, got {value!r}."
            )
        return numeric

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

    def _read_tiff_at_mpp(
        self, path: Path, target_x: float, target_y: float
    ) -> np.ndarray:
        with self._open_tiff(path) as tif:
            levels = self._tiff_levels(tif)
            native_x, native_y = self._tiff_native_mpp(tif, path)
            width0, height0 = self._level_size(levels[0])
            requested_size = self._target_size(
                width0, height0, native_x, native_y, target_x, target_y
            )
            level_idx = self._choose_level(
                [self._level_size(level) for level in levels], requested_size
            )
            level = levels[level_idx]
            array = self._normalise_tiff_array(
                level.asarray(), getattr(level, "axes", "")
            )
        return self._resample_to_size(array, requested_size)

    @staticmethod
    def _normalise_target_mpp(
        mpp: tuple[float, float] | list[float] | float,
    ) -> tuple[float, float]:
        if isinstance(mpp, (int, float, np.number)) and not isinstance(
            mpp, (bool, np.bool_)
        ):
            x = y = float(mpp)
        else:
            if (
                isinstance(mpp, (str, bytes))
                or not hasattr(mpp, "__len__")
                or len(mpp) != 2
            ):
                raise ValueError(
                    "MPP must be a scalar or a two-item (x_mpp, y_mpp) tuple."
                )
            x, y = (float(mpp[0]), float(mpp[1]))
        return (
            NativeWSIReader._positive_mpp(x, "Target MPP X"),
            NativeWSIReader._positive_mpp(y, "Target MPP Y"),
        )

    @staticmethod
    def _normalise_tiff_array(array: np.ndarray, axes: str) -> np.ndarray:
        array = np.asarray(array)
        axes = (axes or "").upper()
        if axes and len(axes) == array.ndim and ("Y" in axes) and ("X" in axes):
            y_axis, x_axis = (axes.index("Y"), axes.index("X"))
            channel_axis = next((axes.index(a) for a in ("S", "C") if a in axes), None)
            keep = {y_axis, x_axis}
            if channel_axis is not None:
                keep.add(channel_axis)
            index = tuple((slice(None) if i in keep else 0 for i in range(array.ndim)))
            array = array[index]
            remaining_axes = [axis for i, axis in enumerate(axes) if i in keep]
            y_axis, x_axis = (remaining_axes.index("Y"), remaining_axes.index("X"))
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
                raise ValueError(
                    f"Cannot interpret TIFF array shape {array.shape} as RGB."
                )
        else:
            raise ValueError(
                f"Cannot interpret TIFF array shape {array.shape} as an image."
            )
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            elif np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize > 1:
                max_value = np.iinfo(array.dtype).max
                array = (
                    (array.astype(np.float32) * (255.0 / max_value))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _resample_to_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if size == (image.shape[1], image.shape[0]):
            return image
        return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))

    @staticmethod
    def _resample_from_current_mpp(
        image: np.ndarray,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
    ) -> np.ndarray:
        current_x = NativeWSIReader._positive_mpp(current_x, "Current MPP X")
        current_y = NativeWSIReader._positive_mpp(current_y, "Current MPP Y")
        target_x, target_y = NativeWSIReader._normalise_target_mpp((target_x, target_y))
        size = NativeWSIReader._target_size(
            image.shape[1], image.shape[0], current_x, current_y, target_x, target_y
        )
        return NativeWSIReader._resample_to_size(image, size)

    def _fallback_mpp_or_raise(self, path: Path) -> tuple[float, float]:
        raw = self.fallback_mpp
        if raw is None:
            env = os.environ.get("WSI_NATIVE_MPP", "").strip()
            if env:
                components = [part.strip() for part in env.split(",")]
                if len(components) == 1:
                    raw = float(components[0])
                elif len(components) == 2:
                    raw = (float(components[0]), float(components[1]))
                else:
                    raise ValueError(
                        "WSI_NATIVE_MPP must be one or two comma-separated positive numbers."
                    )
        if raw is None:
            raise ValueError(
                f"Cannot determine level-0 MPP from {path.name}. Set WSI_NATIVE_MPP to a calibrated value or specify fallback_mpp."
            )
        return self._normalise_target_mpp(raw)

    @staticmethod
    def _unit_to_micrometres(unit: str) -> float:
        normalized = str(unit).casefold().strip().replace("μ", "µ")
        factors = {
            "pm": 1e-6,
            "picometer": 1e-6,
            "picometre": 1e-6,
            "nm": 1e-3,
            "nanometer": 1e-3,
            "nanometre": 1e-3,
            "µm": 1.0,
            "um": 1.0,
            "micrometer": 1.0,
            "micrometre": 1.0,
            "mm": 1e3,
            "millimeter": 1e3,
            "millimetre": 1e3,
            "cm": 1e4,
            "centimeter": 1e4,
            "centimetre": 1e4,
            "dm": 1e5,
            "decimeter": 1e5,
            "decimetre": 1e5,
            "m": 1e6,
            "meter": 1e6,
            "metre": 1e6,
            "km": 1e9,
            "kilometer": 1e9,
            "kilometre": 1e9,
            "å": 1e-4,
            "angstrom": 1e-4,
            "ångström": 1e-4,
            "in": 25400.0,
            "inch": 25400.0,
        }
        if normalized not in factors:
            raise ValueError(f"Unsupported physical pixel-size unit: {unit!r}.")
        return factors[normalized]

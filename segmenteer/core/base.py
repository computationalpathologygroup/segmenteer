from __future__ import annotations
import functools
import importlib
import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, Union
import geojson
import numpy as np
import numpy.typing as npt
import skimage
import yaml
from skimage.color import rgb2gray
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.benchmark.profiling import measure_phase
from segmenteer.core.utils import mask_to_geojson, MASK_POLYGONIZATION_PROTOCOL
from segmenteer.wsi.reader import NativeWSIReader


def _reader_backend_from_env() -> "WSIBackend":
    raw = os.environ.get("WSI_READER", WSIBackend.AUTO.value).casefold().strip()
    try:
        return WSIBackend(raw)
    except ValueError as exc:
        allowed = ", ".join((item.value for item in WSIBackend))
        raise ValueError(
            f"Unsupported WSI_READER={raw!r}. Use one of: {allowed}."
        ) from exc


def get_wsi_reader(
    backend: "WSIBackend | str | None" = None,
    native_mpp: float | tuple[float, float] | None = None,
) -> NativeWSIReader:
    selected = (
        _reader_backend_from_env() if backend is None else WSIBackend(str(backend))
    )
    return NativeWSIReader(selected.value, fallback_mpp=native_mpp)


__all__ = [
    "Segmenter",
    "NumpySegmenter",
    "PathSegmenter",
    "SegmentationResult",
    "WSIBackend",
    "WSI_READER",
    "get_wsi_reader",
    "load_segmenter",
    "segmenter_config_dict",
]


class WSIBackend(StrEnum):
    AUTO = "auto"
    OPENSLIDE = "openslide"
    TIFFFILE = "tifffile"


WSI_READER = _reader_backend_from_env()
_SKIP_ATTRS: frozenset[str] = frozenset(
    {"reader", "to_gray_func", "results", "segmenter", "segmenter_func"}
)


def _get_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("segmenteer")
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    return "unknown"


def _yaml_safe_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _yaml_safe_value(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _yaml_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_yaml_safe_value(item) for item in value]
    type_name = type(value).__name__
    module_name = type(value).__module__
    if type_name == "device" and module_name.startswith("torch"):
        return str(value)
    return str(value)


def segmenter_config_dict(segmenter: Any) -> dict:
    cls = type(segmenter)
    params = {
        key: _yaml_safe_value(value)
        for key, value in vars(segmenter).items()
        if not key.startswith("_") and key not in _SKIP_ATTRS and (not callable(value))
    }
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "params": params,
        "segmenteer_version": _get_version(),
    }


def _read_config(config_path: Path | str) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_segmenter(config_path: Path | str, reader: Any | None = None):
    cfg = _read_config(config_path)
    class_path = cfg["class"]
    params: dict = dict(cfg.get("params") or {})
    module_name, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    if reader is not None:
        params["reader"] = reader
    return cls(**params)


def _wrap_init_with_config(cls: type) -> None:
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, config=None, **kwargs):
        if config is not None:
            yaml_params = _read_config(config).get("params") or {}
            kwargs = {**yaml_params, **kwargs}
        start = time.perf_counter()
        orig_init(self, *args, **kwargs)
        self._constructor_time_s = time.perf_counter() - start

    cls.__init__ = __init__


@dataclass
class SegmentationResult:
    geojson: dict
    execution_time: float
    method_name: str
    metadata: dict


class Segmenter(Protocol):
    @property
    def name(self) -> str: ...

    def segment(self, path: Path) -> geojson.FeatureCollection: ...


class NumpySegmenter(ABC):
    APPLY_TO_GRAYSCALE: bool = True

    @property
    def prediction_protocol(self) -> str:
        # Display labels are excluded from run identity and configuration hashes.
        return MASK_POLYGONIZATION_PROTOCOL

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _wrap_init_with_config(cls)

    def __init__(
        self,
        mpp: float = 10,
        min_area: int = 0,
        to_gray_func: Union[callable, None] = rgb2gray,
        reader: Any | None = None,
        min_area_um2: float | None = None,
    ):
        self.mpp = NativeWSIReader._positive_mpp(mpp, "Segmentation target MPP")
        if not math.isfinite(float(min_area)) or min_area < 0:
            raise ValueError("min_area must be finite and nonnegative.")
        if min_area_um2 is not None and (
            not math.isfinite(float(min_area_um2)) or min_area_um2 < 0
        ):
            raise ValueError("min_area_um2 must be finite and nonnegative.")
        if min_area_um2 is not None and min_area != 0:
            raise ValueError("Specify either min_area or min_area_um2, not both.")
        self.min_area = min_area
        self.min_area_um2 = min_area_um2
        self.to_gray_func = to_gray_func
        self.reader = reader if reader is not None else get_wsi_reader()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def segment(self, path: Path) -> geojson.FeatureCollection:
        with measure_phase("image_loading"):
            wsi = self.reader.read(str(path))
            image = self.load_numpy(wsi)
        with measure_phase("preprocessing"):
            preprocessed = self._preprocess(image)
        with measure_phase("algorithm"):
            mask = self._segment_numpy(preprocessed)
        with measure_phase("polygon_conversion"):
            return self._convert_to_geojson(wsi, mask)

    def load_numpy(self, wsi):
        image = self.reader.get_wsi_at_mpp(wsi, (self.mpp, self.mpp))
        self._current_mpp = self._effective_spacing(wsi, image.shape)
        return image

    def _preprocess(self, image: npt.NDArray[np.int_]) -> npt.NDArray[np.uint8]:
        rgb = self._rgba_to_rgb(image)
        if self.APPLY_TO_GRAYSCALE and rgb.ndim == 3:
            preprocessed = self.to_gray_func(rgb)
        else:
            preprocessed = rgb
        return skimage.util.img_as_ubyte(preprocessed)

    def _rgba_to_rgb(self, image):
        return np.take(image, [0, 1, 2], 2)

    def _effective_spacing(
        self, wsi: Any, image_shape: tuple[int, ...]
    ) -> tuple[float, float]:
        width, height = self.reader.get_size(wsi, 0)
        native_x, native_y = self.reader.get_mpp(wsi, 0)
        return (native_x * width / image_shape[1], native_y * height / image_shape[0])

    def _minimum_area_pixels(self, wsi: Any, image_shape: tuple[int, ...]) -> int:
        if self.min_area_um2 is None:
            return self.min_area
        effective_x, effective_y = self._effective_spacing(wsi, image_shape)
        return math.ceil(self.min_area_um2 / (effective_x * effective_y))

    def _convert_to_geojson(
        self, wsi: Any, mask: npt.NDArray[np.bool]
    ) -> geojson.FeatureCollection:
        level0_width, level0_height = self.reader.get_size(wsi, 0)
        if mask.ndim != 2:
            raise ValueError("Segmentation mask must be two-dimensional.")
        scale_x = mask.shape[1] / level0_width
        scale_y = mask.shape[0] / level0_height
        return mask_to_geojson(
            mask,
            self._minimum_area_pixels(wsi, mask.shape),
            scaling_factor=(scale_x, scale_y),
        )

    @abstractmethod
    def _segment_numpy(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        pass


@dataclass
class TRIDENTSegmenter:
    segmenter: Any
    model_id: str = "Unknown"
    target_mag: int | None = None
    holes_are_tissue: bool = False
    batch_size: int = 8
    num_workers: int = 0
    _MODEL_LABELS = {
        "hest": "HEST",
        "grandqc": "GrandQC",
        "pathprofiler": "PathProfiler",
        "cpg": "CPG",
    }

    def __post_init__(self) -> None:
        model_key = str(self.model_id).strip().casefold()
        if not model_key or model_key == "unknown":
            model_key = self.segmenter.__class__.__name__.casefold()
            model_key = model_key.removeprefix("libtrident").removesuffix("segmenter")
        self.model_id = self._MODEL_LABELS.get(
            model_key, str(self.model_id).strip() or "Unknown"
        )
        if self.target_mag is None and (not hasattr(self.segmenter, "resolve")):
            declared = getattr(self.segmenter, "target_mag", 10)
            self.target_mag = int(declared)

    @property
    def name(self) -> str:
        return "trident_" + self.model_id.casefold().replace(" ", "_")

    @property
    def display_name(self) -> str:
        return f"TRIDENT {self.model_id}"

    @staticmethod
    def _best_device() -> str:
        configured = os.environ.get("SEGMENTEER_TRIDENT_DEVICE", "auto").strip()
        if configured and configured.casefold() != "auto":
            return configured
        return resolve_torch_device("auto")

    @staticmethod
    def _as_feature_collection(payload: Any) -> geojson.FeatureCollection:
        import json

        if hasattr(payload, "to_json"):
            payload = payload.to_json()
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Trident segment_tissue must return a GeoJSON object or JSON string; received {type(payload).__name__}."
            )
        if payload.get("type") != "FeatureCollection" or not isinstance(
            payload.get("features"), list
        ):
            raise ValueError("Trident output is not a GeoJSON FeatureCollection.")
        return geojson.loads(json.dumps(payload))

    def segment(self, path: Path) -> geojson.FeatureCollection:
        try:
            from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
        except ImportError as exc:
            raise ImportError(
                "Trident is required for this selected method. Run:\n  uv sync --extra trident"
            ) from exc
        if (
            hasattr(self.segmenter, "resolve")
            and getattr(self.segmenter, "_model", None) is None
        ):
            with measure_phase("model_initialization"):
                model = self.segmenter.resolve()
        else:
            model = (
                self.segmenter.resolve()
                if hasattr(self.segmenter, "resolve")
                else self.segmenter
            )
        target_mag = self.target_mag
        if target_mag is None:
            target_mag = getattr(model, "target_mag", 10)
        with measure_phase("image_loading"):
            wsi = OpenSlideWSI(str(path))
        try:
            with measure_phase("algorithm"):
                result = wsi.segment_tissue(
                    segmentation_model=model,
                    target_mag=int(target_mag),
                    holes_are_tissue=self.holes_are_tissue,
                    batch_size=self.batch_size,
                    device=self._best_device(),
                    num_workers=self.num_workers,
                )
            with measure_phase("polygon_conversion"):
                return self._as_feature_collection(result)
        finally:
            close = getattr(wsi, "close", None)
            if callable(close):
                close()


class PathSegmenter(ABC):
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _wrap_init_with_config(cls)

    def __init__(self, min_area: int = 0, reader: Any | None = None):
        self.min_area = min_area
        self.reader = reader if reader is not None else get_wsi_reader()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def segment(self, path: Path) -> geojson.FeatureCollection:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / "tissuemask.tif"
            with measure_phase("algorithm"):
                self._segment_path(path, output_path)
            with measure_phase("polygon_conversion"):
                return self._convert_to_geojson(path, output_path)

    @abstractmethod
    def _segment_path(self, image_path: Path, output_path: Path) -> None:
        pass

    def _convert_to_geojson(
        self, input_path: Path, output_path: Path
    ) -> geojson.FeatureCollection:
        try:
            import pyvips
        except ImportError:
            raise ImportError(
                "pyvips is required for PathSegmenter.\nInstall the wsi extra: pip install 'segmenteer[wsi]'"
            ) from None
        wsi = self.reader.read(input_path)
        width, height = self.reader.get_size(wsi, 0)
        mask = pyvips.Image.new_from_file(output_path)
        scaling_factor = (mask.width / width, mask.height / height)
        mask_array = np.asarray(mask.numpy())
        if mask_array.ndim == 3:
            if mask_array.shape[2] != 1:
                raise ValueError(
                    "PathSegmenter expects a single-channel segmentation mask."
                )
            mask_array = mask_array[:, :, 0]
        return mask_to_geojson(mask_array, self.min_area, scaling_factor=scaling_factor)

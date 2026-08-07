from __future__ import annotations

import functools
import importlib
import os
import tempfile
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
from segmenteer.core.utils import mask_to_geojson

# ---------------------------------------------------------------------------
# Lazy WSI-reader configuration
# ---------------------------------------------------------------------------

from segmenteer.wsi.reader import NativeWSIReader


def _reader_backend_from_env() -> "WSIBackend":
    raw = os.environ.get("WSI_READER", WSIBackend.AUTO.value).casefold().strip()
    try:
        return WSIBackend(raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in WSIBackend)
        raise ValueError(f"Unsupported WSI_READER={raw!r}. Use one of: {allowed}.") from exc


def get_wsi_reader(
    backend: "WSIBackend | str | None" = None,
    native_mpp: float | None = None,
) -> NativeWSIReader:
    """Create the lazy native reader selected by ``WSI_READER``.

    This function does not import OpenSlide or tifffile until a slide is read.
    ``WSI_NATIVE_MPP`` is consulted only when a slide has no physical MPP
    metadata and no explicit *native_mpp* is supplied.
    """
    selected = _reader_backend_from_env() if backend is None else WSIBackend(str(backend))
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
    """Supported native WSI reader backends."""

    AUTO = "auto"
    OPENSLIDE = "openslide"
    TIFFFILE = "tifffile"


# Selected only when :func:`get_wsi_reader` is called.  Importing segmenteer
# does not import either optional backend.
WSI_READER = _reader_backend_from_env()

# Attributes that carry no useful hyperparameter information and should not be
# included in config files or run-id slugs.  ``segmenter``/``segmenter_func`` are
# initialized model objects or callables; persisting their repr would make a run
# id unstable and cannot be used to reconstruct a model.
_SKIP_ATTRS: frozenset[str] = frozenset(
    {"reader", "to_gray_func", "results", "segmenter", "segmenter_func"}
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


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
    """Convert runtime-only values into primitives accepted by ``safe_dump``.

    Segmenters commonly retain ``torch.device`` and NumPy scalar objects as
    public configuration attributes.  Serialising those values directly makes
    PyYAML emit Python-specific tags such as ``!!python/object/apply`` which
    cannot be loaded through :func:`yaml.safe_load`.  Config files are intended
    to be portable, so retain only their plain-data representation.
    """
    # ``str, Enum`` members also satisfy ``isinstance(value, str)``.  Check
    # Enum first so PyYAML receives the member's plain, stable value rather
    # than a Python enum instance it cannot represent with SafeDumper.
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

    # Avoid importing Torch just to serialize a configuration.  ``torch.device``
    # is reconstructed by model constructors from its canonical string form.
    type_name = type(value).__name__
    module_name = type(value).__module__
    if type_name == "device" and module_name.startswith("torch"):
        return str(value)
    return str(value)


def segmenter_config_dict(segmenter: Any) -> dict:
    """Return a safe-YAML-serialisable dict that fully describes *segmenter*.

    The output uses only plain scalars, lists and mappings.  It can therefore
    always be persisted with :func:`yaml.safe_dump` and reconstructed through
    :func:`yaml.safe_load` without Python-object YAML tags.
    """
    cls = type(segmenter)
    params = {
        key: _yaml_safe_value(value)
        for key, value in vars(segmenter).items()
        if not key.startswith("_") and key not in _SKIP_ATTRS and not callable(value)
    }
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "params": params,
        "segmenteer_version": _get_version(),
    }


def _read_config(config_path: Path | str) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_segmenter(config_path: Path | str, reader: Any | None = None):
    """Reconstruct a segmenter from a ``config.yaml`` saved during a benchmark run.

    Parameters
    ----------
    config_path:
        Path to a ``config.yaml`` file written by :class:`EnsembleOutputWriter`.
    reader:
        Optional reader object to inject. When *None*, the backend selected by
        ``WSI_READER`` is used.

    Examples
    --------
    >>> seg = load_segmenter("outputs/1903_1241/otsu__mpp=5_min-area=0/config.yaml")
    # equivalent to:
    >>> seg = OtsuSegmenter(mpp=5, min_area=0)
    """
    cfg = _read_config(config_path)
    class_path = cfg["class"]
    params: dict = dict(cfg.get("params") or {})

    module_name, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)

    if reader is not None:
        params["reader"] = reader

    return cls(**params)


def _wrap_init_with_config(cls: type) -> None:
    """Decorate *cls.__init__* to accept an optional ``config=`` keyword.

    When ``config=<path>`` is passed the YAML is loaded and its ``params``
    dict is merged with any explicitly-provided kwargs (explicit values win),
    then the original ``__init__`` is invoked as normal.
    """
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, config=None, **kwargs):  # noqa: N807
        if config is not None:
            yaml_params = _read_config(config).get("params") or {}
            # Explicit kwargs shadow YAML values so partial overrides work.
            kwargs = {**yaml_params, **kwargs}
        orig_init(self, *args, **kwargs)

    cls.__init__ = __init__


@dataclass
class SegmentationResult:
    geojson: dict
    execution_time: float
    method_name: str
    metadata: dict


class Segmenter(Protocol):
    """Protocol defining the interface for all segmenters."""

    @property
    def name(self) -> str: ...

    def segment(self, path: Path) -> geojson.FeatureCollection: ...


class NumpySegmenter(ABC):
    """Base class for segmenters that work on numpy arrays."""

    APPLY_TO_GRAYSCALE: bool = True

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _wrap_init_with_config(cls)

    def __init__(
        self,
        mpp: float = 10,
        min_area: int = 0,
        to_gray_func: Union[callable, None] = rgb2gray,
        reader: Any | None = None,
    ):
        self.mpp = mpp
        self.min_area = min_area
        self.to_gray_func = to_gray_func
        self.reader = reader if reader is not None else get_wsi_reader()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Satisfies Segmenter protocol."""
        wsi = self.reader.read(str(path))
        image = self.load_numpy(wsi)
        preprocessed = self._preprocess(image)
        mask = self._segment_numpy(preprocessed)
        return self._convert_to_geojson(wsi, mask)

    def load_numpy(self, wsi):
        return self.reader.get_wsi_at_mpp(wsi, (self.mpp, self.mpp))

    def _preprocess(self, image: npt.NDArray[np.int_]) -> npt.NDArray[np.uint8]:
        rgb = self._rgba_to_rgb(image)

        if self.APPLY_TO_GRAYSCALE and rgb.ndim == 3:
            preprocessed = self.to_gray_func(rgb)
        else:
            preprocessed = rgb

        return skimage.util.img_as_ubyte(preprocessed)

    def _rgba_to_rgb(self, image):
        return np.take(image, [0, 1, 2], 2)

    def _convert_to_geojson(
        self, wsi: Any, mask: npt.NDArray[np.bool]
    ) -> geojson.FeatureCollection:
        level0_width, level0_height = self.reader.get_size(wsi, 0)
        scale_x = mask.shape[1] / level0_width
        scale_y = mask.shape[0] / level0_height
        if not np.isclose(scale_x, scale_y, rtol=0.01, atol=1e-6):
            raise ValueError(
                "WSI reader returned a non-uniformly scaled image; "
                f"x scale={scale_x:.6f}, y scale={scale_y:.6f}."
            )
        return mask_to_geojson(mask, self.min_area, scaling_factor=scale_x)

    @abstractmethod
    def _segment_numpy(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.bool_]:
        pass


@dataclass
class TRIDENTSegmenter:
    """Adapter that normalises Trident tissue segmentation into GeoJSON.

    ``target_mag`` defaults to the model's declared magnification rather than
    forcing every Trident model to 10x.  That preserves the spacing expected by
    HEST, GrandQC, PathProfiler and CPG adapters.

    ``model_id`` is intentionally persisted with each output configuration.
    Trident's public adapters all share this wrapper class, so the wrapped model
    identity must not be inferred from the wrapper class name by the viewer.
    """

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
            # This fallback keeps manually-created adapters readable while the
            # project factories below provide the canonical explicit identity.
            model_key = self.segmenter.__class__.__name__.casefold()
            model_key = model_key.removeprefix("libtrident").removesuffix("segmenter")
        self.model_id = self._MODEL_LABELS.get(model_key, str(self.model_id).strip() or "Unknown")

        if self.target_mag is None:
            declared = getattr(self.segmenter, "target_mag", 10)
            self.target_mag = int(declared)

    @property
    def name(self) -> str:
        """Filesystem-safe method id used in benchmark run directories."""
        return "trident_" + self.model_id.casefold().replace(" ", "_")

    @property
    def display_name(self) -> str:
        """Human-readable model label for reports and the viewer."""
        return f"TRIDENT {self.model_id}"

    @staticmethod
    def _best_device() -> str:
        configured = os.environ.get("SEGMENTEER_TRIDENT_DEVICE", "auto").strip()
        if configured and configured.casefold() != "auto":
            return configured

        return resolve_torch_device("auto")

    @staticmethod
    def _as_feature_collection(payload: Any) -> geojson.FeatureCollection:
        """Validate the one output contract shared by every Trident adapter."""
        import json

        if hasattr(payload, "to_json"):
            payload = payload.to_json()
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Trident segment_tissue must return a GeoJSON object or JSON string; "
                f"received {type(payload).__name__}."
            )
        if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
            raise ValueError("Trident output is not a GeoJSON FeatureCollection.")
        return geojson.loads(json.dumps(payload))

    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Run Trident and return the canonical level-0 GeoJSON prediction."""
        try:
            from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
        except ImportError as exc:
            raise ImportError(
                "Trident is required for this selected method. Run:\n"
                "  uv sync --extra trident"
            ) from exc

        wsi = OpenSlideWSI(str(path))
        try:
            result = wsi.segment_tissue(
                segmentation_model=self.segmenter,
                target_mag=int(self.target_mag),
                holes_are_tissue=self.holes_are_tissue,
                batch_size=self.batch_size,
                device=self._best_device(),
                num_workers=self.num_workers,
            )
            return self._as_feature_collection(result)
        finally:
            close = getattr(wsi, "close", None)
            if callable(close):
                close()


class PathSegmenter(ABC):
    """Base class for segmenters that need the image file path and output a mask file path."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _wrap_init_with_config(cls)

    def __init__(
        self,
        min_area: int = 0,
        reader: Any | None = None,
    ):
        self.min_area = min_area
        self.reader = reader if reader is not None else get_wsi_reader()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Satisfies Segmenter protocol."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / "tissuemask.tif"
            self._segment_path(path, output_path)
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
                "pyvips is required for PathSegmenter.\n"
                "Install the wsi extra: pip install 'segmenteer[wsi]'"
            ) from None
        wsi = self.reader.read(input_path)
        width = self.reader.get_size(wsi, 0)[0]
        mask = pyvips.Image.new_from_file(output_path)
        scaling_factor = mask.width / width
        return mask_to_geojson(
            mask.numpy(), self.min_area, scaling_factor=scaling_factor
        )

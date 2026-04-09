from __future__ import annotations

import functools
import importlib
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union

import geojson
import numpy as np
import numpy.typing as npt
import skimage
import yaml
from skimage.color import rgb2gray

from segmenteer.core.utils import mask_to_geojson

# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

try:
    from monai.data.wsi_reader import WSIReader as _WSIReader

    _MONAI_AVAILABLE = True
except ImportError:
    _WSIReader = None  # type: ignore[assignment,misc]
    _MONAI_AVAILABLE = False

# TRIDENT is imported lazily on first use to avoid downloading models unnecessarily
_TRIDENT_AVAILABLE: bool | None = None
TRIDENTSegmentationModel = None  # type: ignore[assignment]
_TridentWSI = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from trident.segmentation_models.load import (
        SegmentationModel as TRIDENTSegmentationModel,
    )
    from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI as _TridentWSI

if TYPE_CHECKING:
    from monai.data.wsi_reader import WSIReader


def _require_trident():
    """Lazy import TRIDENT. Downloads models on first use only."""
    global _TRIDENT_AVAILABLE, TRIDENTSegmentationModel, _TridentWSI

    if _TRIDENT_AVAILABLE is not None:
        # Already attempted import
        if not _TRIDENT_AVAILABLE:
            raise ImportError(
                "trident is required for TRIDENT segmenters.\n"
                "Install the trident extra: pip install 'segmenteer[trident]'"
            )
        return TRIDENTSegmentationModel, _TridentWSI

    try:
        from trident.segmentation_models.load import (
            SegmentationModel as _TSM,
        )
        from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI as _TWSI

        TRIDENTSegmentationModel = _TSM
        _TridentWSI = _TWSI
        _TRIDENT_AVAILABLE = True
        return TRIDENTSegmentationModel, _TridentWSI
    except ImportError:
        _TRIDENT_AVAILABLE = False
        raise ImportError(
            "trident is required for TRIDENT segmenters.\n"
            "Install the trident extra: pip install 'segmenteer[trident]'"
        ) from None


def _require_wsi_reader():
    """Return the WSIReader class, raising a helpful error if monai is not installed."""
    if not _MONAI_AVAILABLE:
        raise ImportError(
            "monai is required for WSI reading.\n"
            "Install the wsi extra: pip install 'segmenteer[wsi]'"
        )
    return _WSIReader

__all__ = [
    "Segmenter",
    "NumpySegmenter",
    "PathSegmenter",
    "SegmentationResult",
    "WSIBackend",
    "WSI_READER",
    "load_segmenter",
    "segmenter_config_dict",
]


class WSIBackend(StrEnum):
    """MONAI WSIReader backend. Override the default with the WSI_READER env var."""

    OPENSLIDE = "openslide"
    CUCIM = "cucim"
    TIFFFILE = "tifffile"


# WSI reader backend used by MONAI.  Override with the WSI_READER env var,
# e.g.: WSI_READER=cucim python ...
WSI_READER = WSIBackend(os.environ.get("WSI_READER", WSIBackend.OPENSLIDE))

# Attributes that carry no useful hyperparameter information and should not be
# included in config files or run-id slugs.
_SKIP_ATTRS: frozenset[str] = frozenset({"reader", "to_gray_func", "results"})


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


def segmenter_config_dict(segmenter: Any) -> dict:
    """Return a YAML-serialisable dict that fully describes *segmenter*.

    The dict has three keys:

    ``class``
        Fully-qualified class path, e.g.
        ``segmenteer.methods.classical.threshold.OtsuSegmenter``.
    ``params``
        All public, non-callable instance attributes except those in
        :data:`_SKIP_ATTRS` (reader, functional transforms, …).
    ``segmenteer_version``
        The installed version of the segmenteer package at the time the
        config was written, for reproducibility.

    The dict is intentionally kept flat so that :func:`load_segmenter` can
    reconstruct the segmenter with a plain ``cls(**params)`` call.
    """
    cls = type(segmenter)
    params = {
        k: v
        for k, v in vars(segmenter).items()
        if not k.startswith("_") and k not in _SKIP_ATTRS and not callable(v)
    }
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "params": params,
        "segmenteer_version": _get_version(),
    }


def _read_config(config_path: Path | str) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def load_segmenter(config_path: Path | str, reader: WSIReader | None = None):
    """Reconstruct a segmenter from a ``config.yaml`` saved during a benchmark run.

    Parameters
    ----------
    config_path:
        Path to a ``config.yaml`` file written by :class:`EnsembleOutputWriter`.
    reader:
        Optional ``WSIReader`` to inject.  When *None* the default backend
        (``WSI_READER`` env var or ``"openslide"``) is used.

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
        min_area: int = 10,
        to_gray_func: Union[callable, None] = rgb2gray,
        reader: WSIReader | None = None,
    ):
        self.mpp = mpp
        self.min_area = min_area
        self.to_gray_func = to_gray_func
        self.reader = reader if reader is not None else _require_wsi_reader()(WSI_READER)

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
        scaling_factor = mask.shape[0] / self.reader.get_size(wsi, 0)[0]
        return mask_to_geojson(mask, self.min_area, scaling_factor=scaling_factor)

    @abstractmethod
    def _segment_numpy(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.bool_]:
        pass


@dataclass
class TRIDENTSegmenter:
    """Base class for segmenters that work on numpy arrays."""

    segmenter: Any  # TRIDENTSegmentationModel, imported lazily on first use

    @property
    def name(self) -> str:
        return "trident_" + self.segmenter.__class__.__name__.lower()

    @staticmethod
    def _best_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Satisfies Segmenter protocol."""
        _require_trident()  # Lazy import and validation
        # Access globals set by _require_trident()
        global _TridentWSI
        wsi = _TridentWSI(path)
        return geojson.loads(
            wsi.segment_tissue(
                segmentation_model=self.segmenter,
                target_mag=10,
                holes_are_tissue=True,
                batch_size=8,
                device=self._best_device(),
                num_workers=0,  # >0 tries to pickle OpenSlide ctypes handles → fails on macOS
            ).to_json()
        )


class PathSegmenter(ABC):
    """Base class for segmenters that need the image file path and output a mask file path."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _wrap_init_with_config(cls)

    def __init__(
        self,
        min_area: int = 10,
        reader: WSIReader | None = None,
    ):
        self.min_area = min_area
        self.reader = reader if reader is not None else _require_wsi_reader()(WSI_READER)

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

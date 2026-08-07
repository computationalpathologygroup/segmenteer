"""Core Segmenteer API with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "Segmenter": ("segmenteer.core.base", "Segmenter"),
    "SegmentationResult": ("segmenteer.core.base", "SegmentationResult"),
    "mask_to_geojson": ("segmenteer.core.utils", "mask_to_geojson"),
    "geojson_to_mask": ("segmenteer.core.utils", "geojson_to_mask"),
    "downsample_image": ("segmenteer.core.utils", "downsample_image"),
    "scale_geojson_coordinates": ("segmenteer.core.utils", "scale_geojson_coordinates"),
    "RuntimeSettings": ("segmenteer.core.runtime", "RuntimeSettings"),
    "configure_runtime": ("segmenteer.core.runtime", "configure_runtime"),
    "resolve_torch_device": ("segmenteer.core.runtime", "resolve_torch_device"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'segmenteer.core' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = list(_LAZY_EXPORTS)

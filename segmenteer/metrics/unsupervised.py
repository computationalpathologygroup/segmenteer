"""Unsupervised (reference-free) segmentation quality metrics.

Metrics measure structural properties of the predicted polygons themselves
(area distribution, compactness, solidity, image coverage) without requiring
ground-truth annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import shape


@dataclass(frozen=True)
class UnsupervisedMetrics:
    """Immutable record of reference-free segmentation quality metrics."""

    num_objects: int
    total_area: float
    mean_area: float
    std_area: float
    median_area: float
    min_area: float
    max_area: float
    total_perimeter: float
    mean_perimeter: float
    #: 4πA / P² — 1.0 for a perfect circle, lower for irregular shapes.
    mean_compactness: float
    #: Object area / convex-hull area — 1.0 for convex shapes.
    mean_solidity: float
    #: Total tissue area / image area.
    coverage_ratio: float

    @classmethod
    def zero(cls) -> "UnsupervisedMetrics":
        """All-zero result — used for empty predictions."""
        return cls(
            num_objects=0,
            total_area=0.0,
            mean_area=0.0,
            std_area=0.0,
            median_area=0.0,
            min_area=0.0,
            max_area=0.0,
            total_perimeter=0.0,
            mean_perimeter=0.0,
            mean_compactness=0.0,
            mean_solidity=0.0,
            coverage_ratio=0.0,
        )


def compute_unsupervised_metrics(
    geojson_data: dict, image_area: float
) -> UnsupervisedMetrics:
    """Compute reference-free quality metrics from a GeoJSON prediction.

    Parameters
    ----------
    geojson_data:
        GeoJSON FeatureCollection produced by a segmenter.
    image_area:
        Total number of pixels (or coordinate units squared) in the source
        image — used to normalise ``coverage_ratio``.
    """
    areas: list[float] = []
    perimeters: list[float] = []
    compactnesses: list[float] = []
    solidities: list[float] = []

    for feat in geojson_data.get("features", []):
        try:
            geom = shape(feat["geometry"])
            area = geom.area
            perimeter = geom.length
            areas.append(area)
            perimeters.append(perimeter)
            if perimeter > 0:
                compactnesses.append(4 * np.pi * area / perimeter**2)
            hull_area = geom.convex_hull.area
            if hull_area > 0:
                solidities.append(area / hull_area)
        except (KeyError, TypeError, ValueError):
            continue

    if not areas:
        return UnsupervisedMetrics.zero()

    a = np.asarray(areas)
    p = np.asarray(perimeters)

    return UnsupervisedMetrics(
        num_objects=len(a),
        total_area=float(a.sum()),
        mean_area=float(a.mean()),
        std_area=float(a.std()),
        median_area=float(np.median(a)),
        min_area=float(a.min()),
        max_area=float(a.max()),
        total_perimeter=float(p.sum()),
        mean_perimeter=float(p.mean()),
        mean_compactness=float(np.mean(compactnesses)) if compactnesses else 0.0,
        mean_solidity=float(np.mean(solidities)) if solidities else 0.0,
        coverage_ratio=float(a.sum() / image_area) if image_area > 0 else 0.0,
    )

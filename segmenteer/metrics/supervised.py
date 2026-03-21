"""Supervised segmentation metrics computed in polygon space (Shapely) and
pixel space (NumPy).

All results are stored in an immutable :class:`SupervisedMetrics` dataclass.
Use :func:`compute_all_supervised_metrics` as the single entry-point, or call
individual ``compute_*`` helpers for targeted use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from shapely.errors import TopologicalError
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union


@dataclass(frozen=True)
class SupervisedMetrics:
    """Immutable record of supervised segmentation quality metrics.

    Area-based metrics are computed in polygon / coordinate space.
    Pixel-space metrics (``pixel_accuracy``, ``mae``,
    ``balanced_error_rate``) are ``0.0`` when no ``image_shape`` was supplied
    to :func:`compute_all_supervised_metrics`.
    """

    dice: float
    iou: float
    hausdorff: float
    precision: float
    recall: float
    over_segmentation_rate: float
    under_segmentation_rate: float
    pixel_accuracy: float
    mae: float
    balanced_error_rate: float

    @classmethod
    def zero(cls) -> "SupervisedMetrics":
        """All-zero result — used when prediction or ground truth is empty."""
        return cls(
            dice=0.0,
            iou=0.0,
            hausdorff=float("inf"),
            precision=0.0,
            recall=0.0,
            over_segmentation_rate=0.0,
            under_segmentation_rate=0.0,
            pixel_accuracy=0.0,
            mae=0.0,
            balanced_error_rate=0.0,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _union(geojson_data: dict) -> Optional[Polygon | MultiPolygon]:
    """Return the Shapely union of all valid polygons in *geojson_data*,
    or ``None`` when the collection is empty or malformed."""
    polys = []
    for feat in geojson_data.get("features", []):
        try:
            geom = shape(feat["geometry"])
            if not geom.is_empty and geom.is_valid:
                polys.append(geom)
        except (KeyError, TypeError, ValueError):
            continue
    return unary_union(polys) if polys else None


def _boundary_coords(geom: Polygon | MultiPolygon) -> np.ndarray:
    """Flatten all exterior ring coordinates into an (N, 2) float array."""
    coords: list = []
    if isinstance(geom, Polygon):
        coords.extend(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            coords.extend(poly.exterior.coords)
    return np.array(coords)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_dice(pred_geojson: dict, target_geojson: dict) -> float:
    """Sørensen–Dice coefficient in polygon space (0 = no overlap, 1 = perfect)."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None and t is None:
        return 1.0
    if p is None or t is None:
        return 0.0
    try:
        denom = p.area + t.area
        return 2.0 * p.intersection(t).area / denom if denom > 0 else 1.0
    except (ValueError, TopologicalError):
        return 0.0


def compute_iou(pred_geojson: dict, target_geojson: dict) -> float:
    """Intersection-over-Union in polygon space."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None and t is None:
        return 1.0
    if p is None or t is None:
        return 0.0
    try:
        denom = p.union(t).area
        return p.intersection(t).area / denom if denom > 0 else 1.0
    except (ValueError, TopologicalError):
        return 0.0


def compute_hausdorff(pred_geojson: dict, target_geojson: dict) -> float:
    """Symmetric Hausdorff distance between polygon boundaries (coordinate units)."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None or t is None:
        return float("inf")
    try:
        pc, tc = _boundary_coords(p), _boundary_coords(t)
        if len(pc) == 0 or len(tc) == 0:
            return float("inf")
        return max(
            directed_hausdorff(pc, tc)[0],
            directed_hausdorff(tc, pc)[0],
        )
    except (ValueError, TopologicalError):
        return float("inf")


def compute_precision(pred_geojson: dict, target_geojson: dict) -> float:
    """Precision = TP area / predicted positive area."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None or t is None:
        return 0.0
    try:
        return p.intersection(t).area / p.area if p.area > 0 else 0.0
    except (ValueError, TopologicalError):
        return 0.0


def compute_recall(pred_geojson: dict, target_geojson: dict) -> float:
    """Recall = TP area / actual positive area."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None or t is None:
        return 0.0
    try:
        return p.intersection(t).area / t.area if t.area > 0 else 0.0
    except (ValueError, TopologicalError):
        return 0.0


def compute_over_segmentation_rate(pred_geojson: dict, target_geojson: dict) -> float:
    """FP area / GT area — how much extra tissue was predicted."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None or t is None:
        return 0.0
    try:
        return p.difference(t).area / t.area if t.area > 0 else 0.0
    except (ValueError, TopologicalError):
        return 0.0


def compute_under_segmentation_rate(pred_geojson: dict, target_geojson: dict) -> float:
    """FN area / GT area — how much tissue was missed."""
    p, t = _union(pred_geojson), _union(target_geojson)
    if p is None or t is None:
        return 0.0
    try:
        return t.difference(p).area / t.area if t.area > 0 else 0.0
    except (ValueError, TopologicalError):
        return 0.0


# ---------------------------------------------------------------------------
# Pixel-space helpers
# ---------------------------------------------------------------------------


def compute_pixel_accuracy(
    pred_geojson: dict, target_geojson: dict, image_shape: tuple
) -> float:
    """(TP + TN) / total pixels."""
    from segmenteer.core.utils import geojson_to_mask

    pm = geojson_to_mask(pred_geojson, image_shape)
    tm = geojson_to_mask(target_geojson, image_shape)
    total = image_shape[0] * image_shape[1]
    return float(np.sum(pm == tm)) / total if total > 0 else 0.0


def compute_mae(pred_geojson: dict, target_geojson: dict, image_shape: tuple) -> float:
    """Mean absolute error between the binary prediction and GT masks."""
    from segmenteer.core.utils import geojson_to_mask

    pm = geojson_to_mask(pred_geojson, image_shape).astype(np.float32)
    tm = geojson_to_mask(target_geojson, image_shape).astype(np.float32)
    return float(np.mean(np.abs(pm - tm)))


def compute_balanced_error_rate(
    pred_geojson: dict, target_geojson: dict, image_shape: tuple
) -> float:
    """(FPR + FNR) / 2 in pixel space."""
    from segmenteer.core.utils import geojson_to_mask

    pm = geojson_to_mask(pred_geojson, image_shape)
    tm = geojson_to_mask(target_geojson, image_shape)
    n_pos, n_neg = np.sum(tm), np.sum(~tm)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    fpr = np.sum(pm & ~tm) / n_neg
    fnr = np.sum(~pm & tm) / n_pos
    return float((fpr + fnr) / 2.0)


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------


def compute_all_supervised_metrics(
    pred_geojson: dict,
    target_geojson: dict,
    image_shape: Optional[tuple] = None,
) -> SupervisedMetrics:
    """Compute the full suite of supervised metrics in one call.

    Parameters
    ----------
    pred_geojson:
        GeoJSON FeatureCollection produced by a segmenter.
    target_geojson:
        Ground-truth GeoJSON FeatureCollection.
    image_shape:
        ``(height, width)`` of the source image.  When provided, pixel-space
        metrics are computed; otherwise they are ``0.0``.
    """
    pixel_accuracy = mae = balanced_error_rate = 0.0
    if image_shape is not None:
        pixel_accuracy = compute_pixel_accuracy(
            pred_geojson, target_geojson, image_shape
        )
        mae = compute_mae(pred_geojson, target_geojson, image_shape)
        balanced_error_rate = compute_balanced_error_rate(
            pred_geojson, target_geojson, image_shape
        )

    return SupervisedMetrics(
        dice=compute_dice(pred_geojson, target_geojson),
        iou=compute_iou(pred_geojson, target_geojson),
        hausdorff=compute_hausdorff(pred_geojson, target_geojson),
        precision=compute_precision(pred_geojson, target_geojson),
        recall=compute_recall(pred_geojson, target_geojson),
        over_segmentation_rate=compute_over_segmentation_rate(
            pred_geojson, target_geojson
        ),
        under_segmentation_rate=compute_under_segmentation_rate(
            pred_geojson, target_geojson
        ),
        pixel_accuracy=pixel_accuracy,
        mae=mae,
        balanced_error_rate=balanced_error_rate,
    )

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import Polygon, shape, box
from shapely.ops import unary_union


@dataclass(frozen=True)
class SupervisedMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    over_segmentation_rate: float | None
    under_segmentation_rate: float | None
    pixel_accuracy: float
    mae: float
    balanced_error_rate: float

    @classmethod
    def zero(cls) -> "SupervisedMetrics":
        return cls(0.0, 0.0, 0.0, 0.0, None, None, 0.0, 0.0, 0.0)


def _union(geojson_data: dict):
    if (
        not isinstance(geojson_data, dict)
        or geojson_data.get("type") != "FeatureCollection"
        or not isinstance(geojson_data.get("features"), list)
    ):
        raise ValueError("Expected GeoJSON FeatureCollection")
    polygons = []
    for index, feature in enumerate(geojson_data["features"]):
        try:
            geom = shape(feature["geometry"])
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ValueError(f"Invalid GeoJSON feature {index}") from exc
        if geom.is_empty:
            continue
        if not geom.is_valid:
            raise ValueError(
                f"Invalid geometry in feature {index}: {shapely.is_valid_reason(geom)}"
            )
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"Feature {index} contains non-polygon geometry: {geom.geom_type}"
            )
        if not math.isfinite(geom.area) or geom.area <= 0:
            raise ValueError(f"Invalid polygon area in feature {index}")
        polygons.append(geom)
    result = unary_union(polygons) if polygons else Polygon()
    if not result.is_valid:
        raise ValueError("Polygon union is invalid")
    return result


def _polygonal_only(geometry):
    """Discard zero-area line/point intersections with slide boundaries.

    GEOS can return a LineString, Point, or GeometryCollection after clipping a
    valid polygon. These are not valid tissue regions or NSD boundaries.
    """
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry if geometry.area > 0 else Polygon()
    if geometry.geom_type == "GeometryCollection":
        parts = [_polygonal_only(part) for part in geometry.geoms]
        polygons = [part for part in parts if not part.is_empty]
        return unary_union(polygons) if polygons else Polygon()
    return Polygon()


def _positive_spacing(mpp):
    if isinstance(mpp, (tuple, list)) and len(mpp) == 2:
        sx, sy = float(mpp[0]), float(mpp[1])
    else:
        sx = sy = float(mpp)
    if not math.isfinite(sx) or not math.isfinite(sy) or sx <= 0 or sy <= 0:
        raise ValueError("Native MPP X/Y must be finite positive numbers")
    return sx, sy


def _rings(geom):
    if geom.is_empty:
        return []
    polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    rings = []
    for polygon in polygons:
        rings.append(polygon.exterior)
        rings.extend(polygon.interiors)
    return rings


def _sample_boundary(geom, step):
    parts = []
    weights = []
    count = 0
    for ring in _rings(geom):
        length = ring.length
        if length <= 0:
            continue
        n = max(1, math.ceil(length / step))
        count += n
        if count > 300000:
            raise ValueError(
                "Boundary sampling would exceed 300000 points; increase --boundary-step-um"
            )
        distances = (np.arange(n, dtype=np.float64) + 0.5) * (length / n)
        parts.append(
            shapely.get_coordinates(shapely.line_interpolate_point(ring, distances))
        )
        weights.append(np.full(n, length / n, dtype=np.float64))
    if not parts:
        return np.empty((0, 2)), np.empty(0)
    return np.concatenate(parts), np.concatenate(weights)


def _boundary_tree(geom):
    segments = []
    for ring in _rings(geom):
        coords = np.asarray(ring.coords, dtype=np.float64)
        if len(coords) > 1:
            valid = np.any(coords[:-1] != coords[1:], axis=1)
            if np.any(valid):
                segments.append(
                    np.stack((coords[:-1][valid], coords[1:][valid]), axis=1)
                )
    if not segments:
        return None
    return shapely.STRtree(shapely.linestrings(np.concatenate(segments, axis=0)))


@dataclass(frozen=True)
class PreparedBoundary:
    geometry: object
    coordinates: np.ndarray
    weights: np.ndarray
    tree: object


def _prepare_boundary(geometry, mpp, step):
    sx, sy = _positive_spacing(mpp)
    scaled = scale(geometry, xfact=sx, yfact=sy, origin=(0, 0))
    coordinates, weights = _sample_boundary(scaled, step)
    return PreparedBoundary(scaled, coordinates, weights, _boundary_tree(scaled))


def _within_boundary_tolerance(source, destination, tolerance):
    if destination.tree is None or not len(source.coordinates):
        return 0.0
    points = shapely.points(source.coordinates)
    matched = destination.tree.query_nearest(
        points, max_distance=tolerance, all_matches=False
    )
    return float(source.weights[matched[0]].sum()) if matched.shape[1] else 0.0


def compute_boundary_metrics(
    pred, target, mpp, *, tolerance_um=None, sample_step_um=10.0, prepared_target=None
):
    if tolerance_um is None:
        return {"nsd": None}
    tolerance = float(tolerance_um)
    step = float(sample_step_um)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("NSD tolerance must be finite and positive")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Boundary sample step must be finite and positive")
    if pred.is_empty and target.is_empty:
        return {"nsd": None}
    if pred.is_empty or target.is_empty:
        return {"nsd": 0.0}
    reference = (
        prepared_target
        if prepared_target is not None
        else _prepare_boundary(target, mpp, step)
    )
    if pred.equals(target):
        return {"nsd": 1.0}
    prediction = _prepare_boundary(pred, mpp, step)
    total = float(reference.weights.sum() + prediction.weights.sum())
    if not total:
        return {"nsd": None}
    matched = _within_boundary_tolerance(prediction, reference, tolerance)
    matched += _within_boundary_tolerance(reference, prediction, tolerance)
    return {"nsd": float(matched / total)}


@dataclass(frozen=True)
class PreparedReference:
    geometry: object
    outside_area: float | None
    coverage_ratio: float | None
    boundary: PreparedBoundary | None


def prepare_reference(
    target_geojson,
    *,
    mpp=None,
    image_size=None,
    boundary_tolerance_um=None,
    boundary_step_um=10.0,
):
    target = _union(target_geojson)
    outside = coverage = None
    if image_size is not None:
        width, height = image_size
        if int(width) != width or int(height) != height or width <= 0 or height <= 0:
            raise ValueError("Image size must be positive integer (width, height)")
        target_clipped = _polygonal_only(
            target.intersection(box(0, 0, int(width), int(height)))
        )
        outside = float(max(0.0, target.area - target_clipped.area))
        target = target_clipped
        coverage = float(target.area / (width * height))
    boundary = None
    if boundary_tolerance_um is not None:
        if mpp is None:
            raise ValueError("NSD requires native MPP")
        tolerance, step = float(boundary_tolerance_um), float(boundary_step_um)
        if (
            not math.isfinite(tolerance)
            or tolerance <= 0
            or not math.isfinite(step)
            or step <= 0
        ):
            raise ValueError(
                "NSD tolerance and boundary sampling step must be positive and finite"
            )
        if not target.is_empty:
            boundary = _prepare_boundary(target, mpp, step)
    return PreparedReference(target, outside, coverage, boundary)


def calculate_supervised_metrics(
    pred_geojson,
    target_geojson,
    *,
    mpp=None,
    boundary_tolerance_um=None,
    boundary_step_um=10.0,
    image_size=None,
    progress_stage=None,
    prepared_reference=None,
):
    stage = progress_stage or (lambda name: None)
    stage("Constructing prediction geometry")
    pred = _union(pred_geojson)
    reference = prepared_reference
    if reference is None:
        stage("Preparing reference geometry")
        reference = prepare_reference(
            target_geojson,
            mpp=mpp,
            image_size=image_size,
            boundary_tolerance_um=boundary_tolerance_um,
            boundary_step_um=boundary_step_um,
        )
    target = reference.geometry
    pred_outside = None
    if image_size is not None:
        width, height = image_size
        if int(width) != width or int(height) != height or width <= 0 or height <= 0:
            raise ValueError("Image size must be positive integer (width, height)")
        clipped = _polygonal_only(pred.intersection(box(0, 0, int(width), int(height))))
        pred_outside = float(max(0.0, pred.area - clipped.area))
        pred = clipped
    stage("Calculating overlap and physical area")
    pa, ta = pred.area, target.area
    intersection = pred.intersection(target).area
    fp = max(0.0, pa - intersection)
    fn = max(0.0, ta - intersection)
    both_empty = pa == 0 and ta == 0
    metrics = {
        "dice": float(2 * intersection / (pa + ta)) if pa + ta else 1.0,
        "iou": float(intersection / (pa + ta - intersection))
        if pa + ta - intersection
        else 1.0,
        "precision": float(intersection / pa) if pa else (1.0 if both_empty else 0.0),
        "recall": float(intersection / ta) if ta else (1.0 if both_empty else None),
        "over_segmentation_rate": float(fp / ta) if ta else None,
        "under_segmentation_rate": float(fn / ta) if ta else None,
        "area_bias_pct": float(100 * (pa - ta) / ta) if ta else None,
        "predicted_area_mm2": None,
        "ground_truth_area_mm2": None,
        "false_positive_area_mm2": None,
        "false_negative_area_mm2": None,
        "nsd": None,
        "boundary_tolerance_um": float(boundary_tolerance_um)
        if boundary_tolerance_um is not None
        else None,
        "boundary_step_um": None,
        "prediction_outside_area_px2": pred_outside,
        "gt_outside_area_px2": reference.outside_area,
    }
    if mpp is not None:
        sx, sy = _positive_spacing(mpp)
        factor = sx * sy / 1e6
        metrics.update(
            predicted_area_mm2=float(pa * factor),
            ground_truth_area_mm2=float(ta * factor),
            false_positive_area_mm2=float(fp * factor),
            false_negative_area_mm2=float(fn * factor),
        )
        if boundary_tolerance_um is not None:
            stage("Sampling boundaries and calculating NSD")
            metrics.update(
                compute_boundary_metrics(
                    pred,
                    target,
                    (sx, sy),
                    tolerance_um=boundary_tolerance_um,
                    sample_step_um=boundary_step_um,
                    prepared_target=reference.boundary,
                )
            )
            metrics["boundary_step_um"] = float(boundary_step_um)
    elif boundary_tolerance_um is not None:
        raise ValueError("NSD requires native MPP")
    return metrics


def compute_dice(pred_geojson: dict, target_geojson: dict) -> float:
    return calculate_supervised_metrics(pred_geojson, target_geojson)["dice"]


def compute_iou(pred_geojson: dict, target_geojson: dict) -> float:
    return calculate_supervised_metrics(pred_geojson, target_geojson)["iou"]


def compute_precision(pred_geojson: dict, target_geojson: dict) -> float:
    return calculate_supervised_metrics(pred_geojson, target_geojson)["precision"]


def compute_recall(pred_geojson: dict, target_geojson: dict) -> float | None:
    return calculate_supervised_metrics(pred_geojson, target_geojson)["recall"]


def compute_over_segmentation_rate(
    pred_geojson: dict, target_geojson: dict
) -> float | None:
    return calculate_supervised_metrics(pred_geojson, target_geojson)[
        "over_segmentation_rate"
    ]


def compute_under_segmentation_rate(
    pred_geojson: dict, target_geojson: dict
) -> float | None:
    return calculate_supervised_metrics(pred_geojson, target_geojson)[
        "under_segmentation_rate"
    ]


def compute_pixel_accuracy(
    pred_geojson: dict, target_geojson: dict, image_shape: tuple
) -> float:
    from segmenteer.core.utils import geojson_to_mask

    _union(pred_geojson)
    _union(target_geojson)
    pm = geojson_to_mask(pred_geojson, image_shape)
    tm = geojson_to_mask(target_geojson, image_shape)
    total = image_shape[0] * image_shape[1]
    return float(np.sum(pm == tm)) / total if total > 0 else 0.0


def compute_mae(pred_geojson: dict, target_geojson: dict, image_shape: tuple) -> float:
    from segmenteer.core.utils import geojson_to_mask

    _union(pred_geojson)
    _union(target_geojson)
    pm = geojson_to_mask(pred_geojson, image_shape).astype(np.float32)
    tm = geojson_to_mask(target_geojson, image_shape).astype(np.float32)
    return float(np.mean(np.abs(pm - tm)))


def compute_balanced_error_rate(
    pred_geojson: dict, target_geojson: dict, image_shape: tuple
) -> float | None:
    from segmenteer.core.utils import geojson_to_mask

    _union(pred_geojson)
    _union(target_geojson)
    pm = geojson_to_mask(pred_geojson, image_shape)
    tm = geojson_to_mask(target_geojson, image_shape)
    n_pos, n_neg = np.sum(tm), np.sum(~tm)
    if n_pos == 0 or n_neg == 0:
        return None
    fpr = np.sum(pm & ~tm) / n_neg
    fnr = np.sum(~pm & tm) / n_pos
    return float((fpr + fnr) / 2.0)


def compute_all_supervised_metrics(
    pred_geojson: dict, target_geojson: dict, image_shape: Optional[tuple] = None
) -> SupervisedMetrics:
    values = calculate_supervised_metrics(pred_geojson, target_geojson)
    pixel_accuracy = mae = 0.0
    balanced_error_rate = None
    if image_shape is not None:
        pixel_accuracy = compute_pixel_accuracy(
            pred_geojson, target_geojson, image_shape
        )
        mae = compute_mae(pred_geojson, target_geojson, image_shape)
        balanced_error_rate = compute_balanced_error_rate(
            pred_geojson, target_geojson, image_shape
        )
    return SupervisedMetrics(
        dice=values["dice"],
        iou=values["iou"],
        precision=values["precision"],
        recall=values["recall"],
        over_segmentation_rate=values["over_segmentation_rate"],
        under_segmentation_rate=values["under_segmentation_rate"],
        pixel_accuracy=pixel_accuracy,
        mae=mae,
        balanced_error_rate=balanced_error_rate,
    )

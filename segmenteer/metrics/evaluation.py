from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import shape
from shapely.ops import unary_union


@dataclass
class SupervisedMetrics:
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


def geojson_to_union_polygon(geojson_data: dict):
    features = geojson_data.get("features", [])

    if len(features) == 0:
        return None

    polygons = []
    for feature in features:
        try:
            geom = shape(feature["geometry"])
            if geom.is_valid and not geom.is_empty:
                polygons.append(geom)
        except:
            continue

    if len(polygons) == 0:
        return None

    return unary_union(polygons)


def compute_dice(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if pred_poly is None and target_poly is None:
        return 1.0

    if pred_poly is None or target_poly is None:
        return 0.0

    try:
        intersection = pred_poly.intersection(target_poly).area
        union = pred_poly.area + target_poly.area

        if union == 0:
            return 1.0

        return 2.0 * intersection / union
    except:
        return 0.0


def compute_iou(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if pred_poly is None and target_poly is None:
        return 1.0

    if pred_poly is None or target_poly is None:
        return 0.0

    try:
        intersection = pred_poly.intersection(target_poly).area
        union = pred_poly.union(target_poly).area

        if union == 0:
            return 1.0

        return intersection / union
    except:
        return 0.0


def compute_hausdorff(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if pred_poly is None or target_poly is None:
        return float("inf")

    try:
        from shapely.geometry import MultiPolygon, Polygon
        
        def extract_all_coords(geom):
            coords = []
            if isinstance(geom, Polygon):
                coords.extend(list(geom.exterior.coords))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    coords.extend(list(poly.exterior.coords))
            return coords
        
        pred_coords = np.array(extract_all_coords(pred_poly))
        target_coords = np.array(extract_all_coords(target_poly))

        if len(pred_coords) == 0 or len(target_coords) == 0:
            return float("inf")

        dist_forward = directed_hausdorff(pred_coords, target_coords)[0]
        dist_backward = directed_hausdorff(target_coords, pred_coords)[0]

        return max(dist_forward, dist_backward)
    except Exception:
        return float("inf")


def compute_precision(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if pred_poly is None:
        return 0.0

    if target_poly is None:
        return 0.0

    try:
        true_positive = pred_poly.intersection(target_poly).area
        predicted_positive = pred_poly.area

        if predicted_positive == 0:
            return 0.0

        return true_positive / predicted_positive
    except:
        return 0.0


def compute_recall(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if target_poly is None:
        return 0.0

    if pred_poly is None:
        return 0.0

    try:
        true_positive = pred_poly.intersection(target_poly).area
        actual_positive = target_poly.area

        if actual_positive == 0:
            return 0.0

        return true_positive / actual_positive
    except:
        return 0.0


def compute_over_segmentation_rate(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if target_poly is None or pred_poly is None:
        return 0.0

    try:
        false_positive = pred_poly.difference(target_poly).area
        actual_positive = target_poly.area

        if actual_positive == 0:
            return 0.0

        return false_positive / actual_positive
    except:
        return 0.0


def compute_under_segmentation_rate(pred_geojson: dict, target_geojson: dict) -> float:
    pred_poly = geojson_to_union_polygon(pred_geojson)
    target_poly = geojson_to_union_polygon(target_geojson)

    if target_poly is None or pred_poly is None:
        return 0.0

    try:
        false_negative = target_poly.difference(pred_poly).area
        actual_positive = target_poly.area

        if actual_positive == 0:
            return 0.0

        return false_negative / actual_positive
    except:
        return 0.0


def compute_pixel_accuracy(pred_geojson: dict, target_geojson: dict, image_shape: tuple) -> float:
    from segmenteer.core.utils import geojson_to_mask
    
    pred_mask = geojson_to_mask(pred_geojson, image_shape)
    target_mask = geojson_to_mask(target_geojson, image_shape)
    
    total_pixels = image_shape[0] * image_shape[1]
    
    if total_pixels == 0:
        return 0.0
    
    true_positive = np.sum(pred_mask & target_mask)
    true_negative = np.sum(~pred_mask & ~target_mask)
    
    return (true_positive + true_negative) / total_pixels


def compute_mae(pred_geojson: dict, target_geojson: dict, image_shape: tuple) -> float:
    from segmenteer.core.utils import geojson_to_mask
    
    pred_mask = geojson_to_mask(pred_geojson, image_shape).astype(np.float32)
    target_mask = geojson_to_mask(target_geojson, image_shape).astype(np.float32)
    
    return np.mean(np.abs(pred_mask - target_mask))


def compute_balanced_error_rate(pred_geojson: dict, target_geojson: dict, image_shape: tuple) -> float:
    from segmenteer.core.utils import geojson_to_mask
    
    pred_mask = geojson_to_mask(pred_geojson, image_shape)
    target_mask = geojson_to_mask(target_geojson, image_shape)
    
    num_positive = np.sum(target_mask)
    num_negative = np.sum(~target_mask)
    
    if num_positive == 0 or num_negative == 0:
        return 0.0
    
    false_positive = np.sum(pred_mask & ~target_mask)
    false_negative = np.sum(~pred_mask & target_mask)
    
    fpr = false_positive / num_negative
    fnr = false_negative / num_positive
    
    return (fpr + fnr) / 2.0


def compute_all_supervised_metrics(
    pred_geojson: dict, target_geojson: dict, image_shape: tuple = None
) -> SupervisedMetrics:
    pixel_accuracy = 0.0
    mae = 0.0
    balanced_error_rate = 0.0
    
    if image_shape is not None:
        pixel_accuracy = compute_pixel_accuracy(pred_geojson, target_geojson, image_shape)
        mae = compute_mae(pred_geojson, target_geojson, image_shape)
        balanced_error_rate = compute_balanced_error_rate(pred_geojson, target_geojson, image_shape)
    
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

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


def geojson_to_union_polygon(geojson_data: dict):
    features = geojson_data.get('features', [])
    
    if len(features) == 0:
        return None
    
    polygons = []
    for feature in features:
        try:
            geom = shape(feature['geometry'])
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
        return float('inf')
    
    try:
        pred_coords = np.array(pred_poly.exterior.coords)
        target_coords = np.array(target_poly.exterior.coords)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            return float('inf')
        
        dist_forward = directed_hausdorff(pred_coords, target_coords)[0]
        dist_backward = directed_hausdorff(target_coords, pred_coords)[0]
        
        return max(dist_forward, dist_backward)
    except:
        return float('inf')


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


def compute_all_supervised_metrics(pred_geojson: dict, target_geojson: dict) -> SupervisedMetrics:
    return SupervisedMetrics(
        dice=compute_dice(pred_geojson, target_geojson),
        iou=compute_iou(pred_geojson, target_geojson),
        hausdorff=compute_hausdorff(pred_geojson, target_geojson),
        precision=compute_precision(pred_geojson, target_geojson),
        recall=compute_recall(pred_geojson, target_geojson),
        over_segmentation_rate=compute_over_segmentation_rate(pred_geojson, target_geojson),
        under_segmentation_rate=compute_under_segmentation_rate(pred_geojson, target_geojson),
    )
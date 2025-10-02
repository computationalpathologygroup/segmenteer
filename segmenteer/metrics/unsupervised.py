from dataclasses import dataclass
import numpy as np
from shapely.geometry import shape


@dataclass
class UnsupervisedMetrics:
    num_objects: int
    total_area: float
    mean_area: float
    std_area: float
    median_area: float
    min_area: float
    max_area: float
    total_perimeter: float
    mean_perimeter: float
    mean_compactness: float
    mean_solidity: float
    coverage_ratio: float


def compute_unsupervised_metrics(geojson_data: dict, image_area: float) -> UnsupervisedMetrics:
    features = geojson_data.get('features', [])
    
    if len(features) == 0:
        return UnsupervisedMetrics(
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
    
    areas = []
    perimeters = []
    compactnesses = []
    solidities = []
    
    for feature in features:
        try:
            geom = shape(feature['geometry'])
            
            area = geom.area
            perimeter = geom.length
            
            areas.append(area)
            perimeters.append(perimeter)
            
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                compactnesses.append(compactness)
            
            convex_hull = geom.convex_hull
            if convex_hull.area > 0:
                solidity = area / convex_hull.area
                solidities.append(solidity)
        except:
            continue
    
    if len(areas) == 0:
        return UnsupervisedMetrics(
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
    
    areas_arr = np.array(areas)
    perimeters_arr = np.array(perimeters)
    
    return UnsupervisedMetrics(
        num_objects=len(areas),
        total_area=float(np.sum(areas_arr)),
        mean_area=float(np.mean(areas_arr)),
        std_area=float(np.std(areas_arr)),
        median_area=float(np.median(areas_arr)),
        min_area=float(np.min(areas_arr)),
        max_area=float(np.max(areas_arr)),
        total_perimeter=float(np.sum(perimeters_arr)),
        mean_perimeter=float(np.mean(perimeters_arr)),
        mean_compactness=float(np.mean(compactnesses)) if compactnesses else 0.0,
        mean_solidity=float(np.mean(solidities)) if solidities else 0.0,
        coverage_ratio=float(np.sum(areas_arr) / image_area) if image_area > 0 else 0.0,
    )
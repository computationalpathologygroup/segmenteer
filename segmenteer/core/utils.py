import geojson
import math
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, mapping
from shapely.geometry import shape as shapely_shape
from skimage.measure import find_contours, label
from skimage.transform import rescale
from shapely.validation import make_valid


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    if image.ndim == 3:
        downsampled = rescale(
            image, 1.0 / factor, channel_axis=2, preserve_range=True, anti_aliasing=True
        )
    else:
        downsampled = rescale(
            image, 1.0 / factor, preserve_range=True, anti_aliasing=True
        )
    return downsampled.astype(image.dtype)


def _xy_scale(value: float | tuple[float, float] | list[float]) -> tuple[float, float]:
    if isinstance(value, (int, float, np.number)) and not isinstance(
        value, (bool, np.bool_)
    ):
        x = y = float(value)
    else:
        if (
            isinstance(value, (str, bytes))
            or not hasattr(value, "__len__")
            or len(value) != 2
        ):
            raise ValueError("Scaling must be a scalar or an (x_scale, y_scale) pair.")
        x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0:
        raise ValueError(f"Scaling must be positive and finite, got ({x}, {y}).")
    return x, y


def scale_geojson_coordinates(
    geojson_data: dict, scale_factor: float | tuple[float, float]
) -> dict:
    scale_x, scale_y = _xy_scale(scale_factor)
    if scale_x == 1.0 and scale_y == 1.0:
        return geojson_data
    scaled_features = []
    for feature in geojson_data.get("features", []):
        try:
            geometry = shapely_shape(feature["geometry"])
            if geometry.is_empty:
                continue
            scaled_geometry = affinity.scale(
                geometry, xfact=scale_x, yfact=scale_y, origin=(0, 0, 0)
            )
            scaled_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(scaled_geometry),
                    "properties": dict(feature.get("properties", {})),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    result = {"type": "FeatureCollection", "features": scaled_features}
    if isinstance(geojson_data.get("properties"), dict):
        result["properties"] = dict(geojson_data["properties"])
    return result


# Track geometry conversion separately from model configuration fingerprints.
MASK_POLYGONIZATION_PROTOCOL = "pixel-edges-4-connected-v1"


def _rasterio_features():
    """Load pixel-edge polygonization and rasterization utilities."""
    try:
        from rasterio.features import rasterize, shapes
        from affine import Affine
    except ImportError as exc:
        raise ImportError(
            "Lossless tissue-mask polygon conversion requires rasterio. "
            "Install it with: python -m pip install rasterio"
        ) from exc
    return rasterize, shapes, Affine


def mask_to_geojson(
    mask: np.ndarray, min_area: int = 0, scaling_factor: float | tuple[float, float] = 1
) -> dict:
    """Convert a binary mask to pixel-edge polygons using 4-connectivity.

    Corner-touching regions remain distinct; holes and islands are preserved.
    ``min_area`` is measured in mask pixels before coordinate scaling.
    """
    scale_x, scale_y = _xy_scale(scaling_factor)
    data = np.asarray(mask)
    if data.ndim != 2 or not data.shape[0] or not data.shape[1]:
        raise ValueError("Segmentation mask must be a nonempty two-dimensional array.")
    if not math.isfinite(float(min_area)) or min_area < 0:
        raise ValueError("min_area must be finite and nonnegative.")
    if data.dtype.kind not in "biuf":
        raise ValueError("Segmentation mask must be a numeric or boolean array.")
    if data.dtype.kind == "f" and not np.isfinite(data).all():
        raise ValueError("Segmentation mask must not contain NaN or infinity.")
    binary = np.ascontiguousarray(data.astype(bool), dtype=np.uint8)
    if not binary.any():
        return geojson.FeatureCollection([])
    _, shapes, Affine = _rasterio_features()

    features = []
    for geometry, value in shapes(
        binary, mask=binary.astype(bool), connectivity=4, transform=Affine.identity()
    ):
        if not value:
            continue
        polygon = shapely_shape(geometry)

        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Raster polygonizer produced invalid tissue geometry.")
        if polygon.area < min_area:
            continue
        native = affinity.scale(
            polygon, xfact=1 / scale_x, yfact=1 / scale_y, origin=(0, 0, 0)
        )
        features.append(
            geojson.Feature(
                geometry=mapping(native),
                properties={
                    "object_type": "annotation",
                    "classification": {"name": "Region", "color": [255, 0, 0]},
                },
            )
        )
    return geojson.FeatureCollection(features)


def geojson_to_mask(geojson_data: dict, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize polygon union at pixel centers, preserving holes.

    Overlapping features are order-independent. ``shape`` is (height, width).
    """
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError(
            "Mask shape must be a pair of positive integers (height, width)."
        )
    if (
        not isinstance(geojson_data, dict)
        or geojson_data.get("type") != "FeatureCollection"
    ):
        raise ValueError("Expected GeoJSON FeatureCollection.")
    features = geojson_data.get("features")
    if not isinstance(features, list):
        raise ValueError("Expected GeoJSON features list.")
    rasterize, _, Affine = _rasterio_features()
    polygons = []
    for index, feature in enumerate(features):
        try:
            geometry = shapely_shape(feature["geometry"])
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid GeoJSON feature {index}.") from exc
        if geometry.is_empty:
            continue
        if (
            geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or not geometry.is_valid
        ):
            raise ValueError(f"Invalid tissue polygon in feature {index}.")
        polygons.append((mapping(geometry), 1))
    if not polygons:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        polygons,
        out_shape=shape,
        transform=Affine.identity(),
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

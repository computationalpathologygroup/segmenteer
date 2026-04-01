import geojson
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, mapping
from shapely.geometry import shape as shapely_shape
from skimage.measure import find_contours, label
from skimage.transform import rescale


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


def scale_geojson_coordinates(geojson_data: dict, scale_factor: float) -> dict:
    if scale_factor == 1.0:
        return geojson_data

    scaled_features = []

    for feature in geojson_data.get("features", []):
        try:
            geom = feature["geometry"]

            if geom["type"] == "Polygon":
                scaled_coords = []
                for ring in geom["coordinates"]:
                    scaled_ring = [
                        [x * scale_factor, y * scale_factor] for x, y in ring
                    ]
                    scaled_coords.append(scaled_ring)

                scaled_geom = {"type": "Polygon", "coordinates": scaled_coords}

                scaled_feature = {
                    "type": "Feature",
                    "geometry": scaled_geom,
                    "properties": feature.get("properties", {}),
                }
                scaled_features.append(scaled_feature)
        except:
            continue

    return geojson.FeatureCollection(scaled_features)


def mask_to_geojson(
    mask: np.ndarray, min_area: int = 10, scaling_factor: float = 1
) -> dict:
    # Pad the mask to ensure contours touching image edges are properly closed
    # Without padding, edge-touching contours create invalid polygons when their endpoints are connected
    pad_width = 1
    padded_mask = np.pad(
        mask, pad_width=pad_width, mode="constant", constant_values=False
    )

    labeled = label(padded_mask)
    features = []

    for region_id in range(1, labeled.max() + 1):
        region_mask = labeled == region_id

        if region_mask.sum() < min_area:
            continue

        contours = find_contours(region_mask.astype(float), 0.5)

        if len(contours) == 0:
            continue

        largest_contour = max(contours, key=len)

        if len(largest_contour) < 3:
            continue

        # Adjust coordinates back to original image space by removing padding offset
        coords = [
            (float(x - pad_width), float(y - pad_width)) for y, x in largest_contour
        ]

        if len(coords) < 3:
            continue

        try:
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            polygon = affinity.scale(
                polygon,
                xfact=1 / scaling_factor,
                yfact=1 / scaling_factor,
                origin=(0, 0, 0),
            )

            if polygon.is_valid and not polygon.is_empty:
                feature = geojson.Feature(
                    geometry=mapping(polygon),
                    properties={
                        "object_type": "annotation",
                        "classification": {"name": "Region", "color": [255, 0, 0]},
                    },
                )
                features.append(feature)
        except:  # TODO: specify exception
            continue

    return geojson.FeatureCollection(features)


def geojson_to_mask(geojson_data: dict, shape: tuple) -> np.ndarray:
    from PIL import Image, ImageDraw
    from shapely.geometry import MultiPolygon, Polygon

    mask = np.zeros(shape, dtype=bool)
    features = geojson_data.get("features", [])

    if not features:
        return mask

    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)

    for feature in features:
        try:
            geom = shapely_shape(feature["geometry"])

            polygons = []
            if isinstance(geom, Polygon):
                polygons = [geom]
            elif isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)
            else:
                continue

            for poly in polygons:
                coords = list(poly.exterior.coords)
                coords_tuples = [(x, y) for x, y in coords]
                draw.polygon(coords_tuples, outline=255, fill=255)

        except Exception:
            continue

    mask = np.array(img) > 0

    return mask

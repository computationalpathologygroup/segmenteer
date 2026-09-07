import geojson
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


def scale_geojson_coordinates(geojson_data: dict, scale_factor: float) -> dict:
    """Scale all GeoJSON polygon coordinates around the level-0 origin.

    The previous implementation only handled ``Polygon`` features.  Supervised
    error layers can contain ``MultiPolygon`` geometries and holes, so use
    Shapely's geometry-preserving affine transform for every polygonal feature.
    """
    if scale_factor == 1.0:
        return geojson_data

    scaled_features = []
    for feature in geojson_data.get("features", []):
        try:
            geometry = shapely_shape(feature["geometry"])
            if geometry.is_empty:
                continue
            scaled_geometry = affinity.scale(
                geometry,
                xfact=scale_factor,
                yfact=scale_factor,
                origin=(0, 0, 0),
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


def mask_to_geojson(
    mask: np.ndarray, min_area: int = 0, scaling_factor: float = 1
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

        # ``find_contours`` returns an exterior boundary plus one boundary per
        # enclosed void.  Keeping only the largest contour (the pre-merge
        # behavior) silently filled donut holes and changed the segmentation
        # geometry.  Retain every contour contained by the exterior as a
        # polygon interior.
        contours = sorted(contours, key=len, reverse=True)
        exterior_contour = contours[0]
        if len(exterior_contour) < 3:
            continue

        # Adjust coordinates back to original image space by removing padding offset.
        exterior_coords = [
            (float(x - pad_width), float(y - pad_width))
            for y, x in exterior_contour
        ]
        exterior = Polygon(exterior_coords)
        if exterior.is_empty:
            continue

        holes = []
        for contour in contours[1:]:
            if len(contour) < 3:
                continue
            hole_coords = [
                (float(x - pad_width), float(y - pad_width))
                for y, x in contour
            ]
            # A contour belonging to a nested or adjacent component must not
            # be attached as an interior ring.  A representative point avoids
            # boundary-touching ambiguity from the raw contour start point.
            hole_polygon = Polygon(hole_coords)
            if (
                not hole_polygon.is_empty
                and exterior.contains(hole_polygon.representative_point())
            ):
                holes.append(hole_coords)

        try:
            polygon = Polygon(exterior_coords, holes=holes)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
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
                coords = [(x, y) for x, y in poly.exterior.coords]
                draw.polygon(coords, outline=255, fill=255)
                # Preserve interior rings so masks match the polygon geometry
                # used by supervised evaluation rather than filling holes.
                for interior in poly.interiors:
                    hole = [(x, y) for x, y in interior.coords]
                    draw.polygon(hole, outline=0, fill=0)

        except Exception:
            continue

    mask = np.array(img) > 0

    return mask

from __future__ import annotations
import math
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from segmenteer.io.atomic import write_text_atomic

TISSUE_GROUP = "tissue"
HOLE_GROUP = "hole"
TISSUE_COLOR = "#00E0F9"
HOLE_COLOR = "#FFFFFF"


def _normalise(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def _hex_to_rgb(value: str | None) -> list[int] | None:
    if not value:
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return [int(text[index : index + 2], 16) for index in (0, 2, 4)]
    except ValueError:
        return None


def _ring_from_annotation(annotation: ET.Element) -> list[list[float]]:
    name = annotation.attrib.get("Name", "<unnamed>")
    points = []
    coordinates = annotation.findall("./Coordinates/Coordinate")
    if not coordinates:
        raise ValueError(f"Annotation {name!r} has no coordinates")
    order_values = [node.attrib.get("Order") for node in coordinates]
    if any(value is not None for value in order_values) and not all(
        value is not None for value in order_values
    ):
        raise ValueError(
            f"Annotation {name!r} has partially specified coordinate order"
        )
    if all(value is not None for value in order_values):
        try:
            order = [int(value) for value in order_values]
        except ValueError as exc:
            raise ValueError(
                f"Annotation {name!r} has invalid coordinate order"
            ) from exc
        if len(set(order)) != len(order):
            raise ValueError(f"Annotation {name!r} has duplicate coordinate indices")
        coordinates = [
            node
            for _, node in sorted(zip(order, coordinates), key=lambda item: item[0])
        ]
    for coordinate in coordinates:
        try:
            x, y = float(coordinate.attrib["X"]), float(coordinate.attrib["Y"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Annotation {name!r} has invalid coordinates") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Annotation {name!r} contains non-finite coordinates")
        if not points or points[-1] != [x, y]:
            points.append([x, y])
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len({tuple(point) for point in points}) < 3:
        raise ValueError(f"Annotation {name!r} has fewer than three distinct vertices")
    return points + [points[0].copy()]


def _polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        polygons = []
        for part in geometry.geoms:
            polygons.extend(_polygons(part))
        return polygons
    raise ValueError(f"Unexpected non-area annotation geometry: {geometry.geom_type}")


def load_asap_xml(
    path: str | Path, *, groups: Iterable[str] | None = None, strict: bool = True
) -> dict:
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid ASAP XML annotation file: {path}") from exc
    if root.tag != "ASAP_Annotations":
        raise ValueError(f"Unexpected ASAP XML root element: {root.tag}")
    wanted_groups = _normalise(groups)
    features = []
    tissue = []
    holes = []
    names = {}
    pending = []
    counts = {"total": 0, "tissue": 0, "spline": 0, "hole": 0, "excluded": 0}
    for annotation in root.findall("./Annotations/Annotation"):
        name = annotation.attrib.get("Name", "")
        group = annotation.attrib.get("PartOfGroup", "").strip()
        normalized_group = group.casefold()
        kind = annotation.attrib.get("Type", "").strip().casefold()
        is_hole = normalized_group == HOLE_GROUP
        if (
            wanted_groups is not None
            and normalized_group not in wanted_groups
            and not is_hole
        ):
            if strict and kind in {"polygon", "spline", "rectangle"}:
                raise ValueError(
                    f"Unselected annotation group {group!r} for {name!r}; select explicit --tissue-groups"
                )
            counts["excluded"] += 1
            continue
        if kind not in {"polygon", "spline"}:
            if strict:
                raise ValueError(
                    f"Unsupported annotation type {kind!r} in group {group!r}: {name!r}"
                )
            counts["excluded"] += 1
            continue
        ring = _ring_from_annotation(annotation)
        polygon = Polygon(ring)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError(
                f"Invalid annotation polygon {name!r}: nonpositive area or self-intersecting ring"
            )
        counts["total"] += 1
        if kind == "spline":
            counts["spline"] += 1
        if is_hole:
            counts["hole"] += 1
            prefix = "Hole for "
            parent = (
                name.rsplit(" / ", 1)[0][len(prefix) :]
                if name.startswith(prefix)
                else None
            )
            pending.append((name, parent, polygon))
            continue
        names.setdefault(name, []).append(polygon)
        tissue.append(polygon)
        counts["tissue"] += 1
        color = annotation.attrib.get("Color")
        classification = {"name": group or "unclassified"}
        rgb = _hex_to_rgb(color)
        if rgb is not None:
            classification["color"] = rgb
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "object_type": "annotation",
                    "classification": classification,
                    "asap_name": name,
                    "asap_group": group,
                    "asap_color": color,
                    "asap_type": kind.capitalize(),
                },
            }
        )
    if pending:
        union = unary_union(tissue) if tissue else Polygon()
        for hole_name, parent, hole in pending:
            if parent is not None and parent not in names:
                raise ValueError(
                    f"Hole {hole_name!r} names a missing parent {parent!r}"
                )
            if parent is not None and len(names[parent]) != 1:
                raise ValueError(
                    f"Hole {hole_name!r} refers to ambiguous parent {parent!r}"
                )
            if parent is not None and not names[parent][0].covers(hole):
                raise ValueError(
                    f"Hole {hole_name!r} extends outside its parent tissue annotation"
                )
            if not union.covers(hole):
                raise ValueError(f"Hole {hole_name!r} extends outside the tissue mask")
            holes.append(hole)
        result = union.difference(unary_union(holes))
        if not result.is_valid:
            raise ValueError("Invalid tissue geometry after subtracting ASAP holes")
        features = [
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "object_type": "annotation",
                    "classification": {"name": TISSUE_GROUP},
                },
            }
            for poly in _polygons(result)
        ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "segmenteer_annotation_counts": counts,
    }


def _points(ring: object) -> list[tuple[float, float]]:
    if not isinstance(ring, (list, tuple)):
        raise ValueError("Polygon ring must be an array of coordinates")
    points: list[tuple[float, float]] = []
    for coordinate in ring:
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) < 2:
            raise ValueError("Invalid polygon coordinate")
        x, y = float(coordinate[0]), float(coordinate[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Non-finite polygon coordinate")
        if not points or points[-1] != (x, y):
            points.append((x, y))
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(set(points)) < 3:
        raise ValueError("Polygon ring must contain at least three distinct points")
    return points


def _append_annotation(
    parent: ET.Element, name: str, group: str, color: str, ring: object
) -> None:
    annotation = ET.SubElement(
        parent,
        "Annotation",
        {"Name": name, "Type": "Polygon", "PartOfGroup": group, "Color": color},
    )
    coords = ET.SubElement(annotation, "Coordinates")
    for index, (x, y) in enumerate(_points(ring)):
        ET.SubElement(
            coords,
            "Coordinate",
            {"Order": str(index), "X": format(x, ".17g"), "Y": format(y, ".17g")},
        )


def _polygon_parts(geometry: dict) -> list:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon":
        return [coordinates]
    if kind == "MultiPolygon":
        if not isinstance(coordinates, (list, tuple)):
            raise ValueError("Invalid multipolygon coordinates")
        return list(coordinates)
    if kind == "GeometryCollection":
        parts: list = []
        for member in geometry.get("geometries", []):
            if not isinstance(member, dict):
                raise ValueError("Invalid geometry collection member")
            parts.extend(_polygon_parts(member))
        return parts
    if kind in ("LineString", "MultiLineString", "Point", "MultiPoint"):
        return []
    raise ValueError(f"ASAP XML cannot represent geometry of type {kind!r}")


def save_asap_xml(payload: dict, path: str | Path) -> None:
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "FeatureCollection"
        or not isinstance(payload.get("features"), list)
    ):
        raise ValueError("ASAP XML requires a GeoJSON FeatureCollection")
    root = ET.Element("ASAP_Annotations")
    annotations = ET.SubElement(root, "Annotations")
    group_element = ET.SubElement(root, "AnnotationGroups")
    count = 0
    holes_present = False
    for feature in payload["features"]:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            raise ValueError("Expected a polygon GeoJSON geometry")
        polygons = _polygon_parts(geometry)
        for polygon in polygons:
            if not isinstance(polygon, (list, tuple)) or not polygon:
                raise ValueError("Invalid polygon rings")
            name = f"Annotation {count}"
            _append_annotation(
                annotations, name, TISSUE_GROUP, TISSUE_COLOR, polygon[0]
            )
            for index, interior in enumerate(polygon[1:]):
                _append_annotation(
                    annotations,
                    f"Hole for {name} / {index}",
                    HOLE_GROUP,
                    HOLE_COLOR,
                    interior,
                )
                holes_present = True
            count += 1
    ET.SubElement(
        ET.SubElement(
            group_element,
            "Group",
            {"Name": TISSUE_GROUP, "PartOfGroup": "None", "Color": TISSUE_COLOR},
        ),
        "Attributes",
    )
    if holes_present:
        ET.SubElement(
            ET.SubElement(
                group_element,
                "Group",
                {"Name": HOLE_GROUP, "PartOfGroup": "None", "Color": HOLE_COLOR},
            ),
            "Attributes",
        )
    ET.indent(root, space="\t")
    data = (
        ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8") + "\n"
    )
    write_text_atomic(path, data)

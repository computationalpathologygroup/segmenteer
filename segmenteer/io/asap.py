"""ASAP XML annotation import.

ASAP stores annotation vertices in level-0 pixel coordinates (``X``, ``Y``).
This module converts polygon annotations to a plain GeoJSON FeatureCollection
without rescaling, so it can be compared directly with segmenteer's level-0
prediction GeoJSON.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET


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


def _ring_from_annotation(annotation: ET.Element) -> list[list[float]] | None:
    ring: list[list[float]] = []
    for coordinate in annotation.findall("./Coordinates/Coordinate"):
        try:
            point = [float(coordinate.attrib["X"]), float(coordinate.attrib["Y"])]
        except (KeyError, TypeError, ValueError):
            continue
        if not ring or point != ring[-1]:
            ring.append(point)
    if len(ring) < 3:
        return None
    if ring[0] != ring[-1]:
        ring.append(ring[0].copy())
    return ring


def load_asap_xml(path: str | Path, *, groups: Iterable[str] | None = None) -> dict:
    """Load ASAP ``Type=Polygon`` annotations as a GeoJSON FeatureCollection.

    Coordinates are kept exactly as stored in the XML: level-0 slide pixels.
    ``groups`` is a case-insensitive allow-list; use ``{"tissue"}`` for the
    provided tissue-background ground truth.
    """
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid ASAP XML annotation file: {path}") from exc

    wanted_groups = _normalise(groups)
    features: list[dict] = []
    for annotation in root.findall("./Annotations/Annotation"):
        if annotation.attrib.get("Type", "").strip().casefold() != "polygon":
            continue
        group = annotation.attrib.get("PartOfGroup", "")
        if wanted_groups is not None and group.strip().casefold() not in wanted_groups:
            continue
        ring = _ring_from_annotation(annotation)
        if ring is None:
            continue

        color = annotation.attrib.get("Color")
        classification: dict = {"name": group or "unclassified"}
        rgb = _hex_to_rgb(color)
        if rgb is not None:
            classification["color"] = rgb
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "object_type": "annotation",
                    "classification": classification,
                    "asap_name": annotation.attrib.get("Name", ""),
                    "asap_group": group,
                    "asap_color": color,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}

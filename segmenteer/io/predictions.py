from __future__ import annotations
import json
from pathlib import Path
from segmenteer.io.asap import load_asap_xml, save_asap_xml
from segmenteer.io.atomic import write_json_atomic

PREDICTION_FORMATS = ("geojson", "xml")


def normalize_prediction_format(output_format: str) -> str:
    selected = str(output_format).strip().lower()
    if selected not in PREDICTION_FORMATS:
        raise ValueError(
            f"output_format must be one of: {', '.join(PREDICTION_FORMATS)}"
        )
    return selected


def prediction_path(predictions_dir: Path | str, stem: str, output_format: str) -> Path:
    return (
        Path(predictions_dir) / f"{stem}.{normalize_prediction_format(output_format)}"
    )


def read_prediction(path: Path | str) -> dict:
    path = Path(path)
    if path.suffix.casefold() == ".xml":
        payload = load_asap_xml(path)
    elif path.suffix.casefold() == ".geojson":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported prediction extension: {path.suffix}")
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "FeatureCollection"
        or not isinstance(payload.get("features"), list)
    ):
        raise ValueError(f"Invalid prediction geometry: {path}")
    return payload


def save_prediction(payload: dict, path: Path | str) -> None:
    path = Path(path)
    if path.suffix.casefold() == ".xml":
        save_asap_xml(payload, path)
    elif path.suffix.casefold() == ".geojson":
        write_json_atomic(path, payload)
    else:
        raise ValueError(f"Unsupported prediction extension: {path.suffix}")

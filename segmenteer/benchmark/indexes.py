"""Disk indexes for inference-only runner artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"


@dataclass(frozen=True)
class CompletedArtifact:
    run_id: str
    image_stem: str
    method_dir: Path
    prediction_path: Path
    config: dict
    geojson: dict


def iter_completed_artifacts(output_dir: Path | str) -> Iterator[CompletedArtifact]:
    """Yield valid prediction/config pairs from one runner experiment."""
    root = Path(output_dir)
    if not root.is_dir():
        return
    for method_dir in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        predictions = method_dir / PREDICTIONS_DIR
        if not method_dir.is_dir() or not predictions.is_dir():
            continue
        try:
            config = yaml.safe_load((method_dir / CONFIG_FILE).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            config = {}
        if not isinstance(config, dict):
            config = {}
        for prediction_path in sorted(predictions.glob("*.geojson"), key=lambda path: path.name.casefold()):
            try:
                geojson = json.loads(prediction_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
                continue
            yield CompletedArtifact(
                run_id=method_dir.name,
                image_stem=prediction_path.stem,
                method_dir=method_dir,
                prediction_path=prediction_path,
                config=config,
                geojson=geojson,
            )

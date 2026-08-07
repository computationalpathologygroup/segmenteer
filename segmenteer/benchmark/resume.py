"""Resume support for inference-only runner outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def image_fingerprint(image_path: Path | str) -> dict[str, int]:
    stat = Path(image_path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@dataclass(frozen=True)
class ResumeArtifact:
    geojson: dict


@dataclass
class ResumeStats:
    reused: int = 0
    invalid_artifacts: int = 0
    invalidated_methods: list[str] = field(default_factory=list)


class DatasetResumeStore:
    """Reuse predictions when method configuration and GeoJSON remain valid."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.stats = ResumeStats()

    def prepare(self, configurations: dict[str, dict]) -> None:
        for run_id, expected in configurations.items():
            method_dir = self.output_dir / run_id
            prediction_dir = method_dir / PREDICTIONS_DIR
            if not prediction_dir.exists():
                continue
            saved = self._read_yaml(method_dir / CONFIG_FILE)
            if saved is not None and fingerprint(saved) == fingerprint(expected):
                continue
            self._clear_method_outputs(method_dir)
            self.stats.invalidated_methods.append(run_id)

    def load_completed(
        self,
        *,
        run_id: str,
        image_path: Path,
        expected_config: dict,
        expected_method_name: str,
    ) -> tuple[ResumeArtifact | None, str]:
        del expected_method_name
        prediction_path = (
            self.output_dir / run_id / PREDICTIONS_DIR / f"{image_path.stem}.geojson"
        )
        if not prediction_path.exists():
            return None, "not yet written"

        prediction = self._read_json(prediction_path)
        if prediction is None or not self._valid_geojson(prediction):
            self.stats.invalid_artifacts += 1
            return None, "prediction GeoJSON is invalid"

        saved_config = self._read_yaml(self.output_dir / run_id / CONFIG_FILE)
        if saved_config is None or fingerprint(saved_config) != fingerprint(expected_config):
            self.stats.invalid_artifacts += 1
            return None, "segmenter configuration changed"

        self.stats.reused += 1
        return ResumeArtifact(prediction), "completed"

    def discard_partial(self, run_id: str, image_stem: str) -> None:
        try:
            (self.output_dir / run_id / PREDICTIONS_DIR / f"{image_stem}.geojson").unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _read_json(path: Path) -> Any | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _read_yaml(path: Path) -> dict | None:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _valid_geojson(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("type") == "FeatureCollection"
            and isinstance(payload.get("features"), list)
        )

    @staticmethod
    def _clear_method_outputs(method_dir: Path) -> None:
        predictions = method_dir / PREDICTIONS_DIR
        if predictions.exists():
            shutil.rmtree(predictions)
        try:
            (method_dir / CONFIG_FILE).unlink()
        except FileNotFoundError:
            pass

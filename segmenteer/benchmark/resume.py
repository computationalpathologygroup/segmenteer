from __future__ import annotations
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from segmenteer.io.predictions import (
    normalize_prediction_format,
    prediction_path,
    read_prediction,
)
from segmenteer.benchmark.timing import TIMINGS_DIR, load_timing, timing_path
from segmenteer.io.atomic import write_json_atomic
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
    timing: dict | None = None


@dataclass
class ResumeStats:
    reused: int = 0
    invalid_artifacts: int = 0
    invalidated_methods: list[str] = field(default_factory=list)


class DatasetResumeStore:
    def __init__(self, output_dir: Path | str, output_format: str = "geojson") -> None:
        self.output_dir = Path(output_dir)
        self.output_format = normalize_prediction_format(output_format)
        self.stats = ResumeStats()
        self._config_cache: dict[str, tuple[tuple[int, int], str | None]] = {}

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
        retain_geometry: bool = True,
        expected_prediction_protocol: str | None = None,
    ) -> tuple[ResumeArtifact | None, str]:
        del expected_method_name
        path = prediction_path(
            self.output_dir / run_id / PREDICTIONS_DIR,
            image_path.stem,
            self.output_format,
        )
        if not path.exists():
            return (None, "not yet written")
        try:
            prediction_stat = path.stat()
        except OSError:
            self.stats.invalid_artifacts += 1
            return (None, f"prediction {self.output_format} is unreadable")
        if prediction_stat.st_size == 0:
            self.stats.invalid_artifacts += 1
            return (None, f"prediction {self.output_format} is empty")
        config_path = self.output_dir / run_id / CONFIG_FILE
        try:
            stat = config_path.stat()
        except OSError:
            self.stats.invalid_artifacts += 1
            return (None, "segmenter configuration missing")
        config_signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._config_cache.get(run_id)
        if cached is None or cached[0] != config_signature:
            saved_config = self._read_yaml(config_path)
            saved_fingerprint = (
                fingerprint(saved_config) if saved_config is not None else None
            )
            self._config_cache[run_id] = (config_signature, saved_fingerprint)
        if self._config_cache[run_id][1] != fingerprint(expected_config):
            self.stats.invalid_artifacts += 1
            return (None, "segmenter configuration changed")
        timing = load_timing(
            self.output_dir,
            run_id=run_id,
            image_path=image_path,
            config=expected_config,
        )
        if timing is None:
            self.stats.invalid_artifacts += 1
            return (None, "missing or invalid benchmark timing")
        if (
            expected_prediction_protocol is not None
            and timing.get("prediction_protocol") != expected_prediction_protocol
        ):
            self.stats.invalid_artifacts += 1
            return (
                None,
                "outdated mask-to-polygon conversion; prediction must be regenerated",
            )
        prediction_signature = {
            "size_bytes": prediction_stat.st_size,
            "mtime_ns": prediction_stat.st_mtime_ns,
            "suffix": path.suffix.casefold(),
        }
        recorded_signature = timing.get("prediction_signature")
        if (
            recorded_signature is not None
            and recorded_signature != prediction_signature
        ):
            self.stats.invalid_artifacts += 1
            return (None, "prediction changed since completion")
        if retain_geometry or recorded_signature is None:
            try:
                prediction = read_prediction(path)
            except (OSError, UnicodeDecodeError, ValueError, TypeError):
                self.stats.invalid_artifacts += 1
                return (None, f"prediction {self.output_format} is invalid")
        else:
            prediction = {"type": "FeatureCollection", "features": []}
        if recorded_signature is None:
            timing["prediction_signature"] = prediction_signature
            write_json_atomic(
                timing_path(self.output_dir, run_id, image_path.stem), timing
            )
        self.stats.reused += 1
        return (ResumeArtifact(prediction, timing), "completed")

    def discard_partial(self, run_id: str, image_stem: str) -> None:
        try:
            prediction_path(
                self.output_dir / run_id / PREDICTIONS_DIR,
                image_stem,
                self.output_format,
            ).unlink()
        except FileNotFoundError:
            pass
        try:
            timing_path(self.output_dir, run_id, image_stem).unlink()
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
        timings = method_dir / TIMINGS_DIR
        if timings.exists():
            shutil.rmtree(timings)
        try:
            (method_dir / CONFIG_FILE).unlink()
        except FileNotFoundError:
            pass

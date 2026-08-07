"""Validation and reuse of completed dataset benchmark artifacts.

The resume layer deliberately treats an output as reusable only when its
prediction and score metadata agree with the current method configuration,
source image, and evaluation mode.  This prevents an interrupted run from
silently mixing stale results with current ones.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PREDICTIONS_DIR = "predictions"
EVAL_SCORES_DIR = "eval/scores"
EVAL_COMPLETED_DIR = "eval/completed"
CONFIG_FILE = "config.yaml"
CURRENT_ARTIFACT_SCHEMA_VERSION = 2


def canonical_json(data: Any) -> str:
    """Return a stable JSON representation suitable for equality checks."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(data: Any) -> str:
    """Return a SHA-256 fingerprint for plain JSON-compatible data."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def image_fingerprint(image_path: Path | str) -> dict[str, int]:
    """Return a lightweight source-image identity without reading WSI pixels."""
    stat = Path(image_path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@dataclass(frozen=True)
class ResumeArtifact:
    """A validated persisted result that can replace another model invocation."""

    geojson: dict
    score: dict


@dataclass
class ResumeStats:
    """Counters emitted at the end of a resumed dataset run."""

    reused: int = 0
    invalid_artifacts: int = 0
    invalidated_methods: list[str] = field(default_factory=list)


class DatasetResumeStore:
    """Read and validate per-slide benchmark outputs in an existing run folder.

    ``prepare`` performs run-level invalidation before any model is called.  If
    a method's ``config.yaml`` differs from the current segmenter config, all
    outputs for that method are removed so the resumed directory cannot contain
    a mixture of configurations.
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.stats = ResumeStats()
        self._expected_config_fingerprints: dict[str, str] = {}

    def prepare(self, configurations: dict[str, dict]) -> None:
        """Invalidate method directories whose saved configuration changed."""
        self._expected_config_fingerprints = {
            run_id: fingerprint(config) for run_id, config in configurations.items()
        }

        for run_id, expected_fp in self._expected_config_fingerprints.items():
            method_dir = self.output_dir / run_id
            if not method_dir.exists():
                continue

            has_outputs = any(
                (method_dir / relative).exists()
                for relative in (PREDICTIONS_DIR, EVAL_SCORES_DIR, EVAL_COMPLETED_DIR)
            )
            if not has_outputs:
                continue

            saved_config = self._read_yaml(method_dir / CONFIG_FILE)
            if saved_config is not None and fingerprint(saved_config) == expected_fp:
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
        ground_truth_geojson: dict | None,
    ) -> tuple[ResumeArtifact | None, str]:
        """Return a reusable result, or a reason that the method must run again."""
        image_stem = image_path.stem
        method_dir = self.output_dir / run_id
        prediction_path = method_dir / PREDICTIONS_DIR / f"{image_stem}.geojson"
        score_path = method_dir / EVAL_SCORES_DIR / f"{image_stem}.json"
        completion_path = method_dir / EVAL_COMPLETED_DIR / f"{image_stem}.json"

        exists = [prediction_path.exists(), score_path.exists(), completion_path.exists()]
        if not any(exists):
            return None, "not yet written"
        if not prediction_path.is_file() or not score_path.is_file():
            self.stats.invalid_artifacts += 1
            return None, "prediction or score is missing"

        prediction = self._read_json(prediction_path)
        score = self._read_json(score_path)
        if prediction is None or score is None:
            self.stats.invalid_artifacts += 1
            return None, "prediction or score JSON is invalid"
        if not self._valid_geojson(prediction):
            self.stats.invalid_artifacts += 1
            return None, "prediction is not a GeoJSON FeatureCollection"
        if not isinstance(score, dict):
            self.stats.invalid_artifacts += 1
            return None, "score JSON is not an object"

        if score.get("run_id") != run_id or score.get("image_stem") != image_stem:
            self.stats.invalid_artifacts += 1
            return None, "score metadata points to another method or slide"
        if score.get("method_name") != expected_method_name:
            self.stats.invalid_artifacts += 1
            return None, "method display name changed"

        # New lightweight outputs record GT availability but intentionally omit
        # metric payloads.  Accept the older ``supervised`` mode too, so a new
        # run can resume an already-evaluated legacy folder without rerunning a
        # valid prediction.
        expected_modes = (
            {"ground_truth_available", "supervised"}
            if ground_truth_geojson is not None
            else {"prediction_only"}
        )
        if score.get("evaluation_mode") not in expected_modes:
            self.stats.invalid_artifacts += 1
            return None, "ground-truth availability changed"

        expected_config_fp = fingerprint(expected_config)
        saved_config_fp = score.get("segmenter_config_fingerprint")
        if saved_config_fp is not None:
            if saved_config_fp != expected_config_fp:
                self.stats.invalid_artifacts += 1
                return None, "segmenter configuration changed"
        else:
            # Legacy v0.2.3 outputs did not include an item-level fingerprint.
            # Their method-level config remains a reliable compatibility check.
            saved_config = self._read_yaml(method_dir / CONFIG_FILE)
            if saved_config is None or fingerprint(saved_config) != expected_config_fp:
                self.stats.invalid_artifacts += 1
                return None, "legacy config is missing or changed"

        expected_gt_fp = fingerprint(ground_truth_geojson) if ground_truth_geojson is not None else None
        if "ground_truth_fingerprint" in score and score.get("ground_truth_fingerprint") != expected_gt_fp:
            self.stats.invalid_artifacts += 1
            return None, "ground truth changed"

        expected_image_fp = image_fingerprint(image_path)
        if "source_image_fingerprint" in score and score.get("source_image_fingerprint") != expected_image_fp:
            self.stats.invalid_artifacts += 1
            return None, "source image changed"

        try:
            artifact_schema_version = int(score.get("artifact_schema_version", 1))
        except (TypeError, ValueError):
            self.stats.invalid_artifacts += 1
            return None, "artifact schema version is invalid"

        if completion_path.exists():
            completion = self._read_json(completion_path)
            if (
                not isinstance(completion, dict)
                or completion.get("run_id") != run_id
                or completion.get("image_stem") != image_stem
                or completion.get("segmenter_config_fingerprint") != expected_config_fp
            ):
                self.stats.invalid_artifacts += 1
                return None, "completion marker is invalid"
        elif artifact_schema_version >= CURRENT_ARTIFACT_SCHEMA_VERSION:
            self.stats.invalid_artifacts += 1
            return None, "completion marker is missing"

        self.stats.reused += 1
        return ResumeArtifact(geojson=prediction, score=score), "completed"

    def discard_partial(self, run_id: str, image_stem: str) -> None:
        """Remove incomplete/corrupt artifacts before recomputing one result."""
        method_dir = self.output_dir / run_id
        for relative in (
            f"{PREDICTIONS_DIR}/{image_stem}.geojson",
            f"{EVAL_SCORES_DIR}/{image_stem}.json",
            f"{EVAL_COMPLETED_DIR}/{image_stem}.json",
            f"eval/heatmaps/{image_stem}.png",
            f"eval/errors/{image_stem}_true_positive.geojson",
            f"eval/errors/{image_stem}_false_positive.geojson",
            f"eval/errors/{image_stem}_false_negative.geojson",
            f"eval/error_maps/{image_stem}.png",
        ):
            path = method_dir / relative
            try:
                path.unlink()
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
        for relative in (PREDICTIONS_DIR, "eval"):
            path = method_dir / relative
            if path.exists():
                shutil.rmtree(path)
        try:
            (method_dir / CONFIG_FILE).unlink()
        except FileNotFoundError:
            pass

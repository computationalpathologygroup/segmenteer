"""Validation helpers for rebuilding shared dataset indexes from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PREDICTIONS_DIR = "predictions"
EVAL_SCORES_DIR = "eval/scores"
EVAL_COMPLETED_DIR = "eval/completed"
CURRENT_ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CompletedArtifact:
    """One fully committed, valid prediction/score pair on disk."""

    run_id: str
    image_stem: str
    method_dir: Path
    prediction_path: Path
    score_path: Path
    completion_path: Path
    score: dict


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _valid_geojson(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("type") == "FeatureCollection"
        and isinstance(payload.get("features"), list)
    )


def _completion_is_valid(
    completion_path: Path,
    *,
    run_id: str,
    image_stem: str,
    score: dict,
) -> bool:
    completion = _read_json(completion_path)
    if not isinstance(completion, dict):
        return False
    if completion.get("run_id") != run_id or completion.get("image_stem") != image_stem:
        return False
    if completion.get("segmenter_config_fingerprint") != score.get(
        "segmenter_config_fingerprint"
    ):
        return False
    return True


def iter_completed_artifacts(output_dir: Path | str) -> Iterator[CompletedArtifact]:
    """Yield committed dataset artifacts, skipping partial/corrupt files.

    Schema-v2 artifacts always require a valid completion marker.  Pre-v2
    artifacts remain readable for backwards-compatible resume/index rebuilds.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return

    for method_dir in sorted(output_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not method_dir.is_dir() or method_dir.name.startswith("."):
            continue
        predictions_dir = method_dir / PREDICTIONS_DIR
        scores_dir = method_dir / EVAL_SCORES_DIR
        if not predictions_dir.is_dir() or not scores_dir.is_dir():
            continue

        run_id = method_dir.name
        for score_path in sorted(scores_dir.glob("*.json"), key=lambda item: item.name.casefold()):
            image_stem = score_path.stem
            prediction_path = predictions_dir / f"{image_stem}.geojson"
            completion_path = method_dir / EVAL_COMPLETED_DIR / f"{image_stem}.json"
            if not prediction_path.is_file():
                continue

            score = _read_json(score_path)
            prediction = _read_json(prediction_path)
            if (
                not isinstance(score, dict)
                or not _valid_geojson(prediction)
                or score.get("run_id") != run_id
                or score.get("image_stem") != image_stem
            ):
                continue

            schema_version = score.get("artifact_schema_version", 1)
            try:
                schema_version = int(schema_version)
            except (TypeError, ValueError):
                continue

            if completion_path.exists():
                if not _completion_is_valid(
                    completion_path,
                    run_id=run_id,
                    image_stem=image_stem,
                    score=score,
                ):
                    continue
            elif schema_version >= CURRENT_ARTIFACT_SCHEMA_VERSION:
                # A score is not visible to a shared index until the writer has
                # committed the final marker after every dependent artifact.
                continue

            yield CompletedArtifact(
                run_id=run_id,
                image_stem=image_stem,
                method_dir=method_dir,
                prediction_path=prediction_path,
                score_path=score_path,
                completion_path=completion_path,
                score=score,
            )

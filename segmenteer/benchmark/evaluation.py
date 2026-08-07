"""Post-run evaluation of saved segmenteer GeoJSON outputs.

This module is intentionally separate from the benchmark runner.  Inference
writes prediction GeoJSON plus acquisition/run metadata quickly; this evaluator
can later derive supervised metrics from that saved geometry without rerunning
a model.  Pixel-space metrics are opt-in because they rasterise level-0 masks
and can be memory-intensive for whole-slide images.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from segmenteer.benchmark.ensemble import EVAL_SCORES_DIR, GROUND_TRUTH_DIR, PREDICTIONS_DIR
from segmenteer.benchmark.reporting import (
    METRICS_CSV_COLUMNS,
    SUPERVISED_METRIC_CSV_COLUMNS,
    format_native_spacing,
    format_number,
    write_metrics_csv_rows,
)
from segmenteer.metrics.supervised import compute_all_supervised_metrics

__all__ = ["EvaluationSummary", "evaluate_output_directory"]


@dataclass(frozen=True)
class EvaluationSummary:
    """Result of one post-run evaluator invocation."""

    output_path: Path
    total_predictions: int
    evaluated: int
    no_ground_truth: int
    vector_only: int
    failed: int


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _valid_feature_collection(data: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(data, dict)
        and data.get("type") == "FeatureCollection"
        and isinstance(data.get("features"), list)
    )


def _native_spacing_from_score(score: dict[str, Any]) -> tuple[float, float] | None:
    raw = score.get("native_spacing_um_per_px")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        x_mpp, y_mpp = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp)):
        return None
    return x_mpp, y_mpp


def _metadata_row(score: dict[str, Any], *, run_id: str, stem: str) -> dict[str, str]:
    image_path_raw = score.get("image_path")
    image_path = Path(str(image_path_raw)) if image_path_raw else None
    return {
        "filename": image_path.name if image_path is not None else stem,
        "native_spacing": format_native_spacing(_native_spacing_from_score(score)),
        "magnification_used": str(score.get("magnification_used") or ""),
        "segmentation_method": str(score.get("method_name") or run_id),
        "run_id": run_id,
        "execution_time_s": format_number(score.get("execution_time_s")),
        "seconds_per_megapixel": format_number(
            (float(score["seconds_per_pixel"]) * 1_000_000)
            if isinstance(score.get("seconds_per_pixel"), (int, float))
            else None
        ),
        "ground_truth_available": str(
            bool(score.get("ground_truth_available"))
            or bool((score.get("ground_truth") or {}).get("path"))
        ),
    }


def _empty_metric_cells() -> dict[str, str]:
    return {column: "" for column in SUPERVISED_METRIC_CSV_COLUMNS}


def _level0_shape(image_path: Path) -> tuple[int, int]:
    """Read dimensions only; pixel rasterisation happens later in metrics code."""
    from segmenteer.core.base import get_wsi_reader

    reader = get_wsi_reader()
    wsi = reader.read(str(image_path))
    width, height = reader.get_size(wsi, 0)
    return int(height), int(width)


def _metric_row(
    metadata: dict[str, str],
    *,
    prediction: dict[str, Any],
    ground_truth: dict[str, Any] | None,
    source_image: Path | None,
    pixel_metrics: bool,
) -> tuple[dict[str, str], str]:
    """Build one CSV row and return its terminal evaluation status."""
    row = {**metadata, **_empty_metric_cells(), "evaluation_status": "", "evaluation_error": ""}
    if ground_truth is None:
        row["evaluation_status"] = "no_ground_truth"
        return row, "no_ground_truth"

    image_shape = None
    status = "evaluated_vector"
    warning = ""
    if pixel_metrics:
        if source_image is None or not source_image.is_file():
            warning = "pixel metrics skipped: source WSI is unavailable"
        else:
            try:
                image_shape = _level0_shape(source_image)
                status = "evaluated_pixel"
            except Exception as exc:  # noqa: BLE001
                warning = f"pixel metrics skipped: {type(exc).__name__}: {exc}"

    try:
        metrics = compute_all_supervised_metrics(prediction, ground_truth, image_shape)
    except Exception as exc:  # noqa: BLE001
        row["evaluation_status"] = "failed"
        row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        return row, "failed"

    for field in (
        "dice",
        "iou",
        "hausdorff",
        "precision",
        "recall",
        "over_segmentation_rate",
        "under_segmentation_rate",
    ):
        row[field] = format_number(getattr(metrics, field))

    if metrics.pixel_metrics_computed:
        for field in ("pixel_accuracy", "mae", "balanced_error_rate"):
            row[field] = format_number(getattr(metrics, field))
        row["pixel_metrics_computed"] = "True"
    else:
        # Blank, rather than zero, distinguishes an intentionally uncomputed
        # pixel metric from a real zero-valued result.
        row["pixel_metrics_computed"] = "False"

    row["evaluation_status"] = status
    row["evaluation_error"] = warning
    return row, status


def evaluate_output_directory(
    output_dir: Path | str,
    *,
    pixel_metrics: bool = False,
    output_name: str = "metrics.csv",
) -> EvaluationSummary:
    """Evaluate saved predictions and write a root-level ``metrics.csv``.

    Parameters
    ----------
    output_dir:
        A completed or partially completed segmenteer output directory.
    pixel_metrics:
        When ``False`` (default), calculate only vector/GeoJSON supervised
        metrics.  When ``True``, open each source WSI to obtain level-0 shape
        and additionally rasterise masks for pixel accuracy, MAE, and balanced
        error rate.  This can require substantial memory for large slides.
    output_name:
        Name of the separate output CSV.  It defaults to ``metrics.csv`` and
        never overwrites the benchmark's lightweight ``results.csv``.
    """
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {root}")

    rows: list[dict[str, str]] = []
    total = evaluated = no_ground_truth = vector_only = failed = 0

    for method_dir in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not method_dir.is_dir() or method_dir.name == GROUND_TRUTH_DIR:
            continue
        predictions_dir = method_dir / PREDICTIONS_DIR
        if not predictions_dir.is_dir():
            continue

        run_id = method_dir.name
        scores_dir = method_dir / EVAL_SCORES_DIR
        for prediction_path in sorted(predictions_dir.glob("*.geojson"), key=lambda path: path.name.casefold()):
            total += 1
            stem = prediction_path.stem
            prediction = _read_json(prediction_path)
            score = _read_json(scores_dir / f"{stem}.json") or {}
            metadata = _metadata_row(score, run_id=run_id, stem=stem)

            if not _valid_feature_collection(prediction):
                row = {**metadata, **_empty_metric_cells()}
                row["evaluation_status"] = "failed"
                row["evaluation_error"] = "prediction is not a GeoJSON FeatureCollection"
                rows.append(row)
                failed += 1
                continue

            ground_truth_path = root / GROUND_TRUTH_DIR / f"{stem}.geojson"
            ground_truth = _read_json(ground_truth_path) if ground_truth_path.is_file() else None
            if ground_truth is not None and not _valid_feature_collection(ground_truth):
                ground_truth = None

            source_image_raw = score.get("image_path")
            source_image = Path(str(source_image_raw)) if source_image_raw else None
            row, status = _metric_row(
                metadata,
                prediction=prediction,
                ground_truth=ground_truth,
                source_image=source_image,
                pixel_metrics=pixel_metrics,
            )
            rows.append(row)
            if status == "no_ground_truth":
                no_ground_truth += 1
            elif status == "evaluated_vector":
                evaluated += 1
                vector_only += 1
            elif status == "evaluated_pixel":
                evaluated += 1
            else:
                failed += 1

    output_path = root / output_name
    write_metrics_csv_rows(rows, output_path)
    return EvaluationSummary(
        output_path=output_path,
        total_predictions=total,
        evaluated=evaluated,
        no_ground_truth=no_ground_truth,
        vector_only=vector_only,
        failed=failed,
    )

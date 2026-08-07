"""Lightweight benchmark indexes and separate evaluation-table helpers.

The benchmark path writes a compact root-level ``results.csv`` containing only
run metadata.  Quality metrics are intentionally deferred to
``evaluate_outputs.py``, which reads already-saved GeoJSON predictions and
writes a separate ``metrics.csv``.  This keeps model inference, resume, and
live output persistence independent from geometry-heavy evaluation work.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

from segmenteer.benchmark.runner import BenchmarkResult

# ``results.csv`` is deliberately a small, prediction-time index.  Keep the
# requested descriptive fields first for easy spreadsheet inspection.
DATASET_RESULTS_CSV_COLUMNS: tuple[str, ...] = (
    "filename",
    "native_spacing",
    "magnification_used",
    "segmentation_method",
    "run_id",
    "execution_time_s",
    "seconds_per_megapixel",
    "ground_truth_available",
)

# ``metrics.csv`` is a post-processing artifact.  It repeats the stable key and
# acquisition metadata from ``results.csv`` so it is independently useful and
# can be joined by ``(filename, run_id)`` without reading JSON files.
SUPERVISED_METRIC_CSV_COLUMNS: tuple[str, ...] = (
    "dice",
    "iou",
    "hausdorff",
    "precision",
    "recall",
    "over_segmentation_rate",
    "under_segmentation_rate",
    "pixel_accuracy",
    "mae",
    "balanced_error_rate",
    "pixel_metrics_computed",
)

METRICS_CSV_COLUMNS: tuple[str, ...] = (
    *DATASET_RESULTS_CSV_COLUMNS,
    *SUPERVISED_METRIC_CSV_COLUMNS,
    "evaluation_status",
    "evaluation_error",
)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(value) for value in obj]
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* so interrupted runs never expose a partial CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def format_number(value: object) -> str:
    """Return a stable CSV representation, preserving infinite Hausdorff values."""
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric):
            return ""
        if math.isinf(numeric):
            return "inf" if numeric > 0 else "-inf"
        return f"{numeric:.12g}"
    return str(value)


def format_native_spacing(native_spacing: tuple[float, float] | None) -> str:
    """Format level-0 physical pixel spacing with explicit micrometre units."""
    if native_spacing is None:
        return ""
    try:
        x_mpp, y_mpp = (float(native_spacing[0]), float(native_spacing[1]))
    except (TypeError, ValueError, IndexError):
        return ""
    if not all(math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp)):
        return ""
    return f"{x_mpp:g} x {y_mpp:g} µm/px"


def score_to_dataset_csv_row(score: Mapping[str, object]) -> dict[str, str]:
    """Convert persisted score metadata into the canonical root CSV row.

    Shared-output workers rebuild ``results.csv`` by scanning committed files,
    rather than merging stale in-memory process-local snapshots.
    """
    raw_image_path = score.get("image_path")
    filename = ""
    if raw_image_path:
        try:
            filename = Path(str(raw_image_path)).name
        except (TypeError, ValueError):
            filename = ""
    if not filename:
        filename = str(score.get("image_stem") or "")

    raw_spacing = score.get("native_spacing_um_per_px")
    native_spacing = None
    if isinstance(raw_spacing, (list, tuple)) and len(raw_spacing) == 2:
        try:
            native_spacing = (float(raw_spacing[0]), float(raw_spacing[1]))
        except (TypeError, ValueError):
            native_spacing = None

    try:
        seconds_per_pixel = float(score.get("seconds_per_pixel", 0.0))
    except (TypeError, ValueError):
        seconds_per_pixel = 0.0

    return {
        "filename": filename,
        "native_spacing": format_native_spacing(native_spacing),
        "magnification_used": str(score.get("magnification_used") or ""),
        "segmentation_method": str(score.get("method_name") or ""),
        "run_id": str(score.get("run_id") or ""),
        "execution_time_s": format_number(score.get("execution_time_s")),
        "seconds_per_megapixel": format_number(seconds_per_pixel * 1_000_000),
        "ground_truth_available": str(bool(score.get("ground_truth_available", False))),
    }


def result_to_dataset_csv_row(result: BenchmarkResult) -> dict[str, str]:
    """Convert one completed segmentation into the lightweight root CSV schema."""
    image_path = result.image_path
    return {
        "filename": image_path.name if image_path is not None else "",
        "native_spacing": format_native_spacing(result.native_spacing),
        "magnification_used": result.magnification_used or "",
        "segmentation_method": result.method_name,
        "run_id": result.run_id,
        "execution_time_s": format_number(result.execution_time),
        "seconds_per_megapixel": format_number(result.seconds_per_pixel * 1_000_000),
        "ground_truth_available": str(result.ground_truth_geojson is not None),
    }


def _normalise_rows(
    rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]
) -> list[dict[str, str]]:
    normalised_rows = [
        {
            column: "" if row.get(column) is None else str(row.get(column, ""))
            for column in fieldnames
        }
        for row in rows
    ]
    # Stable ordering makes diffs and resume-generated index files reproducible.
    normalised_rows.sort(
        key=lambda row: (
            row.get("filename", "").casefold(),
            row.get("run_id", "").casefold(),
        )
    )
    return normalised_rows


def write_csv_rows(
    rows: Iterable[Mapping[str, object]],
    output_path: Union[str, Path],
    *,
    fieldnames: Sequence[str],
) -> None:
    """Write normalised CSV rows atomically using the supplied schema."""
    from io import StringIO

    output_path = Path(output_path)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(_normalise_rows(rows, fieldnames))
    _write_text_atomic(output_path, buffer.getvalue())


def write_dataset_results_csv_rows(
    rows: Iterable[Mapping[str, object]], output_path: Union[str, Path]
) -> None:
    """Write root ``results.csv`` rows in the lightweight canonical schema."""
    write_csv_rows(rows, output_path, fieldnames=DATASET_RESULTS_CSV_COLUMNS)


def write_metrics_csv_rows(
    rows: Iterable[Mapping[str, object]], output_path: Union[str, Path]
) -> None:
    """Write post-run ``metrics.csv`` rows in the canonical evaluation schema."""
    write_csv_rows(rows, output_path, fieldnames=METRICS_CSV_COLUMNS)


def read_dataset_results_csv(output_path: Union[str, Path]) -> list[dict[str, str]]:
    """Read compatible existing root rows for live resume-safe CSV updates."""
    path = Path(output_path)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(DATASET_RESULTS_CSV_COLUMNS) - set(reader.fieldnames):
                return []
            return [
                {column: row.get(column, "") or "" for column in DATASET_RESULTS_CSV_COLUMNS}
                for row in reader
            ]
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def export_results_csv(results: Iterable[BenchmarkResult], output_path: Union[str, Path]) -> None:
    """Export one lightweight row for every successful result."""
    write_dataset_results_csv_rows(
        (result_to_dataset_csv_row(result) for result in results if not result.failed),
        output_path,
    )


def export_dataset_results_csv(
    all_results: Mapping[Path, Sequence[BenchmarkResult]],
    output_path: Union[str, Path],
) -> None:
    """Export root ``results.csv`` with one row for every image/method pair."""
    rows = (
        result_to_dataset_csv_row(result)
        for results in all_results.values()
        for result in results
        if not result.failed
    )
    write_dataset_results_csv_rows(rows, output_path)


def export_results_json(results: Iterable[BenchmarkResult], output_path: Union[str, Path]) -> None:
    """Export detailed metadata without forcing any metric computation."""
    output_path = Path(output_path)

    data = []
    for result in results:
        result_dict: dict[str, object] = {
            "filename": result.image_path.name if result.image_path is not None else None,
            "native_spacing_um_per_px": result.native_spacing,
            "magnification_used": result.magnification_used,
            "segmentation_method": result.method_name,
            "run_id": result.run_id,
            "execution_time": result.execution_time,
            "seconds_per_pixel": result.seconds_per_pixel,
            "ground_truth_available": result.ground_truth_geojson is not None,
        }

        # Preserve explicitly supplied legacy metrics without producing new ones.
        if result.unsupervised_metrics is not None:
            from dataclasses import asdict

            result_dict["unsupervised_metrics"] = asdict(result.unsupervised_metrics)
        if result.supervised_metrics is not None:
            from dataclasses import asdict

            result_dict["supervised_metrics"] = asdict(result.supervised_metrics)
        if result.error:
            result_dict["error"] = result.error

        data.append(result_dict)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(data), handle, indent=2)

"""Runner-only tabular reporting.

Evaluation tables live in :mod:`segmenteer.benchmark.evaluation`; this module
contains runner runtime metadata only.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np

from segmenteer.benchmark.runner import BenchmarkResult

DATASET_RESULTS_CSV_COLUMNS: tuple[str, ...] = (
    "filename",
    "native_spacing",
    "magnification_used",
    "segmentation_method",
    "run_id",
    "execution_time_s",
    "seconds_per_megapixel",
)


def _write_text_atomic(path: Path, text: str) -> None:
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
    if native_spacing is None:
        return ""
    try:
        x_mpp, y_mpp = float(native_spacing[0]), float(native_spacing[1])
    except (TypeError, ValueError, IndexError):
        return ""
    if not all(math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp)):
        return ""
    return f"{x_mpp:g} x {y_mpp:g} µm/px"


def result_to_dataset_csv_row(result: BenchmarkResult) -> dict[str, str]:
    return {
        "filename": result.image_path.name if result.image_path is not None else "",
        "native_spacing": format_native_spacing(result.native_spacing),
        "magnification_used": result.magnification_used or "",
        "segmentation_method": result.method_name,
        "run_id": result.run_id,
        "execution_time_s": format_number(result.execution_time),
        "seconds_per_megapixel": format_number(result.seconds_per_pixel * 1_000_000),
    }


def write_csv_rows(rows: Iterable[Mapping[str, object]], output_path: Union[str, Path], *, fieldnames: Sequence[str]) -> None:
    from io import StringIO

    normalised = [
        {column: "" if row.get(column) is None else str(row.get(column, "")) for column in fieldnames}
        for row in rows
    ]
    normalised.sort(key=lambda row: (row.get("filename", "").casefold(), row.get("run_id", "").casefold()))
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(normalised)
    _write_text_atomic(Path(output_path), buffer.getvalue())


def write_dataset_results_csv_rows(rows: Iterable[Mapping[str, object]], output_path: Union[str, Path]) -> None:
    write_csv_rows(rows, output_path, fieldnames=DATASET_RESULTS_CSV_COLUMNS)


def export_results_csv(results: Iterable[BenchmarkResult], output_path: Union[str, Path]) -> None:
    write_dataset_results_csv_rows((result_to_dataset_csv_row(result) for result in results if not result.failed), output_path)


def export_dataset_results_csv(all_results: Mapping[Path, Sequence[BenchmarkResult]], output_path: Union[str, Path]) -> None:
    export_results_csv((result for results in all_results.values() for result in results), output_path)


def export_results_json(results: Iterable[BenchmarkResult], output_path: Union[str, Path]) -> None:
    payload: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "filename": result.image_path.name if result.image_path is not None else None,
            "native_spacing_um_per_px": result.native_spacing,
            "magnification_used": result.magnification_used,
            "segmentation_method": result.method_name,
            "run_id": result.run_id,
            "execution_time": result.execution_time,
            "seconds_per_pixel": result.seconds_per_pixel,
        }
        if result.error:
            row["error"] = result.error
        payload.append(row)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

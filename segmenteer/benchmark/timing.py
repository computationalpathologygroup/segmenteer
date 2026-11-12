from __future__ import annotations
import csv
import io
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from segmenteer.io.atomic import write_json_atomic, write_text_atomic
from segmenteer.methods.identity import architecture_label

TIMINGS_DIR = "timings"
TIMINGS_CSV = "benchmark_timings.csv"
SUMMARY_CSV = "benchmark_summary.csv"
SCHEMA_VERSION = 3
DURATIONS = (
    "initialization_time_s",
    "algorithm_time_s",
    "total_processing_time_s",
    "segmentation_call_time_s",
    "image_loading_time_s",
    "preprocessing_time_s",
    "polygon_conversion_time_s",
    "serialization_time_s",
)
COLUMNS = (
    "filename",
    "segmentation_method",
    "run_id",
    *DURATIONS,
    "seconds_per_native_megapixel",
    "native_pixels",
    "native_spacing_um_per_px_x",
    "native_spacing_um_per_px_y",
    "magnification_used",
    "device",
    "measurement_phase",
    "measured_at_utc",
    "worker_pid",
    "timing_provenance",
    "timing_origin_prediction_format",
)
MEASURES = (
    "initialization_time_s",
    "algorithm_time_s",
    "total_processing_time_s",
    "seconds_per_native_megapixel",
)
SUMMARY_COLUMNS = (
    "segmentation_method",
    "run_id",
    "n_slides",
    "n_initializations",
    *[f"{name}_{suffix}" for name in MEASURES for suffix in ("mean", "std", "n")],
)


def timing_path(output_dir: Path | str, run_id: str, stem: str) -> Path:
    return Path(output_dir) / run_id / TIMINGS_DIR / f"{stem}.json"


def _nonnegative(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _read_record(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        return None
    if any(
        record.get(name) is not None and not _nonnegative(record[name])
        for name in (*DURATIONS, "seconds_per_native_megapixel")
    ):
        return None
    if not _nonnegative(record.get("total_processing_time_s")) or not _nonnegative(
        record.get("segmentation_call_time_s")
    ):
        return None
    if not isinstance(record.get("run_id"), str) or not isinstance(
        record.get("filename"), str
    ):
        return None
    return record


def persist_timing(
    output_dir: Path | str, result: Any, *, prediction_file: Path | None = None
) -> None:
    from segmenteer.benchmark.resume import fingerprint, image_fingerprint

    if (
        result.failed
        or result.image_path is None
        or result.total_processing_time_s is None
    ):
        return
    path = Path(result.image_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result.run_id,
        "prediction_protocol": result.prediction_protocol,
        "method_name": result.method_name,
        "filename": path.name,
        "config_fingerprint": fingerprint(result.segmenter_config),
        "image_fingerprint": image_fingerprint(path),
        **{name: getattr(result, name, None) for name in DURATIONS},
        "seconds_per_native_megapixel": result.seconds_per_pixel * 1000000
        if result.seconds_per_pixel is not None
        else None,
        "native_pixels": result.native_pixels,
        "native_spacing_um_per_px": result.native_spacing,
        "magnification_used": result.magnification_used,
        "device": result.device,
        "measurement_phase": result.measurement_phase,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_pid": os.getpid(),
    }
    if prediction_file is not None:
        prediction_stat = Path(prediction_file).stat()
        payload["prediction_signature"] = {
            "size_bytes": prediction_stat.st_size,
            "mtime_ns": prediction_stat.st_mtime_ns,
            "suffix": Path(prediction_file).suffix.casefold(),
        }
    write_json_atomic(timing_path(output_dir, result.run_id, path.stem), payload)


def load_timing(
    output_dir: Path | str, *, run_id: str, image_path: Path, config: dict
) -> dict | None:
    from segmenteer.benchmark.resume import fingerprint, image_fingerprint

    record = _read_record(timing_path(output_dir, run_id, image_path.stem))
    if (
        record is None
        or record["run_id"] != run_id
        or record["filename"] != image_path.name
    ):
        return None
    if record.get("config_fingerprint") != fingerprint(config):
        return None
    try:
        if record.get("image_fingerprint") != image_fingerprint(image_path):
            return None
    except OSError:
        return None
    return record


def collect_timings(output_dir: Path | str) -> list[dict]:
    root = Path(output_dir)
    records = []
    if not root.is_dir():
        raise NotADirectoryError(root)
    for method_dir in root.iterdir():
        prediction_dir = method_dir / "predictions"
        if not method_dir.is_dir() or not prediction_dir.is_dir():
            continue
        for prediction in prediction_dir.iterdir():
            if not prediction.is_file() or prediction.suffix.casefold() not in (
                ".geojson",
                ".xml",
            ):
                continue
            record = _read_record(timing_path(root, method_dir.name, prediction.stem))
            if record is not None and record["run_id"] == method_dir.name:
                records.append(record)
    records.sort(key=lambda record: (record["run_id"], record["filename"]))
    return records


def summary_rows(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["run_id"]].append(record)
    output = []
    for run_id, entries in sorted(grouped.items()):
        result = {
            "segmentation_method": architecture_label(
                run_id=run_id, formal_name=entries[0]["method_name"]
            ),
            "run_id": run_id,
            "n_slides": len(entries),
            "n_initializations": sum(
                item.get("initialization_time_s") is not None for item in entries
            ),
        }
        for name in MEASURES:
            values = [
                float(entry[name]) for entry in entries if _nonnegative(entry.get(name))
            ]
            result[f"{name}_n"] = len(values)
            result[f"{name}_mean"] = statistics.mean(values) if values else None
            result[f"{name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        output.append(result)
    return output


def _write_csv(path: Path, columns: tuple, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {key: "" if row.get(key) is None else row[key] for key in columns}
        )
    write_text_atomic(path, stream.getvalue())


def rebuild_timing_csv(output_dir: Path | str) -> list[dict]:
    root = Path(output_dir)
    records = collect_timings(root)
    detailed = []
    for record in records:
        spacing = record.get("native_spacing_um_per_px")
        detailed.append(
            {
                "filename": record["filename"],
                "segmentation_method": architecture_label(
                    run_id=record["run_id"], formal_name=record["method_name"]
                ),
                "run_id": record["run_id"],
                **{name: record.get(name) for name in DURATIONS},
                "seconds_per_native_megapixel": record.get(
                    "seconds_per_native_megapixel"
                ),
                "native_pixels": record.get("native_pixels"),
                "native_spacing_um_per_px_x": spacing[0]
                if isinstance(spacing, (list, tuple)) and len(spacing) == 2
                else None,
                "native_spacing_um_per_px_y": spacing[1]
                if isinstance(spacing, (list, tuple)) and len(spacing) == 2
                else None,
                "magnification_used": record.get("magnification_used"),
                "device": record.get("device"),
                "measurement_phase": record.get("measurement_phase"),
                "measured_at_utc": record.get("measured_at_utc"),
                "worker_pid": record.get("worker_pid"),
                "timing_provenance": record.get(
                    "timing_provenance", "direct_measurement"
                ),
                "timing_origin_prediction_format": record.get(
                    "timing_origin_prediction_format"
                ),
            }
        )
    summary = summary_rows(records)
    _write_csv(root / TIMINGS_CSV, COLUMNS, detailed)
    _write_csv(root / SUMMARY_CSV, SUMMARY_COLUMNS, summary)
    return summary

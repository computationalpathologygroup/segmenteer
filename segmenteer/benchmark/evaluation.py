from __future__ import annotations
import argparse
import csv
import hashlib
import json
from functools import lru_cache
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
import yaml
from segmenteer.methods.identity import architecture_label
from segmenteer.metrics.supervised import (
    calculate_supervised_metrics,
    prepare_reference,
    _positive_spacing,
)
from segmenteer.io.predictions import read_prediction
from segmenteer.io.ground_truth import resolve_ground_truth, ANNOTATION_EXTENSIONS
from segmenteer.io.atomic import write_text_atomic
from segmenteer.io.asap import load_asap_xml
from segmenteer.benchmark.evaluation_progress import EvaluationProgress
from segmenteer.benchmark.slide_identity import validate_unique_slide_stems
from segmenteer.benchmark.markdown_report import build_markdown_report
from rich.console import Console
from segmenteer.wsi.reader import NativeWSIReader

__all__ = ["EvaluationSummary", "evaluate_output_directory", "main"]
PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"
SQLITE_SCHEMA_VERSION = 5
EVALUATION_PROTOCOL_VERSION = 4
VECTOR_SUPERVISED_COLUMNS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "over_segmentation_rate",
    "under_segmentation_rate",
    "area_bias_pct",
    "predicted_area_mm2",
    "ground_truth_area_mm2",
    "false_positive_area_mm2",
    "false_negative_area_mm2",
    "nsd",
    "boundary_tolerance_um",
    "boundary_step_um",
    "prediction_outside_area_px2",
    "gt_outside_area_px2",
    "gt_slide_coverage_ratio",
    "gt_annotation_count",
    "gt_tissue_count",
    "gt_spline_count",
    "gt_hole_count",
    "gt_excluded_count",
    "native_mpp_x",
    "native_mpp_y",
    "evaluation_domain",
    "calibration_source",
    "evaluation_tissue_groups",
)
PUBLIC_COLUMNS = (
    "filename",
    "run_id",
    "method_name",
    "formal_name",
    "evaluation_mode",
    "dice_score",
    *VECTOR_SUPERVISED_COLUMNS,
    "oversegmentation",
    "undersegmentation",
    "evaluation_error",
    "computed_at_utc",
)


@dataclass(frozen=True)
class EvaluationSummary:
    output_path: Path
    output_format: str
    total_predictions: int
    supervised: int
    skipped: int
    failed: int
    reused: int = 0
    validation_path: Path | None = None
    markdown_path: Path | None = None
    bootstrap_summary_path: Path | None = None
    ground_truth_matching_path: Path | None = None


def _finite_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if isinstance(value, int):
        return int(value)
    return numeric


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GeoJSON root is not an object")
    if payload.get("type") != "FeatureCollection" or not isinstance(
        payload.get("features"), list
    ):
        raise ValueError("GeoJSON is not a FeatureCollection")
    return payload


def _method_name(method_dir: Path) -> str:
    try:
        payload = yaml.safe_load((method_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        payload = None
    if isinstance(payload, dict):
        for key in ("display_name", "name", "method_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return architecture_label(
                    formal_name=value.strip(), run_id=method_dir.name, config=payload
                )
        class_path = payload.get("class")
        if isinstance(class_path, str) and class_path.strip():
            return architecture_label(run_id=method_dir.name, config=payload)
    return architecture_label(run_id=method_dir.name)


def _discover_predictions(output_dir: Path) -> list[tuple[Path, Path, str, str]]:
    discovered: list[tuple[Path, Path, str, str]] = []
    for method_dir in sorted(
        output_dir.iterdir(), key=lambda path: path.name.casefold()
    ):
        predictions_dir = method_dir / PREDICTIONS_DIR
        if not method_dir.is_dir() or not predictions_dir.is_dir():
            continue
        method_name = _method_name(method_dir)
        for prediction in sorted(
            (
                item
                for item in predictions_dir.iterdir()
                if item.is_file() and item.suffix.casefold() in (".geojson", ".xml")
            ),
            key=lambda path: path.name.casefold(),
        ):
            discovered.append((method_dir, prediction, method_dir.name, method_name))
    discovered.sort(
        key=lambda item: (
            item[1].stem.casefold(),
            item[2].casefold(),
            item[1].suffix.casefold(),
        )
    )
    return discovered


def _match_ground_truth(
    root: Path, stem: str, suffix: str | None = None
) -> Path | None:
    return resolve_ground_truth(root, stem, suffix)


def _ground_truth_match_inventory(
    root: Path, predictions: list, suffix: str | None
) -> tuple[dict[str, Path | None], list[dict[str, str]]]:
    """Resolve annotation matches and report missing or ambiguous files."""
    unique_stems = {
        prediction.stem.casefold(): prediction.stem
        for _, prediction, _, _ in predictions
    }
    matched: dict[str, Path | None] = {}
    records: list[dict[str, str]] = []
    selected_files: set[Path] = set()
    for key, stem in sorted(unique_stems.items()):
        path = _match_ground_truth(root, stem, suffix)
        matched[key] = path
        if path is not None:
            selected_files.add(path.resolve())
        records.append(
            {
                "slide_stem": stem,
                "status": "matched" if path else "missing_annotation",
                "annotation_file": str(path) if path else "",
            }
        )
    for path in sorted(root.iterdir(), key=lambda p: p.name.casefold()):
        if (
            path.is_file()
            and path.suffix.casefold() in ANNOTATION_EXTENSIONS
            and path.resolve() not in selected_files
        ):
            records.append(
                {
                    "slide_stem": path.stem,
                    "status": "unmatched_annotation",
                    "annotation_file": str(path),
                }
            )
    return matched, records


def _ground_truth_csv_inventory(
    csv_path: Path, predictions: list
) -> tuple[dict[str, Path | None], list[dict[str, str]]]:
    """Map slide IDs to exact annotation files; never guess based on filenames.

    `slide_id` is the WSI stem (case-insensitive), not a full slide path.
    Relative ground-truth paths are resolved relative to annotations.csv.
    """
    selected: dict[str, tuple[str, Path]] = {}
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not {"slide_id", "ground_truth_annotation"} <= set(fields):
                raise ValueError(
                    "annotations.csv requires distinct slide_id and ground_truth_annotation columns"
                )
            for row_number, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f"Extra fields at annotations.csv row {row_number}.")
                identifier = (row.get("slide_id") or "").strip()
                raw_path = (row.get("ground_truth_annotation") or "").strip()
                if not identifier or not raw_path or Path(identifier).name != identifier or Path(identifier).stem != identifier:
                    raise ValueError(f"Invalid slide_id or annotation path in annotations.csv row {row_number}.")
                key = identifier.casefold()
                if key in selected:
                    raise ValueError(f"Duplicate annotations.csv slide_id {identifier!r} (row {row_number}).")
                annotation = Path(raw_path).expanduser()
                if not annotation.is_absolute():
                    annotation = csv_path.parent / annotation
                annotation = annotation.resolve()
                if annotation.suffix.casefold() not in ANNOTATION_EXTENSIONS:
                    raise ValueError(f"Unsupported annotation format in row {row_number}: {annotation}")
                if not annotation.is_file():
                    raise FileNotFoundError(f"Annotation not found in row {row_number}: {annotation}")
                selected[key] = (identifier, annotation)
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read annotation manifest {csv_path}: {exc}") from exc
    if not selected:
        raise ValueError(f"Annotation CSV contains no entries: {csv_path}")

    unique_stems = {
        prediction.stem.casefold(): prediction.stem
        for _, prediction, _, _ in predictions
    }
    matches: dict[str, Path | None] = {}
    rows: list[dict[str, str]] = []
    for key, stem in sorted(unique_stems.items()):
        annotation = selected[key][1] if key in selected else None
        matches[key] = annotation
        rows.append({
            "slide_stem": stem,
            "status": "matched" if annotation is not None else "missing_annotation",
            "annotation_file": str(annotation) if annotation else "",
        })
    for key, (identifier, annotation) in sorted(selected.items()):
        if key not in unique_stems:
            rows.append({
                "slide_stem": identifier,
                "status": "unmatched_annotation",
                "annotation_file": str(annotation),
            })
    return matches, rows


def _write_ground_truth_matching(path: Path, rows: list[dict[str, str]]) -> None:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=("slide_stem", "status", "annotation_file")
    )
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, stream.getvalue())


def _signature(path: Path | None) -> tuple[int, int]:
    if path is None or not path.is_file():
        return (-1, -1)
    stat = path.stat()
    return (int(stat.st_size), int(stat.st_mtime_ns))


def _empty_metrics() -> dict[str, Any]:
    return {column: None for column in VECTOR_SUPERVISED_COLUMNS}


def _evaluate_one(
    prediction_path: Path,
    ground_truth_path: Path | None,
    *,
    calibration=None,
    tolerance_um=None,
    step_um=10.0,
    tissue_groups=("tissue",),
    progress_stage=None,
    reference_cache=None,
) -> tuple[str, dict[str, Any]]:
    stage = progress_stage or (lambda name: None)
    metrics = _empty_metrics()
    if ground_truth_path is None:
        return ("skipped", metrics)
    stage("Reading prediction geometry")
    prediction = (
        load_asap_xml(prediction_path, groups=("tissue",))
        if prediction_path.suffix.casefold() == ".xml"
        else read_prediction(prediction_path)
    )
    if ground_truth_path is not None and ground_truth_path.is_file():
        native_mpp = calibration[0] if calibration is not None else None
        if tolerance_um is not None and native_mpp is None:
            raise ValueError(
                "NSD requires native MPP; supply --data or both --native-mpp-x and --native-mpp-y"
            )
        image_size = calibration[1] if calibration is not None else None
        key = (
            ground_truth_path,
            native_mpp,
            image_size,
            tuple(tissue_groups),
            tolerance_um,
            step_um if tolerance_um is not None else None,
        )
        cached = reference_cache.get(key) if reference_cache is not None else None
        if cached is None:
            stage("Reading and validating reference geometry")
            ground_truth = (
                load_asap_xml(ground_truth_path, groups=tissue_groups)
                if ground_truth_path.suffix.casefold() == ".xml"
                else (
                    _read_json(ground_truth_path)
                    if ground_truth_path.suffix.casefold() == ".json"
                    else read_prediction(ground_truth_path)
                )
            )
            reference = prepare_reference(
                ground_truth,
                mpp=native_mpp,
                image_size=image_size,
                boundary_tolerance_um=tolerance_um,
                boundary_step_um=step_um,
            )
            cached = ground_truth, reference
            if reference_cache is not None:
                reference_cache.clear()
                reference_cache[key] = cached
        ground_truth, reference = cached
        stage("Computing calibrated overlap")
        values = calculate_supervised_metrics(
            prediction,
            ground_truth,
            mpp=native_mpp,
            image_size=image_size,
            boundary_tolerance_um=tolerance_um,
            boundary_step_um=step_um,
            progress_stage=stage,
            prepared_reference=reference,
        )
        counts = ground_truth.get("segmenteer_annotation_counts", {})
        values.update(
            gt_annotation_count=counts.get("total"),
            gt_tissue_count=counts.get("tissue"),
            gt_spline_count=counts.get("spline"),
            gt_hole_count=counts.get("hole"),
            gt_excluded_count=counts.get("excluded"),
        )
        values["evaluation_domain"] = (
            "slide_bounds" if image_size is not None else "unbounded"
        )
        values["calibration_source"] = (
            calibration[2] if calibration is not None else "unavailable"
        )
        values["evaluation_tissue_groups"] = ",".join(tissue_groups)
        values["gt_slide_coverage_ratio"] = reference.coverage_ratio
        if native_mpp is not None:
            values["native_mpp_x"], values["native_mpp_y"] = native_mpp
        for column in VECTOR_SUPERVISED_COLUMNS:
            value = values.get(column)
            metrics[column] = (
                value if isinstance(value, str) else _finite_or_none(value)
            )
        return ("supervised", metrics)
    raise FileNotFoundError(
        f"Ground truth file disappeared for {prediction_path.stem!r}"
    )


def _row(
    *,
    filename: str,
    run_id: str,
    formal_name: str,
    mode: str,
    metrics: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    return {
        "filename": filename,
        "run_id": run_id,
        "method_name": run_id,
        "formal_name": formal_name,
        "evaluation_mode": mode,
        "dice_score": metrics.get("dice"),
        **metrics,
        "oversegmentation": metrics.get("over_segmentation_rate"),
        "undersegmentation": metrics.get("under_segmentation_rate"),
        "evaluation_error": error,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_csv_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PUBLIC_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        column: "" if row.get(column) is None else row.get(column)
                        for column in PUBLIC_COLUMNS
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _write_validation_report(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    names = (
        "filename",
        "run_id",
        "evaluation_mode",
        "evaluation_error",
        "gt_annotation_count",
        "gt_tissue_count",
        "gt_spline_count",
        "gt_hole_count",
        "gt_excluded_count",
        "evaluation_domain",
        "calibration_source",
        "evaluation_tissue_groups",
        "native_mpp_x",
        "native_mpp_y",
        "gt_slide_coverage_ratio",
        "gt_outside_area_px2",
        "prediction_outside_area_px2",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {key: "" if row.get(key) is None else row.get(key) for key in names}
                )
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        "\n        CREATE TABLE IF NOT EXISTS metrics (\n            filename TEXT NOT NULL,\n            run_id TEXT NOT NULL,\n            method_name TEXT NOT NULL,\n            formal_name TEXT,\n            evaluation_mode TEXT NOT NULL,\n            dice_score REAL,\n            dice REAL,\n            iou REAL,\n            precision REAL,\n            recall REAL,\n            over_segmentation_rate REAL,\n            under_segmentation_rate REAL,\n            oversegmentation REAL,\n            undersegmentation REAL,\n            evaluation_error TEXT NOT NULL,\n            computed_at_utc TEXT NOT NULL,\n            prediction_size INTEGER NOT NULL,\n            prediction_mtime_ns INTEGER NOT NULL,\n            ground_truth_size INTEGER NOT NULL,\n            ground_truth_mtime_ns INTEGER NOT NULL,\n            PRIMARY KEY (filename, run_id)\n        )\n        "
    )
    existing_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(metrics)")
    }
    if "hausdorff" in existing_columns:
        legacy = "metrics_schema_v1_with_hausdorff"
        connection.execute(f"DROP TABLE IF EXISTS {legacy}")
        connection.execute(f"ALTER TABLE metrics RENAME TO {legacy}")
        connection.execute(
            "\n            CREATE TABLE metrics (\n                filename TEXT NOT NULL,\n                run_id TEXT NOT NULL,\n                method_name TEXT NOT NULL,\n                formal_name TEXT,\n                evaluation_mode TEXT NOT NULL,\n                dice_score REAL,\n                dice REAL,\n                iou REAL,\n                precision REAL,\n                recall REAL,\n                over_segmentation_rate REAL,\n                under_segmentation_rate REAL,\n                oversegmentation REAL,\n                undersegmentation REAL,\n                evaluation_error TEXT NOT NULL,\n                computed_at_utc TEXT NOT NULL,\n                prediction_size INTEGER NOT NULL,\n                prediction_mtime_ns INTEGER NOT NULL,\n                ground_truth_size INTEGER NOT NULL,\n                ground_truth_mtime_ns INTEGER NOT NULL,\n                PRIMARY KEY (filename, run_id)\n            )\n            "
        )
        retained = [
            "filename",
            "run_id",
            "method_name",
            "formal_name",
            "evaluation_mode",
            "dice_score",
            "dice",
            "iou",
            "precision",
            "recall",
            "over_segmentation_rate",
            "under_segmentation_rate",
            "oversegmentation",
            "undersegmentation",
            "evaluation_error",
            "computed_at_utc",
            "prediction_size",
            "prediction_mtime_ns",
            "ground_truth_size",
            "ground_truth_mtime_ns",
        ]
        available_legacy = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({legacy})")
        }
        copy_columns = [column for column in retained if column in available_legacy]
        if copy_columns:
            names = ",".join(copy_columns)
            connection.execute(
                f"INSERT INTO metrics ({names}) SELECT {names} FROM {legacy}"
            )
        connection.execute(f"DROP TABLE {legacy}")
        existing_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(metrics)")
        }
    # Migrate retired metrics columns while retaining supervised records.
    retired_columns = (
        "num_objects",
        "total_area",
        "mean_area",
        "std_area",
        "median_area",
        "min_area",
        "max_area",
        "total_perimeter",
        "mean_perimeter",
        "mean_compactness",
        "mean_solidity",
        "coverage_ratio",
    )
    for column in retired_columns:
        if column in existing_columns:
            connection.execute(f"ALTER TABLE metrics DROP COLUMN {column}")
            existing_columns.remove(column)
    for column, sql_type in {
        "formal_name": "TEXT",
        "dice_score": "REAL",
        "oversegmentation": "REAL",
        "undersegmentation": "REAL",
        **{
            name: "TEXT"
            if name
            in {"evaluation_domain", "calibration_source", "evaluation_tissue_groups"}
            else "REAL"
            for name in VECTOR_SUPERVISED_COLUMNS
        },
    }.items():
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE metrics ADD COLUMN {column} {sql_type}")
    if "evaluation_fingerprint" not in existing_columns:
        connection.execute("ALTER TABLE metrics ADD COLUMN evaluation_fingerprint TEXT")
    if "hd95_um" in existing_columns:
        connection.execute("ALTER TABLE metrics DROP COLUMN hd95_um")
    connection.execute(
        "UPDATE metrics SET formal_name = COALESCE(formal_name, method_name)"
    )
    connection.execute(
        "UPDATE metrics SET method_name = run_id WHERE run_id IS NOT NULL AND run_id != ''"
    )
    connection.execute("UPDATE metrics SET dice_score = COALESCE(dice_score, dice)")
    connection.execute(
        "UPDATE metrics SET oversegmentation = COALESCE(oversegmentation, over_segmentation_rate)"
    )
    connection.execute(
        "UPDATE metrics SET undersegmentation = COALESCE(undersegmentation, under_segmentation_rate)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
        (str(SQLITE_SCHEMA_VERSION),),
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_filename ON metrics(filename)"
    )
    connection.commit()
    return connection


@lru_cache(maxsize=1)
def _metric_engine_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "core" / "utils.py",
        root / "metrics" / "supervised.py",
        root / "io" / "asap.py",
        root / "io" / "predictions.py",
        root / "io" / "ground_truth.py",
        root / "wsi" / "reader.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _evaluation_fingerprint(
    prediction: Path,
    ground_truth: Path | None,
    slide: Path | None,
    *,
    ground_truth_root: Path | None,
    data_root: Path | None,
    native_mpp: tuple[float, float] | None,
    boundary_tolerance_um: float | None,
    boundary_step_um: float,
    tissue_groups: tuple[str, ...],
    annotation_suffix: str | None = None,
    slide_signature: tuple[int, int] | None = None,
) -> str:
    def file_state(
        path: Path | None, signature: tuple[int, int] | None = None
    ) -> dict[str, Any] | None:
        if path is None:
            return None
        size, mtime_ns = signature if signature is not None else _signature(path)
        return {"path": str(path.resolve()), "size": size, "mtime_ns": mtime_ns}

    payload = {
        "protocol": EVALUATION_PROTOCOL_VERSION,
        "metric_engine": _metric_engine_fingerprint(),
        "prediction": file_state(prediction),
        "ground_truth": file_state(ground_truth),
        "ground_truth_root": str(ground_truth_root)
        if ground_truth_root is not None
        else None,
        "annotation_suffix": annotation_suffix,
        "slide": file_state(slide, slide_signature),
        "data_root": str(data_root) if data_root is not None else None,
        "native_mpp": native_mpp,
        "boundary_tolerance_um": boundary_tolerance_um,
        "boundary_step_um": boundary_step_um
        if boundary_tolerance_um is not None
        else None,
        "tissue_groups": sorted(
            {str(group).strip().casefold() for group in tissue_groups}
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sqlite_row_matches(
    existing: sqlite3.Row,
    pred_sig: tuple[int, int],
    gt_sig: tuple[int, int],
    fingerprint: str | None = None,
) -> bool:
    valid = (
        int(existing["prediction_size"]) == pred_sig[0]
        and int(existing["prediction_mtime_ns"]) == pred_sig[1]
        and int(existing["ground_truth_size"]) == gt_sig[0]
        and int(existing["ground_truth_mtime_ns"]) == gt_sig[1]
    )
    if fingerprint is not None:
        valid = (
            valid
            and existing["evaluation_fingerprint"] == fingerprint
            and existing["evaluation_mode"] in {"supervised", "skipped"}
        )
    return valid


def _upsert_sqlite(
    connection: sqlite3.Connection,
    row: dict[str, Any],
    pred_sig: tuple[int, int],
    gt_sig: tuple[int, int],
    fingerprint: str | None = None,
) -> None:
    columns = list(PUBLIC_COLUMNS) + [
        "prediction_size",
        "prediction_mtime_ns",
        "ground_truth_size",
        "ground_truth_mtime_ns",
        "evaluation_fingerprint",
    ]
    values = [row.get(column) for column in PUBLIC_COLUMNS] + [
        *pred_sig,
        *gt_sig,
        fingerprint,
    ]
    placeholders = ",".join(("?" for _ in columns))
    updates = ",".join(
        (
            f"{column}=excluded.{column}"
            for column in columns
            if column not in {"filename", "run_id"}
        )
    )
    connection.execute(
        f"INSERT INTO metrics ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT(filename, run_id) DO UPDATE SET {updates}",
        values,
    )
    connection.commit()


def evaluate_output_directory(
    output_dir: str | Path,
    *,
    ground_truth_dir: str | Path | None = None,
    annotations_csv: str | Path | None = None,
    annotation_suffix: str | None = None,
    output_format: str = "sqlite",
    metrics_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    native_mpp: tuple[float, float] | None = None,
    boundary_tolerance_um: float | None = None,
    boundary_step_um: float = 10.0,
    tissue_groups: Iterable[str] = ("tissue",),
    recompute: bool = False,
    validation_path: str | Path | None = None,
    show_progress: bool = False,
    progress_heartbeat_seconds: float = 15.0,
    report_markdown: bool = True,
    report_path: str | Path | None = None,
    bootstrap_resamples: int = 10000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 42,
    min_bootstrap_slides: int = 20,
) -> EvaluationSummary:
    if not math.isfinite(progress_heartbeat_seconds) or progress_heartbeat_seconds < 0:
        raise ValueError("Progress heartbeat interval must be finite and nonnegative")
    if report_markdown:
        if (
            bootstrap_resamples < 100
            or not 0 < bootstrap_confidence < 1
            or min_bootstrap_slides < 2
            or bootstrap_seed < 0
        ):
            raise ValueError(
                "Invalid bootstrap settings: resamples >= 100, 0 < confidence < 1, minimum slides >= 2, seed >= 0 are required"
            )
    if show_progress:
        Console(stderr=True).print(
            "Discovering completed prediction files...", highlight=False
        )
    output_root = Path(output_dir).expanduser().resolve()
    if not output_root.is_dir():
        raise NotADirectoryError(f"Runner output directory not found: {output_root}")
    if annotations_csv is not None and ground_truth_dir is not None:
        raise ValueError("Use either --annotations-csv or --ground-truth, not both.")
    if annotations_csv is not None and annotation_suffix is not None:
        raise ValueError("--annotation-suffix is only valid with --ground-truth.")
    annotations_manifest = (
        Path(annotations_csv).expanduser().resolve()
        if annotations_csv is not None else None
    )
    gt_root = (
        Path(ground_truth_dir).expanduser().resolve() if ground_truth_dir else None
    )
    if gt_root is not None and (not gt_root.is_dir()):
        raise NotADirectoryError(f"Ground-truth directory not found: {gt_root}")
    output_format = output_format.casefold()
    if output_format not in {"csv", "sqlite"}:
        raise ValueError("output_format must be 'csv' or 'sqlite'")
    default_name = "metrics.csv" if output_format == "csv" else "metrics.sqlite3"
    destination = (
        Path(metrics_path).expanduser().resolve()
        if metrics_path
        else output_root / default_name
    )
    predictions = _discover_predictions(output_root)
    manifest_path = output_root / "dataset_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid dataset manifest: {manifest_path}") from exc
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("slides"), list)
            or not all(isinstance(name, str) for name in manifest["slides"])
        ):
            raise ValueError(f"Invalid slide list in dataset manifest: {manifest_path}")
        validate_unique_slide_stems(
            manifest["slides"], context=f"evaluation of {manifest_path}"
        )
    if not predictions:
        raise FileNotFoundError(
            f"No <method>/predictions/*.geojson or *.xml files found under: {output_root}"
        )
    if gt_root is None and annotations_manifest is None:
        raise ValueError(
            "Supervised evaluation requires --ground-truth or --annotations-csv. "
            "Inference and benchmarking do not require reference annotations."
        )
    if annotations_manifest is not None:
        ground_truth_matches, match_rows = _ground_truth_csv_inventory(
            annotations_manifest, predictions
        )
    else:
        ground_truth_matches, match_rows = _ground_truth_match_inventory(
            gt_root, predictions, annotation_suffix
        )
    matching_path = output_root / "ground_truth_matching.csv"
    _write_ground_truth_matching(matching_path, match_rows)
    missing_count = sum(row["status"] == "missing_annotation" for row in match_rows)
    unused_count = sum(row["status"] == "unmatched_annotation" for row in match_rows)
    if missing_count or unused_count:
        Console(stderr=True).print(
            f"Ground-truth pairing: {missing_count} prediction slide(s) without annotations; {unused_count} annotation file(s) unused. Details: {matching_path}",
            style="yellow",
            highlight=False,
        )
    if boundary_tolerance_um is not None and (
        not math.isfinite(boundary_tolerance_um) or boundary_tolerance_um <= 0
    ):
        raise ValueError("Boundary tolerance must be positive and finite")
    if not math.isfinite(boundary_step_um) or boundary_step_um <= 0:
        raise ValueError("Boundary step must be positive and finite")
    if native_mpp is not None:
        native_mpp = _positive_spacing(native_mpp)
    slides = {}
    data_root = None
    tissue_groups = tuple(tissue_groups)
    if data_dir is not None:
        if show_progress:
            Console(stderr=True).print(
                "Indexing source slide files for native MPP and slide bounds...",
                highlight=False,
            )
        data_root = Path(data_dir).expanduser().resolve()
        if not data_root.is_dir():
            raise NotADirectoryError(data_root)
        slide_exts = {".svs", ".ndpi", ".mrxs", ".scn", ".qptiff", ".tif", ".tiff"}
        for slide in data_root.rglob("*"):
            if slide.is_file() and slide.suffix.lower() in slide_exts:
                key = slide.stem.casefold()
                if key in slides:
                    raise ValueError(
                        f"Ambiguous slide stem {slide.stem!r}: {slides[key]} and {slide}"
                    )
                slides[key] = slide
    calibration_cache = {}
    slide_signature_cache = {}
    reference_cache = {}
    reader = NativeWSIReader()

    def calibration_for(stem):
        key = stem.casefold()
        if key not in calibration_cache:
            size = None
            spacing = native_mpp
            if data_dir is not None:
                if key not in slides:
                    raise FileNotFoundError(
                        f"WSI for {stem!r} not found under {data_dir}"
                    )
                wsi = reader.read(slides[key])
                slide_size, slide_spacing = reader.get_size_and_mpp(
                    wsi, read_mpp=spacing is None
                )
                size = slide_size
                if spacing is None:
                    spacing = slide_spacing
            calibration_cache[key] = (
                (
                    spacing,
                    size,
                    "explicit_mpp" if native_mpp is not None else "wsi_metadata",
                )
                if spacing is not None or size is not None
                else None
            )
        return calibration_cache[key]

    supervised = skipped = failed = reused = 0
    cache_misses = {
        "new": 0,
        "changed_inputs": 0,
        "changed_protocol": 0,
        "retry_failed": 0,
        "forced": 0,
    }
    report_rows = []
    cache_path = (
        destination if output_format == "sqlite" else output_root / "metrics.sqlite3"
    )
    with EvaluationProgress(
        len(predictions),
        enabled=show_progress,
        heartbeat_seconds=progress_heartbeat_seconds,
    ) as progress:
        progress.finish_stage("Opening persistent evaluation cache")
        connection = _connect(cache_path)
        try:
            current_pairs = {
                (prediction.name, run_id) for _, prediction, run_id, _ in predictions
            }
            for db_row in connection.execute(
                "SELECT filename, run_id FROM metrics"
            ).fetchall():
                pair = (str(db_row["filename"]), str(db_row["run_id"]))
                if pair not in current_pairs:
                    connection.execute(
                        "DELETE FROM metrics WHERE filename=? AND run_id=?", pair
                    )
            connection.commit()
            for index, (_, prediction_path, run_id, method_name) in enumerate(
                predictions, start=1
            ):
                progress.begin(index, prediction_path, run_id)
                gt_path = None
                fingerprint = None
                try:
                    gt_path = ground_truth_matches[prediction_path.stem.casefold()]
                    pred_sig = _signature(prediction_path)
                    gt_sig = _signature(gt_path)
                    slide_path = (
                        slides.get(prediction_path.stem.casefold())
                        if data_root is not None
                        else None
                    )
                    if (
                        slide_path is not None
                        and slide_path not in slide_signature_cache
                    ):
                        slide_signature_cache[slide_path] = _signature(slide_path)
                    fingerprint = _evaluation_fingerprint(
                        prediction_path,
                        gt_path,
                        slide_path,
                        ground_truth_root=gt_root,
                        data_root=data_root,
                        native_mpp=native_mpp,
                        boundary_tolerance_um=boundary_tolerance_um,
                        boundary_step_um=boundary_step_um,
                        tissue_groups=tissue_groups,
                        annotation_suffix=annotation_suffix,
                        slide_signature=slide_signature_cache.get(slide_path),
                    )
                    existing = connection.execute(
                        "SELECT * FROM metrics WHERE filename=? AND run_id=?",
                        (prediction_path.name, run_id),
                    ).fetchone()
                    if (
                        not recompute
                        and existing is not None
                        and _sqlite_row_matches(existing, pred_sig, gt_sig, fingerprint)
                    ):
                        reused += 1
                        mode = str(existing["evaluation_mode"])
                        supervised += mode == "supervised"
                        skipped += mode == "skipped"
                        progress.advance("reused")
                        continue
                    if recompute:
                        cache_misses["forced"] += 1
                    elif existing is None:
                        cache_misses["new"] += 1
                    elif existing["evaluation_mode"] == "failed":
                        cache_misses["retry_failed"] += 1
                    elif not _sqlite_row_matches(existing, pred_sig, gt_sig):
                        cache_misses["changed_inputs"] += 1
                    else:
                        cache_misses["changed_protocol"] += 1
                    progress.stage("Reading slide calibration")
                    calibration = (
                        calibration_for(prediction_path.stem)
                        if gt_path is not None
                        else None
                    )
                    mode, metrics = _evaluate_one(
                        prediction_path,
                        gt_path,
                        calibration=calibration,
                        tolerance_um=boundary_tolerance_um,
                        step_um=boundary_step_um,
                        tissue_groups=tissue_groups,
                        progress_stage=progress.stage,
                        reference_cache=reference_cache,
                    )
                    supervised += mode == "supervised"
                    skipped += mode == "skipped"
                    if (
                        _signature(prediction_path) != pred_sig
                        or _signature(gt_path) != gt_sig
                    ):
                        raise RuntimeError(
                            "Prediction or ground truth changed during evaluation; retry this result"
                        )
                    result = _row(
                        filename=prediction_path.name,
                        run_id=run_id,
                        formal_name=method_name,
                        mode=mode,
                        metrics=metrics,
                    )
                except Exception as exc:
                    failed += 1
                    error = f"{type(exc).__name__}: {exc}"
                    result = _row(
                        filename=prediction_path.name,
                        run_id=run_id,
                        formal_name=method_name,
                        mode="failed",
                        metrics=_empty_metrics(),
                        error=error,
                    )
                progress.stage("Saving evaluation result")
                _upsert_sqlite(
                    connection,
                    result,
                    _signature(prediction_path),
                    _signature(gt_path),
                    fingerprint if result["evaluation_mode"] != "failed" else None,
                )
                progress.advance(
                    result["evaluation_mode"],
                    error=f"{prediction_path.name}: {result['evaluation_error']}"
                    if result["evaluation_error"]
                    else "",
                )
            progress.finish_stage("Collecting all evaluation results")
            report_rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM metrics ORDER BY run_id COLLATE NOCASE, filename COLLATE NOCASE"
                )
            ]
            # Resolve display labels without modifying cached metric records.
            labels_by_run = {rid: display for _, _, rid, display in predictions}
            for row in report_rows:
                row["formal_name"] = labels_by_run.get(
                    row["run_id"],
                    architecture_label(
                        run_id=row["run_id"], formal_name=row.get("formal_name", "")
                    ),
                )

        finally:
            connection.close()
        if output_format == "csv":
            progress.finish_stage("Exporting consolidated metrics CSV")
            _write_csv_atomic(destination, report_rows)
        report = (
            Path(validation_path).expanduser().resolve()
            if validation_path is not None
            else (
                output_root / "evaluation_validation.csv"
                if gt_root is not None or annotations_manifest is not None
                else None
            )
        )
        if report is not None:
            progress.finish_stage("Writing validation report")
            _write_validation_report(report, report_rows)
    if show_progress:
        detail = ", ".join(
            f"{name.replace('_', ' ')}={count}"
            for name, count in cache_misses.items()
            if count
        )
        Console(stderr=True).print(
            f"Evaluation cache: reused={reused}"
            + (f", {detail}" if detail else ", no recalculations"),
            highlight=False,
        )
    markdown = None
    bootstrap_summary = None
    if report_markdown:
        markdown = (
            Path(report_path).expanduser().resolve()
            if report_path is not None
            else output_root / "benchmark_report.md"
        )
        if show_progress:
            Console(stderr=True).print(
                "Generating Markdown benchmark report and bootstrap intervals...",
                highlight=False,
            )
        bootstrap_summary = output_root / "bootstrap_summary.csv"
        build_markdown_report(
            report_rows,
            output_path=markdown,
            metrics_path=destination,
            validation_path=report,
            total_slides=len(slides) if data_dir is not None else None,
            output_format=output_format,
            ground_truth_dir=gt_root or annotations_manifest,
            data_dir=data_dir,
            boundary_tolerance_um=boundary_tolerance_um,
            boundary_step_um=boundary_step_um,
            tissue_groups=tissue_groups,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=bootstrap_seed,
            min_bootstrap_slides=min_bootstrap_slides,
            bootstrap_summary_path=bootstrap_summary,
            ground_truth_matching_path=matching_path,
        )
    return EvaluationSummary(
        output_path=destination,
        output_format=output_format,
        total_predictions=len(predictions),
        supervised=supervised,
        skipped=skipped,
        failed=failed,
        reused=reused,
        validation_path=report,
        markdown_path=markdown,
        bootstrap_summary_path=bootstrap_summary,
        ground_truth_matching_path=matching_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Segmenteer runner predictions independently of inference."
    )
    parser.add_argument(
        "--output", "--output-dir",
        required=True,
        metavar="DIR",
        help="Runner experiment directory containing <method>/predictions/.",
    )
    parser.add_argument(
        "--ground-truth",
        metavar="DIR",
        help="GeoJSON, JSON, or ASAP XML ground-truth directory.",
    )
    parser.add_argument(
        "--annotations-csv",
        metavar="FILE",
        help="CSV mapping slide_id to ground_truth_annotation; alternative to --ground-truth.",
    )
    parser.add_argument(
        "--annotation-suffix",
        default=None,
        metavar="SUFFIX",
        help="Optional annotation naming suffix, e.g. _gt, _gt.geojson or _manual.xml; by default accept both <stem> and <stem>_gt.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "sqlite"),
        default="sqlite",
        dest="output_format",
        help="Metrics storage format.",
    )
    parser.add_argument(
        "--metrics",
        metavar="PATH",
        help="Output metrics file. Defaults to <output>/metrics.csv or metrics.sqlite3.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recalculate all predictions, ignoring saved evaluation results.",
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        help="Original level-0 WSIs for native X/Y MPP and slide bounds.",
    )
    parser.add_argument(
        "--native-mpp-x",
        type=float,
        help="Explicit native X MPP when scanner metadata is unavailable.",
    )
    parser.add_argument(
        "--native-mpp-y",
        type=float,
        help="Explicit native Y MPP when scanner metadata is unavailable.",
    )
    parser.add_argument(
        "--boundary-tolerance-um",
        type=float,
        help="Predeclared physical boundary tolerance for NSD.",
    )
    parser.add_argument(
        "--boundary-step-um",
        type=float,
        default=10.0,
        help="Maximum physical sampling interval for NSD.",
    )
    parser.add_argument(
        "--tissue-groups",
        default="tissue",
        help="Comma-separated ASAP annotation groups to count as tissue.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable evaluation progress reporting.",
    )
    parser.add_argument(
        "--progress-heartbeat-seconds",
        type=float,
        default=15.0,
        help="Heartbeat interval for redirected logs when an evaluation is still running.",
    )
    parser.add_argument(
        "--validation-report",
        metavar="PATH",
        help="Validation CSV path, default <output>/evaluation_validation.csv when ground truth is supplied.",
    )
    parser.add_argument(
        "--report-path",
        metavar="PATH",
        help="Markdown benchmark report path; defaults to <output>/benchmark_report.md.",
    )
    parser.add_argument(
        "--no-report-markdown",
        action="store_true",
        help="Disable automatic Markdown report and bootstrap intervals.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Paired slide bootstrap replicates for aggregate confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for percentile bootstrap intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Reproducible bootstrap random seed.",
    )
    parser.add_argument(
        "--min-bootstrap-slides",
        type=int,
        default=20,
        help="Minimum complete paired slides for bootstrap intervals.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if (args.native_mpp_x is None) != (args.native_mpp_y is None):
        raise ValueError("Specify both --native-mpp-x and --native-mpp-y")
    native_mpp = (
        (args.native_mpp_x, args.native_mpp_y)
        if args.native_mpp_x is not None
        else None
    )
    groups = tuple(
        value.strip() for value in args.tissue_groups.split(",") if value.strip()
    )
    if not groups:
        raise ValueError("--tissue-groups must not be empty")
    summary = evaluate_output_directory(
        args.output,
        ground_truth_dir=args.ground_truth,
        annotations_csv=args.annotations_csv,
        annotation_suffix=args.annotation_suffix,
        output_format=args.output_format,
        metrics_path=args.metrics,
        data_dir=args.data,
        native_mpp=native_mpp,
        boundary_tolerance_um=args.boundary_tolerance_um,
        boundary_step_um=args.boundary_step_um,
        tissue_groups=groups,
        recompute=args.recompute,
        validation_path=args.validation_report,
        show_progress=not args.no_progress,
        progress_heartbeat_seconds=args.progress_heartbeat_seconds,
        report_markdown=not args.no_report_markdown,
        report_path=args.report_path,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
        min_bootstrap_slides=args.min_bootstrap_slides,
    )
    print(f"Metrics: {summary.output_path}")
    print(
        f"Predictions={summary.total_predictions}, supervised={summary.supervised}, skipped={summary.skipped}, failed={summary.failed}, reused={summary.reused}"
    )
    if summary.validation_path is not None:
        print(f"Validation: {summary.validation_path}")
    if summary.ground_truth_matching_path is not None:
        print(f"Ground-truth matching: {summary.ground_truth_matching_path}")
    if summary.markdown_path is not None:
        print(f"Markdown report: {summary.markdown_path}")
    if summary.bootstrap_summary_path is not None:
        print(f"Bootstrap summary: {summary.bootstrap_summary_path}")


if __name__ == "__main__":
    main()

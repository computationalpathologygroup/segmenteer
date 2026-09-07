"""Independent post-run evaluator for Segmenteer predictions.

The evaluator consumes runner output; it never participates in inference.
With a ground-truth directory it computes supervised polygon metrics. Without
matching ground truth it computes reference-free structural metrics from the
prediction itself. Results are written to either one CSV file or one SQLite
file, selected explicitly by the caller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from segmenteer.metrics.supervised import (
    compute_dice,
    compute_iou,
    compute_over_segmentation_rate,
    compute_precision,
    compute_recall,
    compute_under_segmentation_rate,
)
from segmenteer.metrics.unsupervised import compute_unsupervised_metrics

__all__ = ["EvaluationSummary", "evaluate_output_directory", "main"]

PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"
SQLITE_SCHEMA_VERSION = 2

VECTOR_SUPERVISED_COLUMNS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "over_segmentation_rate",
    "under_segmentation_rate",
)
UNSUPERVISED_COLUMNS = (
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
    *UNSUPERVISED_COLUMNS,
    "evaluation_error",
    "computed_at_utc",
)


@dataclass(frozen=True)
class EvaluationSummary:
    output_path: Path
    output_format: str
    total_predictions: int
    supervised: int
    unsupervised: int
    failed: int
    reused: int = 0


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
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
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
                return value.strip()
        class_path = payload.get("class")
        if isinstance(class_path, str) and class_path.strip():
            return class_path.rsplit(".", 1)[-1]
    return method_dir.name


def _discover_predictions(output_dir: Path) -> list[tuple[Path, Path, str, str]]:
    discovered: list[tuple[Path, Path, str, str]] = []
    for method_dir in sorted(output_dir.iterdir(), key=lambda path: path.name.casefold()):
        predictions_dir = method_dir / PREDICTIONS_DIR
        if not method_dir.is_dir() or not predictions_dir.is_dir():
            continue
        method_name = _method_name(method_dir)
        for prediction in sorted(predictions_dir.glob("*.geojson"), key=lambda path: path.name.casefold()):
            discovered.append((method_dir, prediction, method_dir.name, method_name))
    return discovered


def _signature(path: Path | None) -> tuple[int, int]:
    if path is None or not path.is_file():
        return -1, -1
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _empty_metrics() -> dict[str, Any]:
    return {column: None for column in (*VECTOR_SUPERVISED_COLUMNS, *UNSUPERVISED_COLUMNS)}


def _evaluate_one(prediction_path: Path, ground_truth_path: Path | None) -> tuple[str, dict[str, Any]]:
    prediction = _read_json(prediction_path)
    metrics = _empty_metrics()
    if ground_truth_path is not None and ground_truth_path.is_file():
        ground_truth = _read_json(ground_truth_path)
        values = {
            "dice": compute_dice(prediction, ground_truth),
            "iou": compute_iou(prediction, ground_truth),
            "precision": compute_precision(prediction, ground_truth),
            "recall": compute_recall(prediction, ground_truth),
            "over_segmentation_rate": compute_over_segmentation_rate(prediction, ground_truth),
            "under_segmentation_rate": compute_under_segmentation_rate(prediction, ground_truth),
        }
        for column in VECTOR_SUPERVISED_COLUMNS:
            metrics[column] = _finite_or_none(values.get(column))
        return "supervised", metrics

    unsupervised = asdict(compute_unsupervised_metrics(prediction, image_area=0.0))
    for column in UNSUPERVISED_COLUMNS:
        # Coverage requires the source image area; runner outputs intentionally
        # do not contain WSI pixels/dimensions for evaluator use, so do not
        # fabricate it from prediction bounds.
        metrics[column] = None if column == "coverage_ratio" else _finite_or_none(unsupervised.get(column))
    return "unsupervised", metrics


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
        # Compatibility with the supplied official evaluator: method_name is
        # the prediction-directory ID and formal_name is the display label.
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
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PUBLIC_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: "" if row.get(column) is None else row.get(column) for column in PUBLIC_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            filename TEXT NOT NULL,
            run_id TEXT NOT NULL,
            method_name TEXT NOT NULL,
            formal_name TEXT,
            evaluation_mode TEXT NOT NULL,
            dice_score REAL,
            dice REAL,
            iou REAL,
            precision REAL,
            recall REAL,
            over_segmentation_rate REAL,
            under_segmentation_rate REAL,
            oversegmentation REAL,
            undersegmentation REAL,
            num_objects INTEGER,
            total_area REAL,
            mean_area REAL,
            std_area REAL,
            median_area REAL,
            min_area REAL,
            max_area REAL,
            total_perimeter REAL,
            mean_perimeter REAL,
            mean_compactness REAL,
            mean_solidity REAL,
            coverage_ratio REAL,
            evaluation_error TEXT NOT NULL,
            computed_at_utc TEXT NOT NULL,
            prediction_size INTEGER NOT NULL,
            prediction_mtime_ns INTEGER NOT NULL,
            ground_truth_size INTEGER NOT NULL,
            ground_truth_mtime_ns INTEGER NOT NULL,
            PRIMARY KEY (filename, run_id)
        )
        """
    )
    existing_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(metrics)")}
    if "hausdorff" in existing_columns:
        # Schema v2 removes Hausdorff entirely. Rebuild the table so an existing
        # resumable database is migrated rather than silently retaining that column.
        legacy = "metrics_schema_v1_with_hausdorff"
        connection.execute(f"DROP TABLE IF EXISTS {legacy}")
        connection.execute(f"ALTER TABLE metrics RENAME TO {legacy}")
        connection.execute(
            """
            CREATE TABLE metrics (
                filename TEXT NOT NULL,
                run_id TEXT NOT NULL,
                method_name TEXT NOT NULL,
                formal_name TEXT,
                evaluation_mode TEXT NOT NULL,
                dice_score REAL,
                dice REAL,
                iou REAL,
                precision REAL,
                recall REAL,
                over_segmentation_rate REAL,
                under_segmentation_rate REAL,
                oversegmentation REAL,
                undersegmentation REAL,
                num_objects INTEGER,
                total_area REAL,
                mean_area REAL,
                std_area REAL,
                median_area REAL,
                min_area REAL,
                max_area REAL,
                total_perimeter REAL,
                mean_perimeter REAL,
                mean_compactness REAL,
                mean_solidity REAL,
                coverage_ratio REAL,
                evaluation_error TEXT NOT NULL,
                computed_at_utc TEXT NOT NULL,
                prediction_size INTEGER NOT NULL,
                prediction_mtime_ns INTEGER NOT NULL,
                ground_truth_size INTEGER NOT NULL,
                ground_truth_mtime_ns INTEGER NOT NULL,
                PRIMARY KEY (filename, run_id)
            )
            """
        )
        retained = [
            "filename", "run_id", "method_name", "formal_name", "evaluation_mode",
            "dice_score", "dice", "iou", "precision", "recall",
            "over_segmentation_rate", "under_segmentation_rate",
            "oversegmentation", "undersegmentation", "num_objects", "total_area",
            "mean_area", "std_area", "median_area", "min_area", "max_area",
            "total_perimeter", "mean_perimeter", "mean_compactness", "mean_solidity",
            "coverage_ratio", "evaluation_error", "computed_at_utc",
            "prediction_size", "prediction_mtime_ns", "ground_truth_size",
            "ground_truth_mtime_ns",
        ]
        available_legacy = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({legacy})")}
        copy_columns = [column for column in retained if column in available_legacy]
        if copy_columns:
            names = ",".join(copy_columns)
            connection.execute(f"INSERT INTO metrics ({names}) SELECT {names} FROM {legacy}")
        connection.execute(f"DROP TABLE {legacy}")
        existing_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(metrics)")}
    for column, sql_type in {
        "formal_name": "TEXT",
        "dice_score": "REAL",
        "oversegmentation": "REAL",
        "undersegmentation": "REAL",
    }.items():
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE metrics ADD COLUMN {column} {sql_type}")
    connection.execute("UPDATE metrics SET formal_name = COALESCE(formal_name, method_name)")
    connection.execute("UPDATE metrics SET method_name = run_id WHERE run_id IS NOT NULL AND run_id != ''")
    connection.execute("UPDATE metrics SET dice_score = COALESCE(dice_score, dice)")
    connection.execute("UPDATE metrics SET oversegmentation = COALESCE(oversegmentation, over_segmentation_rate)")
    connection.execute("UPDATE metrics SET undersegmentation = COALESCE(undersegmentation, under_segmentation_rate)")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)", (str(SQLITE_SCHEMA_VERSION),))
    connection.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_metrics_filename ON metrics(filename)")
    connection.commit()
    return connection


def _sqlite_row_matches(existing: sqlite3.Row, pred_sig: tuple[int, int], gt_sig: tuple[int, int]) -> bool:
    return (
        int(existing["prediction_size"]) == pred_sig[0]
        and int(existing["prediction_mtime_ns"]) == pred_sig[1]
        and int(existing["ground_truth_size"]) == gt_sig[0]
        and int(existing["ground_truth_mtime_ns"]) == gt_sig[1]
    )


def _upsert_sqlite(
    connection: sqlite3.Connection,
    row: dict[str, Any],
    pred_sig: tuple[int, int],
    gt_sig: tuple[int, int],
) -> None:
    columns = list(PUBLIC_COLUMNS) + ["prediction_size", "prediction_mtime_ns", "ground_truth_size", "ground_truth_mtime_ns"]
    values = [row.get(column) for column in PUBLIC_COLUMNS] + [*pred_sig, *gt_sig]
    placeholders = ",".join("?" for _ in columns)
    updates = ",".join(f"{column}=excluded.{column}" for column in columns if column not in {"filename", "run_id"})
    connection.execute(
        f"INSERT INTO metrics ({','.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(filename, run_id) DO UPDATE SET {updates}",
        values,
    )
    connection.commit()


def evaluate_output_directory(
    output_dir: str | Path,
    *,
    ground_truth_dir: str | Path | None = None,
    output_format: str = "sqlite",
    metrics_path: str | Path | None = None,
) -> EvaluationSummary:
    """Evaluate all ``<method>/predictions/*.geojson`` under *output_dir*.

    ``ground_truth_dir`` is optional. Matching ``<stem>.geojson`` files enable
    supervised metrics; predictions without a matching reference are evaluated
    with unsupervised structural metrics instead.
    """
    output_root = Path(output_dir).expanduser().resolve()
    if not output_root.is_dir():
        raise NotADirectoryError(f"Runner output directory not found: {output_root}")
    gt_root = Path(ground_truth_dir).expanduser().resolve() if ground_truth_dir else None
    if gt_root is not None and not gt_root.is_dir():
        raise NotADirectoryError(f"Ground-truth directory not found: {gt_root}")

    output_format = output_format.casefold()
    if output_format not in {"csv", "sqlite"}:
        raise ValueError("output_format must be 'csv' or 'sqlite'")
    default_name = "metrics.csv" if output_format == "csv" else "metrics.sqlite3"
    destination = Path(metrics_path).expanduser().resolve() if metrics_path else output_root / default_name

    predictions = _discover_predictions(output_root)
    if not predictions:
        raise FileNotFoundError(f"No <method>/predictions/*.geojson files found under: {output_root}")

    supervised = unsupervised = failed = reused = 0

    if output_format == "csv":
        rows: list[dict[str, Any]] = []
        for _, prediction_path, run_id, method_name in predictions:
            gt_path = gt_root / prediction_path.name if gt_root is not None else None
            try:
                mode, metrics = _evaluate_one(prediction_path, gt_path)
                supervised += mode == "supervised"
                unsupervised += mode == "unsupervised"
                rows.append(_row(filename=prediction_path.name, run_id=run_id, formal_name=method_name, mode=mode, metrics=metrics))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                rows.append(_row(
                    filename=prediction_path.name,
                    run_id=run_id,
                    formal_name=method_name,
                    mode="failed",
                    metrics=_empty_metrics(),
                    error=f"{type(exc).__name__}: {exc}",
                ))
        rows.sort(key=lambda item: (str(item["run_id"]).casefold(), str(item["filename"]).casefold()))
        _write_csv_atomic(destination, rows)
    else:
        connection = _connect(destination)
        try:
            current_pairs = {(prediction.name, run_id) for _, prediction, run_id, _ in predictions}
            for db_row in connection.execute("SELECT filename, run_id FROM metrics").fetchall():
                pair = (str(db_row["filename"]), str(db_row["run_id"]))
                if pair not in current_pairs:
                    connection.execute("DELETE FROM metrics WHERE filename=? AND run_id=?", pair)
            connection.commit()

            for _, prediction_path, run_id, method_name in predictions:
                gt_path = gt_root / prediction_path.name if gt_root is not None else None
                pred_sig = _signature(prediction_path)
                gt_sig = _signature(gt_path)
                existing = connection.execute(
                    "SELECT * FROM metrics WHERE filename=? AND run_id=?",
                    (prediction_path.name, run_id),
                ).fetchone()
                if existing is not None and _sqlite_row_matches(existing, pred_sig, gt_sig):
                    reused += 1
                    mode = str(existing["evaluation_mode"])
                    supervised += mode == "supervised"
                    unsupervised += mode == "unsupervised"
                    failed += mode == "failed"
                    continue
                try:
                    mode, metrics = _evaluate_one(prediction_path, gt_path)
                    supervised += mode == "supervised"
                    unsupervised += mode == "unsupervised"
                    result = _row(filename=prediction_path.name, run_id=run_id, formal_name=method_name, mode=mode, metrics=metrics)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    result = _row(
                        filename=prediction_path.name,
                        run_id=run_id,
                        formal_name=method_name,
                        mode="failed",
                        metrics=_empty_metrics(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                _upsert_sqlite(connection, result, pred_sig, gt_sig)
        finally:
            connection.close()

    return EvaluationSummary(
        output_path=destination,
        output_format=output_format,
        total_predictions=len(predictions),
        supervised=supervised,
        unsupervised=unsupervised,
        failed=failed,
        reused=reused,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Segmenteer runner predictions independently of inference.")
    parser.add_argument("--output", required=True, metavar="DIR", help="Runner experiment directory containing <method>/predictions/.")
    parser.add_argument("--ground-truth", metavar="DIR", help="Optional GeoJSON ground-truth directory.")
    parser.add_argument("--format", choices=("csv", "sqlite"), default="sqlite", dest="output_format", help="Metrics storage format.")
    parser.add_argument("--metrics", metavar="PATH", help="Output metrics file. Defaults to <output>/metrics.csv or metrics.sqlite3.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = evaluate_output_directory(
        args.output,
        ground_truth_dir=args.ground_truth,
        output_format=args.output_format,
        metrics_path=args.metrics,
    )
    print(f"Metrics: {summary.output_path}")
    print(
        f"Predictions={summary.total_predictions}, supervised={summary.supervised}, "
        f"unsupervised={summary.unsupervised}, failed={summary.failed}, reused={summary.reused}"
    )


if __name__ == "__main__":
    main()

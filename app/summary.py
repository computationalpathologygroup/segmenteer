"""Official-results aggregate metrics for the dedicated viewer.

This summary is intentionally narrower than the generic segmenteer viewer:
only official evaluation metrics are reported, and the official CSV cohort is
treated as the dataset definition.  Prediction morphology, pixel metrics,
runtime metrics, and Hausdorff distance are omitted from the UI summaries.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from app.loader import IndexData


METRIC_SPECS: tuple[dict[str, str], ...] = (
    {"key": "dice", "label": "Dice", "group": "Evaluation", "source": "supervised"},
    {"key": "iou", "label": "IoU", "group": "Evaluation", "source": "supervised"},
    {"key": "precision", "label": "Precision", "group": "Evaluation", "source": "supervised"},
    {"key": "recall", "label": "Recall", "group": "Evaluation", "source": "supervised"},
    {"key": "over_segmentation_rate", "label": "Over-segmentation", "group": "Evaluation", "source": "supervised"},
    {"key": "under_segmentation_rate", "label": "Under-segmentation", "group": "Evaluation", "source": "supervised"},
)


@dataclass(frozen=True)
class SummaryCell:
    method_name: str
    run_id: str
    metric_key: str
    metric_label: str
    metric_group: str
    n: int
    mean: float | None
    standard_deviation: float | None


@dataclass(frozen=True)
class DatasetCounts:
    slides_in_dataset: int
    slides_discovered: int
    ground_truth_available_slides: int
    prediction_only_slides: int
    skipped_no_ground_truth: int
    not_run_slides: int


def dataset_counts(index: IndexData) -> DatasetCounts:
    slides = len({wsi.stem for wsi in index.wsis})
    # Retain generic fields for backwards compatibility with older frontend code,
    # but the official UI displays only slides_in_dataset.
    return DatasetCounts(
        slides_in_dataset=slides,
        slides_discovered=slides,
        ground_truth_available_slides=slides,
        prediction_only_slides=0,
        skipped_no_ground_truth=0,
        not_run_slides=0,
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(entry: dict[str, Any], key: str) -> float | None:
    supervised = entry.get("metrics", {}).get("supervised", {})
    if not isinstance(supervised, dict):
        return None
    return _finite_number(supervised.get(key))


def supervised_summary(index: IndexData) -> list[SummaryCell]:
    stems = [wsi.stem for wsi in index.wsis]
    rows: list[SummaryCell] = []
    for method in index.methods.values():
        for spec in METRIC_SPECS:
            values: list[float] = []
            for stem in stems:
                entry = index.scores.get(stem, {}).get(method.run_id)
                if not isinstance(entry, dict):
                    continue
                value = _metric_value(entry, spec["key"])
                if value is not None:
                    values.append(value)
            n = len(values)
            rows.append(
                SummaryCell(
                    method_name=method.name,
                    run_id=method.run_id,
                    metric_key=spec["key"],
                    metric_label=spec["label"],
                    metric_group=spec["group"],
                    n=n,
                    mean=statistics.fmean(values) if values else None,
                    standard_deviation=statistics.stdev(values) if n >= 2 else None,
                )
            )
    return rows


def summary_payload(index: IndexData) -> dict[str, Any]:
    return {
        "scope": "official evaluation cohort",
        "standard_deviation": "sample SD (n - 1); blank when n < 2",
        "counts": asdict(dataset_counts(index)),
        "rows": [asdict(row) for row in supervised_summary(index)],
    }


def summary_csv(index: IndexData) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "scope",
        "method_name",
        "run_id",
        "metric_group",
        "metric",
        "n_evaluated_slides",
        "mean",
        "sample_standard_deviation",
    ])
    for row in supervised_summary(index):
        writer.writerow([
            "official evaluation cohort",
            row.method_name,
            row.run_id,
            row.metric_group,
            row.metric_label,
            row.n,
            "" if row.mean is None else f"{row.mean:.12g}",
            "" if row.standard_deviation is None else f"{row.standard_deviation:.12g}",
        ])
    return output.getvalue()

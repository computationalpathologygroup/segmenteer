"""Aggregate optional evaluator metrics for the output-first viewer."""

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
    {"key": "num_objects", "label": "Objects", "group": "Reference-free", "source": "unsupervised"},
    {"key": "total_area", "label": "Total area", "group": "Reference-free", "source": "unsupervised"},
    {"key": "mean_area", "label": "Mean area", "group": "Reference-free", "source": "unsupervised"},
    {"key": "mean_compactness", "label": "Compactness", "group": "Reference-free", "source": "unsupervised"},
    {"key": "mean_solidity", "label": "Solidity", "group": "Reference-free", "source": "unsupervised"},
    {"key": "coverage_ratio", "label": "Coverage", "group": "Reference-free", "source": "unsupervised"},
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
    supervised_slides: int
    unsupervised_slides: int


def dataset_counts(index: IndexData) -> DatasetCounts:
    wsis = index.wsis
    return DatasetCounts(
        slides_in_dataset=len(wsis),
        slides_discovered=len(wsis),
        ground_truth_available_slides=sum(wsi.has_ground_truth for wsi in wsis),
        prediction_only_slides=sum(wsi.run_status == "prediction_only" for wsi in wsis),
        supervised_slides=sum(wsi.run_status == "supervised" for wsi in wsis),
        unsupervised_slides=sum(wsi.run_status == "unsupervised" for wsi in wsis),
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_summary(index: IndexData) -> list[SummaryCell]:
    stems = [wsi.stem for wsi in index.wsis]
    rows: list[SummaryCell] = []
    for method in index.methods.values():
        for spec in METRIC_SPECS:
            values: list[float] = []
            for stem in stems:
                entry = index.scores.get(stem, {}).get(method.run_id)
                if not isinstance(entry, dict):
                    continue
                group = entry.get("metrics", {}).get(spec["source"], {})
                if not isinstance(group, dict):
                    continue
                value = _finite_number(group.get(spec["key"]))
                if value is not None:
                    values.append(value)
            if not values:
                continue
            n = len(values)
            rows.append(
                SummaryCell(
                    method_name=method.label,
                    run_id=method.run_id,
                    metric_key=spec["key"],
                    metric_label=spec["label"],
                    metric_group=spec["group"],
                    n=n,
                    mean=statistics.fmean(values),
                    standard_deviation=statistics.stdev(values) if n >= 2 else None,
                )
            )
    return rows


def summary_payload(index: IndexData) -> dict[str, Any]:
    return {
        "scope": "viewer metrics file",
        "standard_deviation": "sample SD (n - 1); blank when n < 2",
        "counts": asdict(dataset_counts(index)),
        "rows": [asdict(row) for row in metric_summary(index)],
    }


def summary_csv(index: IndexData) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "scope", "method_name", "run_id", "metric_group", "metric",
        "n_slides", "mean", "sample_standard_deviation",
    ])
    for row in metric_summary(index):
        writer.writerow([
            "viewer metrics file", row.method_name, row.run_id, row.metric_group,
            row.metric_label, row.n,
            "" if row.mean is None else f"{row.mean:.12g}",
            "" if row.standard_deviation is None else f"{row.standard_deviation:.12g}",
        ])
    return output.getvalue()

from __future__ import annotations
import math
import csv
import hashlib
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from segmenteer.methods.identity import architecture_label

METRIC_LABELS = (
    ("dice", "Dice", "↑"),
    ("iou", "IoU", "↑"),
    ("nsd", "NSD", "↑"),
    ("precision", "Precision", "↑"),
    ("recall", "Recall", "↑"),
    ("false_positive_area_mm2", "FP area (mm²)", "↓"),
    ("false_negative_area_mm2", "FN area (mm²)", "↓"),
    ("area_bias_pct", "Area bias (%)", "→ 0"),
)


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _label(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ").replace("\r", " ").strip()


def _cell(value: float | None, decimals: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}"


def _mean_ci(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    confidence: float,
    rng: np.random.Generator,
    replications: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(values, axis=0)
    results = np.empty((replications, values.shape[1]), dtype=np.float64)
    block_size = max(1, min(128, 1048576 // max(1, values.shape[0] * values.shape[1])))
    for start in range(0, replications, block_size):
        stop = min(start + block_size, replications)
        sampled = rng.integers(0, len(indices), size=(stop - start, len(indices)))
        results[start:stop] = np.mean(values[sampled], axis=1)
    alpha = (1 - confidence) / 2
    return (
        means,
        np.quantile(results, alpha, axis=0),
        np.quantile(results, 1 - alpha, axis=0),
    )


def _statistics(
    records: list[dict[str, Any]],
    *,
    replications: int,
    confidence: float,
    seed: int,
    min_slides: int,
) -> dict[str, Any]:
    methods = sorted({str(row["run_id"]) for row in records})
    paired = defaultdict(dict)
    labels = {}
    duplicates = set()
    for row in records:
        method = str(row["run_id"])
        labels[method] = str(row.get("formal_name") or method)
        if str(row.get("evaluation_mode")) == "supervised" and not str(
            row.get("evaluation_error") or ""
        ):
            slide = Path(str(row["filename"])).stem.casefold()
            if method in paired[slide]:
                duplicates.add((slide, method))
                paired[slide][method] = None
            else:
                paired[slide][method] = row
    counts = {
        method: sum(
            method in scores and scores[method] is not None
            for scores in paired.values()
        )
        for method in methods
    }
    metrics = {}
    for key, title, direction in METRIC_LABELS:
        shared_slides = sorted(
            slide
            for slide, by_method in paired.items()
            if all(
                method in by_method
                and by_method[method] is not None
                and _number(by_method[method].get(key)) is not None
                for method in methods
            )
        )
        points = {}
        lower = {}
        upper = {}
        sd = {}
        median = {}
        q25 = {}
        q75 = {}
        n_by_method = {}
        eligible_by_method = {}
        for method in methods:
            values = np.asarray(
                [
                    value
                    for slide in sorted(paired)
                    if (record := paired[slide].get(method)) is not None
                    and (value := _number(record.get(key))) is not None
                ],
                dtype=np.float64,
            )
            count = len(values)
            n_by_method[method] = count
            eligible = count >= min_slides
            eligible_by_method[method] = eligible
            points[method] = float(np.mean(values)) if count else float("nan")
            sd[method] = float(np.std(values, ddof=1)) if count > 1 else float("nan")
            median[method] = float(np.median(values)) if count else float("nan")
            q25[method] = float(np.quantile(values, 0.25)) if count else float("nan")
            q75[method] = float(np.quantile(values, 0.75)) if count else float("nan")
            lower[method] = float("nan")
            upper[method] = float("nan")
            if eligible:
                method_seed = int.from_bytes(
                    hashlib.sha256(f"{key}:{method}:{seed}".encode("utf-8")).digest()[
                        :8
                    ],
                    "big",
                )
                _, lower_ci, upper_ci = _mean_ci(
                    values[:, None],
                    np.arange(count),
                    confidence=confidence,
                    rng=np.random.default_rng(method_seed),
                    replications=replications,
                )
                lower[method] = float(lower_ci[0])
                upper[method] = float(upper_ci[0])
        metrics[key] = {
            "label": title,
            "direction": direction,
            "n": len(shared_slides),
            "n_by_method": n_by_method,
            "eligible_by_method": eligible_by_method,
            "points": points,
            "lower": lower,
            "upper": upper,
            "sd": sd,
            "median": median,
            "q25": q25,
            "q75": q75,
            "differences": [],
        }
    return {
        "methods": methods,
        "labels": labels,
        "counts": counts,
        "metrics": metrics,
        "slides_with_predictions": len(paired),
        "duplicate_slide_method_pairs": len(duplicates),
    }


def build_markdown_report(
    rows: Iterable[dict[str, Any]],
    *,
    output_path: str | Path,
    metrics_path: str | Path,
    validation_path: str | Path | None = None,
    ground_truth_matching_path: str | Path | None = None,
    total_slides: int | None = None,
    output_format: str = "csv",
    ground_truth_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    boundary_tolerance_um: float | None = None,
    boundary_step_um: float = 10.0,
    tissue_groups: Iterable[str] = ("tissue",),
    bootstrap_resamples: int = 10000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 42,
    min_bootstrap_slides: int = 20,
    bootstrap_summary_path: str | Path | None = None,
) -> Path:
    if bootstrap_resamples < 100:
        raise ValueError("Bootstrap repetitions must be at least 100")
    if not 0 < bootstrap_confidence < 1:
        raise ValueError("Bootstrap confidence level must lie strictly between 0 and 1")
    if min_bootstrap_slides < 2:
        raise ValueError("Minimum slides for bootstrap must be at least 2")
    if bootstrap_seed < 0:
        raise ValueError("Bootstrap seed must be nonnegative")
    rows = list(rows)
    destination = Path(output_path).expanduser().resolve()
    metrics_path = Path(metrics_path).expanduser().resolve()
    validation_path = (
        Path(validation_path).expanduser().resolve()
        if validation_path is not None
        else None
    )
    result = _statistics(
        rows,
        replications=bootstrap_resamples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
        min_slides=min_bootstrap_slides,
    )
    bootstrap_summary = (
        Path(bootstrap_summary_path).expanduser().resolve()
        if bootstrap_summary_path is not None
        else metrics_path.parent / "bootstrap_summary.csv"
    )
    methods = result["methods"]
    counts = result["counts"]
    status_counts = Counter(str(row.get("evaluation_mode") or "") for row in rows)
    slide_ids = {Path(str(row["filename"])).stem.casefold() for row in rows}
    observed_pairs = len(
        {
            (Path(str(row["filename"])).stem.casefold(), str(row["run_id"]))
            for row in rows
        }
    )
    potential = (total_slides if total_slides is not None else len(slide_ids)) * len(
        methods
    )
    partial = (
        observed_pairs < potential
        or status_counts["failed"] > 0
        or status_counts["skipped"] > 0
    )
    if methods and result["metrics"]["dice"]["n"] < len(slide_ids):
        partial = True
    if result["duplicate_slide_method_pairs"]:
        partial = True
    descriptor = (
        "INCOMPLETE / PROVISIONAL" if partial else "COMPLETE AMONG DISCOVERED METHODS"
    )
    ci_text = f"{bootstrap_confidence:.0%}"
    lines = [
        "# Segmenteer tissue segmentation benchmark",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Status: {descriptor}.**",
        "",
        "## 1. Evaluation coverage",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
        f"| Predictions discovered | {len(rows):,} |",
        f"| Distinct predicted slides | {len(slide_ids):,} |",
        f"| Slide files indexed | {f'{total_slides:,}' if total_slides is not None else 'Not supplied'} |",
        f"| Represented method configurations | {len(methods):,} |",
        f"| Supervised evaluations | {status_counts['supervised']:,} |",
        f"| Skipped (no ground truth) | {status_counts['skipped']:,} |",
        f"| Failed evaluations | {status_counts['failed']:,} |",
        f"| Ambiguous duplicate slide–method results | {result['duplicate_slide_method_pairs']:,} |",
        f"| Potential pairs (indexed slides × represented configurations) | {potential:,} |",
        f"| Paired complete-case slides for Dice | {result['metrics']['dice']['n']:,} |",
        "",
        "The potential-pair count describes coverage of method configurations found in these results. It is not a verified run manifest; old or not-yet-started method configurations may change the intended denominator.",
        "",
    ]
    if partial:
        lines += [
            "**Incomplete data warning:** missing predictions, missing supervised scores, or failures may bias the results. Rankings and method comparisons are exploratory until the intended cohort is complete and exclusions are reviewed.",
            "",
        ]
    lines += [
        "## 2. Primary overlap performance",
        "",
        "Each method is summarized using all of its own valid evaluated slides for each metric. Confidence intervals are not estimated for methods with fewer than "
        + str(min_bootstrap_slides)
        + " valid slides. Bootstrap intervals describe uncertainty in the macro-average slide score under the assumption of independent slides. Because methods can have different evaluated cohorts during a partial run, comparisons of their aggregate means are provisional until evaluation coverage is matched.",
        "",
        "| Method configuration | Evaluated slides | Dice mean "
        + ci_text
        + " CI ↑ | IoU mean ↑ | NSD mean ↑ | Precision mean ↑ | Recall mean ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def display_name(method: str) -> str:
        cfg = method.split("__cfg=")[-1] if "__cfg=" in method else None
        return _label(
            architecture_label(formal_name=result["labels"][method], run_id=method)
        ) + (f" (cfg `{cfg}`)" if cfg else "")

    dice_stats = result["metrics"]["dice"]
    rank_methods = sorted(
        methods,
        key=lambda method: (
            -dice_stats["points"][method]
            if math.isfinite(dice_stats["points"][method])
            else math.inf,
            method,
        ),
    )
    for method in rank_methods:
        score = dice_stats["points"][method]
        score_text = _cell(score)
        if dice_stats["eligible_by_method"][method]:
            score_text += f" [{_cell(dice_stats['lower'][method])}, {_cell(dice_stats['upper'][method])}]"
        lines.append(
            f"| {display_name(method)} | {counts[method]} | {score_text} | {_cell(result['metrics']['iou']['points'][method])} | {_cell(result['metrics']['nsd']['points'][method])} | {_cell(result['metrics']['precision']['points'][method])} | {_cell(result['metrics']['recall']['points'][method])} |"
        )
    lines += [
        "",
        "### Per-metric sample sizes and bootstrap coverage",
        "",
        "| Metric | Minimum slides per method | Maximum slides per method | Confidence intervals |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, name, direction in METRIC_LABELS:
        stats = result["metrics"][key]
        sample_sizes = list(stats["n_by_method"].values())
        available = sum(stats["eligible_by_method"].values())
        lines.append(
            f"| {_label(name)} {direction} | {min(sample_sizes, default=0):,} | {max(sample_sizes, default=0):,} | {available}/{len(methods)} methods have {ci_text} percentile bootstrap intervals |"
        )
    lines += [
        "",
        "### Complete aggregate bootstrap summary",
        "",
        "Each row uses all valid evaluated slides for that particular method and metric. Intervals are blank if too few slide observations are available.",
        "",
        "| Metric | Method | Evaluated slides | Mean | SD | Median | IQR (25th–75th) | "
        + ci_text
        + " CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for key, name, direction in METRIC_LABELS:
        stats = result["metrics"][key]
        for method in rank_methods:
            confidence_interval = (
                f"[{_cell(stats['lower'][method])}, {_cell(stats['upper'][method])}]"
                if stats["eligible_by_method"][method]
                else "—"
            )
            lines.append(
                f"| {_label(name)} {direction} | {display_name(method)} | {stats['n_by_method'][method]:,} | {_cell(stats['points'][method])} | {_cell(stats['sd'][method])} | {_cell(stats['median'][method])} | {_cell(stats['q25'][method])}–{_cell(stats['q75'][method])} | {confidence_interval} |"
            )
    lines += [
        "",
        "## 3. Calibrated area errors",
        "",
        "| Method configuration | FP area (mm²) ↓ | FN area (mm²) ↓ | Mean area bias (%) → 0 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in rank_methods:
        lines.append(
            f"| {display_name(method)} | {_cell(result['metrics']['false_positive_area_mm2']['points'][method])} | {_cell(result['metrics']['false_negative_area_mm2']['points'][method])} | {_cell(result['metrics']['area_bias_pct']['points'][method])} |"
        )
    lines += [
        "",
        "Each area column is the mean of available per-slide measurements for that method. Areas require native scanner calibration; blank cells indicate unavailable calibration or other missing values.",
        "",
    ]
    lines += [
        "## 4. Annotation and data-quality validation",
        "",
        "| Finding | Count |",
        "| --- | ---: |",
    ]
    for metric, title in [
        ("gt_spline_count", "Evaluations involving at least one GT spline"),
        ("gt_hole_count", "Evaluations involving at least one GT hole"),
        ("gt_outside_area_px2", "Evaluations with out-of-slide GT area"),
        (
            "prediction_outside_area_px2",
            "Evaluations with out-of-slide prediction area",
        ),
    ]:
        count = sum((_number(row.get(metric)) or 0) > 0 for row in rows)
        lines.append(f"| {title} | {count:,} |")
    errors = [
        row
        for row in rows
        if str(row.get("evaluation_mode")) == "failed"
        or str(row.get("evaluation_error") or "")
    ]
    if errors:
        lines += [
            "",
            "### Failed evaluations (first 25)",
            "",
            "| Slide | Method | Error |",
            "| --- | --- | --- |",
        ]
        for row in errors[:25]:
            lines.append(
                f"| `{_label(row.get('filename', ''))}` | `{_label(str(row.get('run_id', ''))[:50])}` | {_label(row.get('evaluation_error', ''))[:240]} |"
            )
        if len(errors) > 25:
            lines += [
                "",
                f"{len(errors) - 25:,} more failures appear in the validation CSV.",
            ]
    if result["duplicate_slide_method_pairs"]:
        lines += [
            "",
            f"**Duplicate results excluded from aggregate statistics:** {result['duplicate_slide_method_pairs']:,} ambiguous slide–method pairs. Check for multiple prediction formats or colliding filenames.",
        ]
    lines += [
        "",
        "## 5. Evaluation protocol and statistical limitations",
        "",
        "| Parameter | Setting |",
        "| --- | --- |",
        "| Input/output | ASAP XML or GeoJSON polygon geometry |",
        "| Ground-truth directory | " + _label(ground_truth_dir or "None") + " |",
        "| Annotation matching report | "
        + _label(ground_truth_matching_path or "Not generated")
        + " |",
        "| Source-slide directory | " + _label(data_dir or "Not supplied") + " |",
        "| Foreground tissue groups | " + _label(", ".join(tissue_groups)) + " |",
        "| NSD tolerance | "
        + (
            f"{boundary_tolerance_um:g} µm"
            if boundary_tolerance_um is not None
            else "Not calculated"
        )
        + " |",
        "| NSD boundary sampling | "
        + (
            f"{boundary_step_um:g} µm"
            if boundary_tolerance_um is not None
            else "Not applicable"
        )
        + " |",
        "| Confidence interval | "
        + ci_text
        + " method-specific slide-level percentile bootstrap |",
        "| Bootstrap resamples | " + f"{bootstrap_resamples:,}" + " |",
        "| Bootstrap seed | " + str(bootstrap_seed) + " |",
        "| Minimum evaluated slides per method for CI | "
        + str(min_bootstrap_slides)
        + " |",
        "| P-values and multiplicity correction | Not calculated |",
        "| Bootstrap clustering | Slide (patient/specimen identifiers unavailable) |",
        "",
    ]
    lines += [
        "**Statistical interpretation:** bootstrap intervals are calculated independently for each method from its valid slide scores. They quantify slide-sampling variability conditional on slides being independent; multiple slides from one patient or specimen can make them too narrow. Cohort differences between methods during incomplete runs prevent reliable head-to-head comparisons. No patient-level inference or formal significance claims are justified by these intervals alone.",
        "",
    ]
    lines += [
        "**Data interpretation:** scores assume comparable, complete foreground annotations and valid native MPP. Missing ground truth, failed predictions, and selectively completed methods are not assumed to be missing at random. The confidence interval calculation does not resolve annotation error, unverified tissue groups, or slide-coordinate misalignment.",
        "",
    ]
    bootstrap_summary.parent.mkdir(parents=True, exist_ok=True)
    fd, summary_temporary = tempfile.mkstemp(
        prefix=f".{bootstrap_summary.name}.", dir=bootstrap_summary.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            columns = (
                "run_id",
                "formal_name",
                "metric",
                "direction",
                "n_evaluated_slides",
                "n_paired_slides",
                "mean",
                "standard_deviation",
                "median",
                "q25",
                "q75",
                "ci_lower",
                "ci_upper",
                "confidence_level",
                "bootstrap_resamples",
                "bootstrap_seed",
                "inference_unit",
                "ci_status",
            )
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for key, title, direction in METRIC_LABELS:
                stats = result["metrics"][key]
                for method in methods:
                    point = stats["points"][method]
                    writer.writerow(
                        {
                            "run_id": method,
                            "formal_name": result["labels"][method],
                            "metric": key,
                            "direction": direction,
                            "n_evaluated_slides": stats["n_by_method"][method],
                            "n_paired_slides": stats["n"],
                            "mean": point if math.isfinite(point) else "",
                            "standard_deviation": stats["sd"][method]
                            if math.isfinite(stats["sd"][method])
                            else "",
                            "median": stats["median"][method]
                            if math.isfinite(stats["median"][method])
                            else "",
                            "q25": stats["q25"][method]
                            if math.isfinite(stats["q25"][method])
                            else "",
                            "q75": stats["q75"][method]
                            if math.isfinite(stats["q75"][method])
                            else "",
                            "ci_lower": stats["lower"][method]
                            if stats["eligible_by_method"][method]
                            else "",
                            "ci_upper": stats["upper"][method]
                            if stats["eligible_by_method"][method]
                            else "",
                            "confidence_level": bootstrap_confidence,
                            "bootstrap_resamples": bootstrap_resamples,
                            "bootstrap_seed": bootstrap_seed,
                            "inference_unit": "slide",
                            "ci_status": "calculated"
                            if stats["eligible_by_method"][method]
                            else "insufficient_method_slides",
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())
        Path(summary_temporary).replace(bootstrap_summary)
    except Exception:
        Path(summary_temporary).unlink(missing_ok=True)
        raise
    lines += [
        "## 6. Source artifacts",
        "",
        f"- Raw evaluation metrics: `{_label(os.path.relpath(metrics_path, destination.parent))}`",
    ]
    if validation_path is not None:
        lines.append(
            f"- Annotation/geometry validation: `{_label(os.path.relpath(validation_path, destination.parent))}`"
        )
    lines.append(
        f"- Aggregate bootstrap intervals: `{_label(os.path.relpath(bootstrap_summary, destination.parent))}`"
    )
    lines += ["", f"- Metrics storage format: `{_label(output_format)}`", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(destination)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .metadata import load_metadata, normalize_slide
from segmenteer.methods.identity import architecture_label


METRICS = {"dice": ("dice", "Dice"), "nsd": ("nsd", "NSD")}
DIMENSIONS = ("source", "stain", "organ", "center")
CYAN = "#00AFDC"
SILVER = "#C0C0C0"
GRAY = "#777777"
MISSING = "#E9EDF0"


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _method_name(formal, run_id):
    enriched = architecture_label(formal_name=formal, run_id=run_id)
    if "/" in enriched:
        return enriched
    original = re.sub(r"Segmenter$", "", formal or "")
    if original == "TRIDENT":
        original = "TRIDENT " + (
            run_id.split("__", 1)[0].replace("trident_", "").replace("_", " ").title()
        )
    mpp = re.search(r"(?:^|[_-])mpp=([0-9.]+)", run_id)
    if mpp:
        original += f" ({mpp.group(1)} MPP)"
    return original or run_id.split("__", 1)[0]


def _bootstrap(values, resamples, confidence, seed):
    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 20 or resamples <= 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    step = max(1, min(500, 200000 // array.size))
    for first in range(0, resamples, step):
        batch = min(step, resamples - first)
        ids = rng.integers(0, array.size, size=(batch, array.size))
        samples[first : first + batch] = array[ids].mean(axis=1)
    alpha = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(samples, (alpha, 1 - alpha)))


def _assemble(
    metrics,
    metadata,
    metric,
    summary=None,
    minimum_method=1,
    matched=False,
    resamples=10000,
    confidence=0.95,
    seed=42,
):
    rows = _read_csv(metrics)
    successes = defaultdict(dict)
    missing_meta = set()
    skipped = Counter()
    for record in rows:
        rid = record.get("run_id", "")
        filename = record.get("filename", "")
        if not rid or not filename:
            continue
        if record.get("evaluation_mode") != "supervised" or record.get(
            "evaluation_error"
        ):
            skipped["not_supervised_or_failed"] += 1
            continue
        if _number(record.get(metric)) is None:
            skipped["missing_metric"] += 1
            continue
        slide = normalize_slide(filename)
        if slide not in metadata:
            missing_meta.add(slide)
            continue
        if slide in successes[rid]:
            raise ValueError(
                f"Duplicate evaluation row for method={rid}, slide={slide}"
            )
        successes[rid][slide] = record
    successes = {rid: rr for rid, rr in successes.items() if len(rr) >= minimum_method}
    if not successes:
        raise ValueError(
            "No supervised scores with matching slide metadata and the requested metric"
        )
    if matched:
        shared = set.intersection(*(set(slides) for slides in successes.values()))
        if not shared:
            raise ValueError("No shared evaluated slides across the selected methods")
        successes = {
            rid: {slide: r for slide, r in rr.items() if slide in shared}
            for rid, rr in successes.items()
        }
    elif len(successes) > 1:
        shared = set.intersection(*(set(slides) for slides in successes.values()))
    else:
        shared = set(next(iter(successes.values())))
    summaries = {}
    if summary and not matched:
        for record in _read_csv(summary):
            if (
                record.get("metric") == metric
                and record.get("ci_status") == "calculated"
            ):
                summaries[record.get("run_id", "")] = record
    data = []
    for rid, records in successes.items():
        vals = [_number(row.get(metric)) for row in records.values()]
        vals = [v for v in vals if v is not None]
        name = _method_name(next(iter(records.values())).get("formal_name", ""), rid)
        entry = {
            "run_id": rid,
            "name": name,
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "rows": records,
        }
        cached = summaries.get(rid)
        if (
            cached
            and int(cached.get("n_evaluated_slides", -1)) == len(vals)
            and _number(cached.get("ci_lower")) is not None
            and _number(cached.get("ci_upper")) is not None
            and abs(float(cached.get("confidence_level", 0)) - confidence) < 1e-9
            and int(cached.get("bootstrap_resamples", 0)) == resamples
            and int(cached.get("bootstrap_seed", -1)) == seed
        ):
            entry["ci"] = (_number(cached["ci_lower"]), _number(cached["ci_upper"]))
        else:
            entry["ci"] = _bootstrap(vals, resamples, confidence, seed)
        for short, key in (
            ("under", "under_segmentation_rate"),
            ("over", "over_segmentation_rate"),
        ):
            sample = [_number(row.get(key)) for row in records.values()]
            sample = [v for v in sample if v is not None]
            entry[short] = sum(sample) / len(sample) if sample else None
            entry[short + "_ci"] = _bootstrap(
                sample, resamples, confidence, seed + (1 if short == "under" else 2)
            )
        data.append(entry)
    data.sort(key=lambda x: (-x["mean"], -x["n"], x["run_id"]))
    label_occurrences = Counter(item["name"] for item in data)
    for entry in data:
        if label_occurrences[entry["name"]] > 1:
            match = re.search(r"cfg=([0-9a-f]+)", entry["run_id"])
            entry["name"] += " · " + (
                match.group(1)[:6] if match else entry["run_id"][-6:]
            )
    return data, {
        "input_rows": len(rows),
        "skipped": dict(skipped),
        "missing_metadata_slides": len(missing_meta),
        "matched_slides": len(shared),
        "evaluated_slides": len(set().union(*(set(r) for r in successes.values()))),
    }


def _figure(
    rows,
    metadata,
    dimension,
    metric,
    output,
    minimum_cell,
    minimum_group,
    fmt,
    dpi,
    confidence,
):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

    groups = defaultdict(set)
    for method in rows:
        for slide in method["rows"]:
            label = metadata[slide][dimension]
            if label:
                groups[label].add(slide)
    kept = [
        (group, len(slides))
        for group, slides in groups.items()
        if len(slides) >= minimum_group
    ]
    excluded = [
        (dimension, group, len(slides))
        for group, slides in groups.items()
        if len(slides) < minimum_group
    ]
    kept.sort(key=lambda item: (-item[1], item[0].casefold()))
    if not kept:
        return None, [], excluded
    labels = [name for name, _ in kept]
    n = len(rows)
    values = np.full((n, len(labels)), np.nan)
    cell_n = np.zeros_like(values, dtype=int)
    for i, method in enumerate(rows):
        bucket = defaultdict(list)
        for slide, record in method["rows"].items():
            group = metadata[slide][dimension]
            value = _number(record.get(metric))
            if group and value is not None:
                bucket[group].append(value)
        for j, name in enumerate(labels):
            cell_n[i, j] = len(bucket[name])
            if len(bucket[name]) >= minimum_cell:
                values[i, j] = sum(bucket[name]) / len(bucket[name])
    colors = LinearSegmentedColormap.from_list(
        "cpg_cyan", ["#FFFFFF", "#E5F8FC", "#BDEBF6", "#81D9ED", "#42C6E5", CYAN], N=256
    )
    colors.set_bad(MISSING)
    heatmap_width = max(6.3, len(labels) * 0.62)
    fig_w = max(11.9, 4.1 + heatmap_width + 1.05 + 1.48)
    fig_h = max(4.0, 2.4 + 0.32 * n)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [heatmap_width, 1.05, 1.48], "wspace": 0.045},
    )
    ax, bx, cx = axes
    plt.rcParams["svg.fonttype"] = "none"
    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (
            font
            for font in ("Arial", "Liberation Sans", "DejaVu Sans")
            if font in available_fonts
        ),
        "DejaVu Sans",
    )
    for axis in axes:
        axis.set_facecolor("white")
        axis.tick_params(
            which="both", colors="black", width=0.38, labelsize=7, length=2.4
        )
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.spines["bottom"].set_visible(True)
        axis.spines["bottom"].set_color("black")
        axis.spines["bottom"].set_linewidth(0.38)
    y = np.arange(n)
    ax.imshow(
        np.ma.masked_invalid(values),
        vmin=0,
        vmax=1,
        cmap=colors,
        aspect="auto",
        interpolation="nearest",
    )
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{i + 1}  {r['name']}  (n={r['n']})" for i, r in enumerate(rows)],
        fontsize=7.3,
        color="black",
    )
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(
        [f"{v}\n(n={count})" for v, count in kept],
        rotation=42 if len(labels) > 6 else 35,
        ha="right",
        rotation_mode="anchor",
        fontsize=6.7,
    )
    ax.tick_params(axis="x", length=0, pad=3)
    if len(labels) <= 15:
        for i in range(n):
            for j in range(len(labels)):
                if np.isfinite(values[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        color="#002331",
                    )
    for i, row in enumerate(rows):
        mean = row["mean"]
        bx.barh(i, mean, height=0.38, color=colors(mean), edgecolor="none")
        lo, hi = row["ci"]
        if lo is not None and hi is not None:
            bx.plot([lo, hi], [i, i], color="black", linewidth=0.52, zorder=5)
            bx.vlines(
                [lo, hi], i - 0.08, i + 0.08, color="black", linewidth=0.44, zorder=5
            )
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    bx.set_xlabel(
        f"Mean {metric.upper() if metric == 'nsd' else 'Dice'}",
        fontsize=7.2,
        labelpad=4,
    )
    bx.tick_params(axis="y", left=False, labelleft=False)
    scale = 0.12

    def transform(val):
        return np.arcsinh(np.asarray(val, dtype=float) / scale)

    upper = max(
        [
            float(r["over_ci"][1] if r["over_ci"][1] is not None else r["over"] or 0)
            for r in rows
        ]
        + [1.0]
    )
    right = max(1.15, upper * 1.08)
    for i, row in enumerate(rows):
        u = row["under"]
        o = row["over"]
        if u is not None:
            cx.barh(
                i, -float(transform(u)), height=0.38, color=SILVER, edgecolor="none"
            )
        if o is not None:
            cx.barh(i, float(transform(o)), height=0.38, color=GRAY, edgecolor="none")
        for field, sign in (("under", -1), ("over", 1)):
            lo, hi = row[field + "_ci"]
            if lo is None or hi is None:
                continue
            x1, x2 = sorted((sign * float(transform(lo)), sign * float(transform(hi))))
            cx.plot([x1, x2], [i, i], color="black", linewidth=0.51, zorder=5)
            cx.vlines(
                [x1, x2], i - 0.075, i + 0.075, color="black", linewidth=0.44, zorder=5
            )
    cx.axvline(0, color="black", linewidth=0.38)
    cx.set_xlim(-float(transform(1.08)), float(transform(right)))
    ticks = [-1.0, 0.0, 1.0]
    if right >= 2.5:
        ticks.append(3.0)
    if right >= 6.5:
        ticks.append(10.0)
    if right >= 13.0:
        ticks.append(20.0)
    if right >= 32.0:
        ticks.append(50.0)
    ticks = [v for v in ticks if v <= right]
    cx.set_xticks(
        [float(transform(v)) for v in ticks], [str(abs(round(100 * v))) for v in ticks]
    )
    cx.set_xlabel("Under / over (%) · asinh scale", fontsize=7.2, labelpad=4)
    cx.tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(
        left=min(0.29, max(0.15, 2.9 / fig_w)),
        right=0.985,
        top=0.89,
        bottom=min(0.39, max(0.24, 1.5 / fig_h)),
    )
    for axis, text in zip(
        axes,
        (
            f"a  {metric.upper() if metric == 'nsd' else 'Dice'} by {dimension}",
            f"b  Mean {metric.upper() if metric == 'nsd' else 'Dice'}",
            "c  Area error",
        ),
    ):
        axis.text(
            0,
            1.035,
            text,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.1,
            fontweight="bold",
            color="black",
        )
    fig.patch.set_facecolor("white")
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for extension in fmt:
        destination = output / f"performance_by_{dimension}_{metric}.{extension}"
        fig.savefig(
            destination,
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.07,
            facecolor="white",
        )
        saved.append(str(destination))
    plt.close(fig)
    coverage = [
        {
            "dimension": dimension,
            "subgroup": group,
            "metadata_slides_with_successful_evaluation": group_n,
            "method": row["name"],
            "run_id": row["run_id"],
            "n_evaluated": int(cell_n[i, j]),
            "mean_score": f"{values[i, j]:.9f}" if np.isfinite(values[i, j]) else "",
        }
        for j, (group, group_n) in enumerate(kept)
        for i, row in enumerate(rows)
    ]
    return saved, coverage, excluded


def generate_performance_figures(
    metrics,
    metadata,
    output_dir,
    bootstrap_summary=None,
    metric="nsd",
    min_cell_slides=5,
    min_group_slides=5,
    min_method_slides=1,
    matched=False,
    bootstrap_resamples=10000,
    bootstrap_confidence=0.95,
    bootstrap_seed=42,
    formats=("png", "svg"),
    dpi=250,
    pareto=True,
    timing_source=None,
    pareto_timing_scope="available",
    min_pareto_timing_slides=5,
):
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    if not 0 < bootstrap_confidence < 1:
        raise ValueError("bootstrap_confidence must be between zero and one")
    if any(v < 1 for v in (min_cell_slides, min_group_slides, min_method_slides)):
        raise ValueError("Minimum slide counts must be positive")
    if not set(formats) <= {"png", "svg", "pdf"}:
        raise ValueError("formats must contain png, svg, or pdf")
    source = load_metadata(metadata)
    rows, status = _assemble(
        metrics,
        source,
        metric,
        summary=bootstrap_summary,
        minimum_method=min_method_slides,
        matched=matched,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pareto_info = None
    if pareto:
        from .speed_accuracy import generate_speed_accuracy_figure
        pareto_info = generate_speed_accuracy_figure(
            rows, metrics=metrics, output_dir=output, timing_source=timing_source,
            timing_scope=pareto_timing_scope, matched=matched,
            min_method_slides=min_method_slides, min_timing_slides=min_pareto_timing_slides,
            formats=formats, dpi=dpi)
    figures = {}
    coverage = []
    excluded_groups = []
    for dimension in DIMENSIONS:
        saved, subset, excluded = _figure(
            rows,
            source,
            dimension,
            metric,
            output,
            min_cell_slides,
            min_group_slides,
            formats,
            dpi,
            bootstrap_confidence,
        )
        figures[dimension] = saved or []
        coverage.extend(subset)
        excluded_groups.extend(excluded)
    with (output / f"plot_coverage_{metric}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dimension",
                "subgroup",
                "metadata_slides_with_successful_evaluation",
                "method",
                "run_id",
                "n_evaluated",
                "mean_score",
            ],
        )
        writer.writeheader()
        writer.writerows(coverage)
    with (output / f"plot_excluded_groups_{metric}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["dimension", "subgroup", "n_unique_successful_slides", "reason"]
        )
        for dimension, group, count in sorted(excluded_groups):
            writer.writerow(
                [dimension, group, count, f"below min_group_slides={min_group_slides}"]
            )
    with (output / f"plot_rankings_{metric}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "run_id",
                "method",
                "n_evaluated",
                "mean_score",
                "ci_lower",
                "ci_upper",
                "under_rate",
                "over_rate",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "run_id": row["run_id"],
                    "method": row["name"],
                    "n_evaluated": row["n"],
                    "mean_score": row["mean"],
                    "ci_lower": row["ci"][0],
                    "ci_upper": row["ci"][1],
                    "under_rate": row["under"],
                    "over_rate": row["over"],
                }
            )
    missing_fields = {
        dimension: len(
            {
                slide
                for row in rows
                for slide in row["rows"]
                if not source[slide][dimension]
            }
        )
        for dimension in DIMENSIONS
    }
    manifest = {
        "metric": metric,
        "matched_cohort": matched,
        "confidence_level": bootstrap_confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "min_cell_slides": min_cell_slides,
        "min_group_slides": min_group_slides,
        "min_method_slides": min_method_slides,
        "source_metrics": str(Path(metrics).resolve()),
        "source_metadata": str(Path(metadata).resolve()),
        "source_bootstrap_summary": str(Path(bootstrap_summary).resolve())
        if bootstrap_summary
        else None,
        "methods": len(rows),
        "figures": figures,
        **({"speed_accuracy": pareto_info} if pareto_info is not None else {}),
        "excluded_subgroup_counts": dict(
            Counter(dimension for dimension, _, _ in excluded_groups)
        ),
        "slides_missing_subgroup_metadata": missing_fields,
        **status,
        "notes": [
            "Bootstrap intervals treat slides as independent; patient identifiers may be unavailable.",
            "Overall rankings use each method's evaluated slides unless matched_cohort is true.",
            "Under/over panel uses a labeled inverse-hyperbolic-sine display transform; statistics retain their original units.",
            "Heatmap cells with fewer than the minimum evaluated slide count are displayed as missing, not zero.",
        ],
    }
    (output / f"plot_manifest_{metric}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest

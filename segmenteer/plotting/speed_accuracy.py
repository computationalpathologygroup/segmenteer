"""Generate speed–accuracy plots and identify Pareto-efficient methods.

Mean Dice and NSD are derived from valid per-slide evaluations. Processing
times are read from measured timing records and associated with methods by
run identifier. The figure and accompanying CSV report the same method-level summaries.
"""

from __future__ import annotations

import csv
import math
import warnings
from collections import defaultdict
from pathlib import Path

from .metadata import normalize_slide

METRICS = ("dice", "nsd")
CYAN = "#00AFDC"
GREY = "#92999F"
BLACK = "#151515"


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _timing_records(source):
    """Load validated per-slide timing records from a directory or CSV.

    Per-prediction JSON records take precedence when present. A benchmark
    timing CSV is used if the directory contains no valid JSON records.
    """
    from segmenteer.benchmark.timing import collect_timings

    source = Path(source)
    if source.is_dir():
        records = collect_timings(source)
        if records:
            return records, "per-prediction-json"
        source = source / "benchmark_timings.csv"
    if not source.is_file():
        return [], None
    with source.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    return records, "benchmark-timings-csv"


def _timing_lookup(records):
    by_run = defaultdict(dict)
    devices = defaultdict(set)
    for item in records:
        run = str(item.get("run_id") or "").strip()
        slide = normalize_slide(item.get("filename") or "")
        seconds = _finite(item.get("total_processing_time_s"))
        if not run or not slide or seconds is None or seconds <= 0:
            continue
        prior = by_run[run].get(slide)
        if prior is not None and not math.isclose(prior, seconds, rel_tol=1e-8):
            raise ValueError(f"Conflicting timings for {run}: {slide}")
        by_run[run][slide] = seconds
        device = str(item.get("device") or "").strip()
        if device:
            devices[run].add(device)
    return by_run, devices


def _pareto_flags(entries, metric):
    for item in entries:
        item[f"pareto_{metric}"] = not any(
            other is not item
            and other["seconds"] <= item["seconds"]
            and other[metric] >= item[metric]
            and (other["seconds"] < item["seconds"] or other[metric] > item[metric])
            for other in entries
        )


def _select_labels(entries, metric, maximum=3):
    # Annotate a compact subset of methods; the exported CSV identifies all points.
    efficient = sorted(
        (item for item in entries if item[f"pareto_{metric}"]),
        key=lambda item: (-item[metric], item["seconds"], item["run_id"]),
    )
    accurate = sorted(entries, key=lambda item: (-item[metric], item["seconds"], item["run_id"]))
    chosen = []
    for item in [*efficient, *accurate]:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= maximum:
            break
    return sorted(chosen, key=lambda item: -item[metric])


def _short_name(name, limit=29):
    """Return a concise method name for direct point annotations."""
    name = str(name).split("/", 1)[0].strip()
    return name if len(name) <= limit else name[:limit - 1].rstrip() + "…"


def _place_nearby_labels(fig, ax, selected, metric):
    """Place method labels near their points using short connectors.

    Candidate offsets are evaluated in display coordinates to minimize label
    overlap, clipping at axis boundaries, and occlusion of other marked points.
    The maximum candidate offset is 19 points vertically or 16 horizontally.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    scale = fig.dpi / 72.0
    used = []
    annotations = []
    all_points = [ax.transData.transform((item["seconds"], item[metric])) for item in selected]
    # Evaluate compact offsets on all four sides of each point.
    offsets = [(7, 8), (7, -8), (-7, 8), (-7, -8),
               (11, 14), (11, -14), (-11, 14), (-11, -14),
               (7, 19), (7, -19), (-7, 19), (-7, -19),
               (16, 8), (16, -8), (-16, 8), (-16, -8)]

    for i, item in enumerate(selected):
        label = _short_name(item["name"])
        text_measure = ax.text(.5, .5, label, transform=ax.transAxes,
                               fontsize=7.7, color=BLACK, alpha=0)
        fig.canvas.draw()
        extent = text_measure.get_window_extent(renderer)
        text_measure.remove()
        width, height = extent.width, extent.height
        x, y = all_points[i]
        chosen = None
        best = float("inf")
        for dx, dy in offsets:
            left = x + dx * scale if dx >= 0 else x + dx * scale - width
            right = left + width
            bottom = y + dy * scale if dy >= 0 else y + dy * scale - height
            top = bottom + height
            rectangle = Bbox.from_extents(left, bottom, right, top)
            # Penalize annotation bounds extending beyond the plotting area.
            overflow = max(0, bounds.x0 + 3 - left) + max(0, right - bounds.x1 + 2) \
                       + max(0, bounds.y0 + 2 - bottom) + max(0, top - bounds.y1 + 3)
            penalty = overflow * 100
            for previous in used:
                overlap_w = max(0, min(right, previous.x1 + 5) - max(left, previous.x0 - 5))
                overlap_h = max(0, min(top, previous.y1 + 4) - max(bottom, previous.y0 - 4))
                penalty += overlap_w * overlap_h * 20
            # Penalize positions that conceal other annotated points.
            for j, (px, py) in enumerate(all_points):
                if j != i and left - 5 <= px <= right + 5 and bottom - 5 <= py <= top + 5:
                    penalty += 200
            penalty += (.2 * (abs(dx) + abs(dy)))
            if penalty < best:
                best, chosen = penalty, (dx, dy, rectangle)
        dx, dy, rectangle = chosen
        used.append(rectangle)
        annotations.append(ax.annotate(label, xy=(item["seconds"], item[metric]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    fontsize=7.7, color=BLACK, zorder=6,
                    arrowprops={"arrowstyle": "-", "color": "#8F969A",
                                "linewidth": .5, "shrinkA": 2, "shrinkB": 3}))
    return annotations


def _draw(entries, output_dir, formats, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": .85, "svg.fonttype": "none", "pdf.fonttype": 42,
    }):
        # Allocate space for axes and direct point annotations.
        fig, axs = plt.subplots(1, 2, figsize=(11.3, 4.35), facecolor="white")
        fig.subplots_adjust(left=.072, right=.987, bottom=.19, top=.85, wspace=.26)
        for index, (ax, metric) in enumerate(zip(axs, METRICS)):
            title = "Dice" if metric == "dice" else "NSD"
            ax.text(0, 1.065, "AB"[index], transform=ax.transAxes,
                    fontweight="bold", fontsize=10, ha="left", va="bottom", color="black")
            ax.text(.065, 1.065, title, transform=ax.transAxes,
                    fontsize=10, ha="left", va="bottom", color="black")
            ax.set_ylabel(f"Mean {title}", color="black", fontsize=8.5)
            ax.set_xlabel("Mean processing time (s/slide)", color="black", fontsize=8.3, labelpad=8)
            ax.set_xscale("log")
            speeds = [e["seconds"] for e in entries]
            low = min(speeds) * .79
            high = max(speeds) * 1.23
            if low <= 0 or high <= low:
                high = max(high, low * 1.4)
            ax.set_xlim(low, high)
            ax.set_ylim(0, 1.01)
            xticks = [v for v in (.1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500)
                      if low < v < high]
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_yticks([0, .2, .4, .6, .8, 1.0])
            ax.grid(False)
            ax.tick_params(axis="both", colors="black", direction="out", length=3, width=.8, labelsize=7.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for axis in ("left", "bottom"):
                ax.spines[axis].set_color("black")
            frontier = sorted((r for r in entries if r[f"pareto_{metric}"]),
                              key=lambda r: (r["seconds"], r[metric]))
            if len(frontier) > 1:
                ax.plot([r["seconds"] for r in frontier], [r[metric] for r in frontier],
                        color=CYAN, lw=.85, zorder=2, alpha=.85)
            ordinary = [r for r in entries if not r[f"pareto_{metric}"]]
            ax.scatter([r["seconds"] for r in ordinary], [r[metric] for r in ordinary],
                       color=GREY, s=18, alpha=.75, edgecolors="white", linewidths=.28, zorder=3)
            ax.scatter([r["seconds"] for r in frontier], [r[metric] for r in frontier],
                       color=CYAN, s=27, edgecolors="white", linewidths=.5, zorder=4)
            _place_nearby_labels(fig, ax, _select_labels(entries, metric), metric)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for extension in formats:
            target = output_dir / f"pareto_speed_accuracy.{extension}"
            fig.savefig(target, dpi=dpi if extension == "png" else None,
                        facecolor="white", bbox_inches="tight", pad_inches=.07)
            saved.append(str(target))
        plt.close(fig)
    return saved


def generate_speed_accuracy_figure(
    rows, *, metrics, output_dir, timing_source=None,
    timing_scope="available", matched=False, min_method_slides=1, min_timing_slides=5,
    formats=("png", "svg"), dpi=250,
):
    """Export Dice–time and NSD–time panels with a method-level data table.

    Accuracy uses slides with valid values for both Dice and NSD. Processing
    time is aggregated by exact run identifier, optionally restricted to the
    evaluated slides when ``timing_scope`` is ``"matched"``.
    """
    if min_timing_slides < 1:
        raise ValueError("min_timing_slides must be positive")
    if timing_scope not in {"available", "matched"}:
        raise ValueError("timing_scope must be available or matched")
    timing_source = Path(timing_source) if timing_source else Path(metrics).parent
    records, timing_origin = _timing_records(timing_source)
    if not records:
        return {"files": [], "methods": 0, "reason": f"No timing records found at {timing_source}"}
    by_run, devices = _timing_lookup(records)
    entries = []
    for method in rows:
        run = method["run_id"]
        # Use the same valid per-slide scores for both accuracy metrics.
        both = []
        for slide, record in method["rows"].items():
            dice, nsd = _finite(record.get("dice")), _finite(record.get("nsd"))
            if dice is not None and nsd is not None and 0 <= dice <= 1 and 0 <= nsd <= 1:
                both.append((slide, dice, nsd))
        if len(both) < min_method_slides:
            continue
        timings = by_run.get(run, {})
        if timing_scope == "matched":
            eligible = {slide for slide, _, _ in both}
            durations = [v for slide, v in timings.items() if slide in eligible]
        else:
            durations = list(timings.values())
        if len(durations) < min_timing_slides:
            continue
        entries.append({
            "run_id": run, "name": method["name"], "dice": sum(x[1] for x in both) / len(both),
            "nsd": sum(x[2] for x in both) / len(both),
            "seconds": sum(durations) / len(durations),
            "accuracy_n": len(both), "timing_n": len(durations),
            "device": ", ".join(sorted(devices.get(run, set()))),
        })
    if not entries:
        return {"files": [], "methods": 0, "reason": f"No eligible method has both scores and at least {min_timing_slides} matching timing records"}
    _pareto_flags(entries, "dice")
    _pareto_flags(entries, "nsd")
    output = Path(output_dir)
    saved = _draw(entries, output, formats, dpi)
    with (output / "pareto_speed_accuracy_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "method", "mean_processing_s_per_slide", "mean_dice", "mean_nsd",
            "n_accuracy_slides", "n_timing_slides", "device", "pareto_dice", "pareto_nsd",
        ])
        writer.writeheader()
        for row in sorted(entries, key=lambda x: (x["seconds"], x["run_id"])):
            writer.writerow({"run_id": row["run_id"], "method": row["name"],
                             "mean_processing_s_per_slide": f'{row["seconds"]:.8g}',
                             "mean_dice": f'{row["dice"]:.8g}', "mean_nsd": f'{row["nsd"]:.8g}',
                             "n_accuracy_slides": row["accuracy_n"], "n_timing_slides": row["timing_n"],
                             "device": row["device"], "pareto_dice": row["pareto_dice"],
                             "pareto_nsd": row["pareto_nsd"]})
    all_devices = set().union(*(devices.get(e["run_id"], set()) for e in entries))
    if len(all_devices) > 1:
        warnings.warn("Pareto plot mixes timing device labels; cross-device speed comparisons may be invalid", stacklevel=2)
    return {"files": saved, "methods": len(entries), "timing_source": timing_origin,
            "timing_scope": timing_scope, "mixed_devices": len(all_devices) > 1,
            "source": str(timing_source),
            "notes": "Accuracy: paired Dice/NSD slide scores from current plotting cohort. "
                     "Speed: arithmetic mean of observed per-slide processing times; timing cohorts may differ "
                     "from accuracy cohort unless timing_scope=matched."}

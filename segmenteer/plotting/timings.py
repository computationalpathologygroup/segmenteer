from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from segmenteer.methods.identity import architecture_label

CYAN = "#00AFDC"
PALE_CYAN = "#D0F1F9"
BLACK = "#000000"


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def _read_raw_timings(path):
    source = Path(path).expanduser()
    if source.is_dir() or (
        not source.exists()
        and source.name == "benchmark_timings.csv"
        and source.parent.is_dir()
    ):
        root = source if source.is_dir() else source.parent
        from segmenteer.benchmark.timing import collect_timings

        records = collect_timings(root)
        if not records:
            raise ValueError(
                f"No completed prediction timing records found under {root}"
            )
        raw = []
        seen = {}
        for record in records:
            identity = (record.get("run_id"), record.get("filename"))
            previous = seen.get(identity)
            if previous is not None:
                if previous != record:
                    raise ValueError(f"Conflicting timing records for {identity!r}")
                continue
            seen[identity] = record
            raw.append(
                {
                    "filename": record.get("filename"),
                    "run_id": record.get("run_id"),
                    "segmentation_method": record.get("method_name"),
                    "total_processing_time_s": record.get("total_processing_time_s"),
                    "native_pixels": record.get("native_pixels"),
                    "device": record.get("device"),
                    "measurement_phase": record.get("measurement_phase"),
                }
            )
        return raw
    if not source.is_file():
        raise FileNotFoundError(
            f"Timing input not found: {source}. Pass the benchmark output directory or a timing CSV."
        )
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "run_id", "total_processing_time_s", "native_pixels"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Missing timing CSV columns: " + ", ".join(sorted(missing))
            )
        return list(reader)


def _read_timings(path, minimum, matched, exclude_first):
    if minimum < 1:
        raise ValueError("min_slides must be positive")
    raw = _read_raw_timings(path)
    methods = defaultdict(dict)
    for record in raw:
        run = (record.get("run_id") or "").strip()
        filename = (record.get("filename") or "").strip()
        if not run or not filename:
            continue
        if exclude_first and record.get("measurement_phase") == "first_call_in_process":
            continue
        duration = _number(record.get("total_processing_time_s"))
        if duration is None:
            continue
        slide = _slide_key(filename)
        if slide in methods[run]:
            raise ValueError(
                f"Multiple timing records for the same run and slide: {run} / {filename}"
            )
        record["_duration"] = duration
        pixels = _number(record.get("native_pixels"))
        record["_normalized"] = duration * 1000000.0 / pixels if pixels else None
        record["_slide"] = slide
        methods[run][slide] = record
    methods = {run: recs for run, recs in methods.items() if len(recs) >= minimum}
    if not methods:
        raise ValueError("No method has enough valid slide-level timings")
    if matched:
        common = set.intersection(*(set(recs) for recs in methods.values()))
        if not common:
            raise ValueError("No common slides across eligible methods")
        methods = {
            run: {slide: rec for slide, rec in recs.items() if slide in common}
            for run, recs in methods.items()
        }
    return methods


def _slide_key(filename):
    value = Path(filename.replace("\\", "/")).name.casefold()
    return re.sub(r"\.(?:ome\.tiff|tiff|tif|svs|ndpi|mrxs|scn|qptiff)$", "", value)


def _method_label(run, records):
    original = (next(iter(records.values())).get("segmentation_method") or "").strip()
    stem = run.split("__", 1)[0]
    lower = (original or stem).casefold()
    known = (
        ("atlaspatch", "AtlasPatchSAM2"),
        ("fastsam", "FastSAM"),
        ("improved_fesi", "FESI"),
        ("fesi", "FESI"),
        ("backgroundsubtractor", "MOG2"),
        ("hsv-threshold-v2", "HSV Threshold v2"),
        ("hsv_threshold_v2", "HSV Threshold v2"),
        ("hsvthreshold_v2", "HSV Threshold v2"),
        ("hsv", "HSV Threshold"),
        ("watershed", "Watershed"),
        ("morphological", "Morphological"),
        ("entropy", "EntropyMasker"),
        ("histomicstk", "HistomicsTK"),
        ("grandqc_patchwise", "GrandQC patchwise"),
        ("grandqc", "TRIDENT GrandQC"),
        ("nki_all_stains", "NKI AllStains"),
        ("nkiall", "NKI AllStains"),
        ("rtlucassen", "RTLucassen"),
        ("trident_pathprofiler", "TRIDENT PathProfiler"),
        ("trident_path", "TRIDENT PathProfiler"),
        ("trident_hest", "TRIDENT HEST"),
        ("bigpicture", "BigPicture"),
        ("otsu", "Otsu"),
        ("yen", "Yen"),
        ("li_tissue", "Li"),
    )
    name = next(
        (display for prefix, display in known if prefix in lower),
        original or stem.replace("_", " "),
    )
    if name == "HistomicsTK" and "saliency" in lower:
        name += " saliency"
    mpp = re.search(r"(?:^|[_-])mpp=([0-9.]+)", run)
    if mpp is None:
        mpp = re.search(r"(?:^|[_-])mpp([0-9.]+)(?:[_-]|$)", run)
    if mpp:
        name += f" ({float(mpp.group(1)):g} MPP)"

    enriched = architecture_label(run_id=run, formal_name=original)
    return enriched if "/" in enriched else name


def _prepare(methods):
    import numpy as np

    rows = []
    for run, records in methods.items():
        times = np.array(
            [entry["_duration"] for entry in records.values()], dtype=float
        )
        rows.append(
            {
                "run_id": run,
                "records": records,
                "median": float(np.median(times)),
                "name": _method_label(run, records),
            }
        )
    all_devices = {
        str(record.get("device") or "").strip()
        for item in rows
        for record in item["records"].values()
    }
    all_devices.discard("")
    if len(all_devices) > 1:
        for entry in rows:
            devices = {
                str(record.get("device") or "").strip()
                for record in entry["records"].values()
            }
            devices.discard("")
            if len(devices) == 1:
                entry["name"] += f" · {next(iter(devices))}"
            elif len(devices) > 1:
                entry["name"] += " · mixed"
    rows.sort(key=lambda item: (item["median"], item["run_id"]))
    collisions = Counter(item["name"] for item in rows)
    for entry in rows:
        if collisions[entry["name"]] > 1:
            match = re.search(r"cfg=([a-f0-9]+)", entry["run_id"])
            fingerprint = (
                match.group(1)[:6]
                if match
                else hashlib.sha256(entry["run_id"].encode()).hexdigest()[:6]
            )
            entry["name"] += f" [{fingerprint}]"
    return rows


def _offsets(run, slides):
    import numpy as np

    seed = int(hashlib.sha256(run.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return {slide: float(rng.uniform(-0.145, 0.145)) for slide in sorted(slides)}


def _format_log_tick(value, _position):
    if not math.isfinite(value) or value <= 0:
        return ""
    exponent = math.log10(value)
    if abs(exponent - round(exponent)) > 1e-7:
        return ""
    if exponent <= -3:
        return rf"$10^{{{round(exponent)}}}$"
    if exponent >= 4:
        return rf"$10^{{{round(exponent)}}}$"
    if value < 1:
        return f"{value:.{max(0, -round(exponent))}f}"
    return f"{value:g}"


def _plot(rows, output, formats, dpi):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    available = {entry.name for entry in font_manager.fontManager.ttflist}
    family = next(
        (
            name
            for name in ("Arial", "Liberation Sans", "DejaVu Sans")
            if name in available
        ),
        "DejaVu Sans",
    )
    with plt.rc_context(
        {
            "font.family": family,
            "font.size": 7.5,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    ):
        height = max(3.3, 1.12 + 0.35 * len(rows))
        figure = plt.figure(figsize=(12.1, height), facecolor="white")
        grid = figure.add_gridspec(
            1, 4, width_ratios=(3.0, 0.37, 3.55, 3.55), wspace=0.055
        )
        label_axis = figure.add_subplot(grid[0, 0])
        count_axis = figure.add_subplot(grid[0, 1], sharey=label_axis)
        left = figure.add_subplot(grid[0, 2], sharey=label_axis)
        right = figure.add_subplot(grid[0, 3], sharey=label_axis)
        stat_rows = []
        all_values = ([], [])
        for index, entry in enumerate(rows):
            records = entry["records"]
            positions = _offsets(entry["run_id"], records)
            name = entry["name"]
            label_axis.text(
                0.99,
                index,
                name,
                transform=label_axis.transData,
                ha="right",
                va="center",
                color=BLACK,
                fontsize=7.25,
            )
            count_axis.text(
                0.5,
                index,
                f"{len(records)}",
                transform=count_axis.transData,
                ha="center",
                va="center",
                color=BLACK,
                fontsize=7.0,
            )
            for panel_index, (axis, field) in enumerate(
                ((left, "_duration"), (right, "_normalized"))
            ):
                points = [
                    (slide, record[field])
                    for slide, record in records.items()
                    if record[field] is not None
                ]
                if not points:
                    continue
                points.sort(key=lambda pair: pair[0])
                values = np.array([value for _, value in points], dtype=float)
                all_values[panel_index].extend(values.tolist())
                if values.size >= 2:
                    box = axis.boxplot(
                        [values],
                        positions=[index],
                        vert=False,
                        widths=0.34,
                        whis=1.5,
                        showfliers=False,
                        patch_artist=True,
                        manage_ticks=False,
                        boxprops={
                            "facecolor": to_rgba(PALE_CYAN, 0.38),
                            "edgecolor": BLACK,
                            "linewidth": 0.60,
                        },
                        medianprops={"color": BLACK, "linewidth": 0.80},
                        whiskerprops={"color": BLACK, "linewidth": 0.50},
                        capprops={"color": BLACK, "linewidth": 0.50},
                        zorder=2,
                    )
                    for patch in box["boxes"]:
                        patch.set_zorder(2)
                offsets = np.array([positions[slide] for slide, _ in points])
                axis.scatter(
                    values,
                    index + offsets,
                    s=5.3,
                    color=CYAN,
                    alpha=0.22,
                    linewidths=0,
                    zorder=3,
                )
                if values.size >= 2:
                    for patch in box["boxes"]:
                        axis.add_patch(
                            PathPatch(
                                patch.get_path(),
                                transform=patch.get_transform(),
                                facecolor="none",
                                edgecolor=BLACK,
                                linewidth=0.60,
                                zorder=5,
                            )
                        )
                    for artist in box["medians"] + box["whiskers"] + box["caps"]:
                        artist.set_zorder(6)
                stat_rows.append(
                    {
                        "run_id": entry["run_id"],
                        "method": entry["name"],
                        "metric": "total_processing_time_s"
                        if panel_index == 0
                        else "total_processing_seconds_per_native_megapixel",
                        "n": len(values),
                        "median": float(np.median(values)),
                        "mean": float(np.mean(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "q3": float(np.quantile(values, 0.75)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
        for axis in (label_axis, count_axis, left, right):
            axis.set_ylim(len(rows) - 0.54, -0.54)
            axis.set_facecolor("white")
            axis.grid(False)
            axis.set_yticks([])
            axis.tick_params(
                axis="y", which="both", length=0, labelleft=False, labelright=False
            )
        for axis in (label_axis, count_axis):
            axis.set_xlim(0, 1)
            axis.set_xticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        count_axis.text(
            0.5,
            -0.95,
            "n",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=7.3,
            color=BLACK,
        )
        for axis, values in ((left, all_values[0]), (right, all_values[1])):
            axis.set_xscale("log")
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.spines["bottom"].set_visible(True)
            axis.spines["bottom"].set_color(BLACK)
            axis.spines["bottom"].set_linewidth(0.42)
            axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=12))
            axis.xaxis.set_major_formatter(FuncFormatter(_format_log_tick))
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.tick_params(
                axis="x",
                which="major",
                labelcolor=BLACK,
                color=BLACK,
                labelsize=7.1,
                width=0.42,
                length=2.4,
                pad=4,
            )
            if values:
                lo, hi = min(values), max(values)
                if lo == hi:
                    axis.set_xlim(lo / 1.6, hi * 1.6)
                else:
                    span = math.log10(hi / lo)
                    padding = max(0.055 * span, 0.065)
                    axis.set_xlim(
                        10 ** (math.log10(lo) - padding),
                        10 ** (math.log10(hi) + padding),
                    )
        left.set_xlabel(
            "Processing time (s per slide)", color=BLACK, fontsize=7.6, labelpad=6
        )
        right.set_xlabel(
            "Processing time (s per native MP)", color=BLACK, fontsize=7.6, labelpad=6
        )
        left.set_title(
            "a  Total processing time",
            loc="left",
            fontsize=8,
            fontweight="bold",
            color=BLACK,
            pad=11,
        )
        right.set_title(
            "b  Size-normalized time",
            loc="left",
            fontsize=8,
            fontweight="bold",
            color=BLACK,
            pad=11,
        )
        figure.subplots_adjust(
            left=0.024, right=0.991, bottom=max(0.12, 0.47 / height), top=0.93
        )
        output.mkdir(parents=True, exist_ok=True)
        generated = []
        for extension in formats:
            path = output / f"timing_boxdots.{extension}"
            figure.savefig(
                path,
                dpi=dpi if extension == "png" else None,
                bbox_inches="tight",
                pad_inches=0.07,
                facecolor="white",
            )
            generated.append(str(path))
        plt.close(figure)
        return generated, stat_rows


def generate_timing_figures(
    timings,
    output_dir,
    min_slides=1,
    matched=False,
    exclude_first_call=False,
    formats=("png", "svg"),
    dpi=300,
):
    if not set(formats) <= {"png", "svg"} or not formats:
        raise ValueError("formats must contain png or svg")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    methods = _read_timings(timings, min_slides, matched, exclude_first_call)
    rows = _prepare(methods)
    output = Path(output_dir)
    generated, stats = _plot(rows, output, formats, dpi)
    with (output / "timing_plot_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "method",
                "metric",
                "n",
                "median",
                "mean",
                "q1",
                "q3",
                "minimum",
                "maximum",
            ],
        )
        writer.writeheader()
        writer.writerows(stats)
    manifest = {
        "timings": str(Path(timings).resolve()),
        "matched": matched,
        "excluded_first_call": exclude_first_call,
        "min_slides": min_slides,
        "method_count": len(rows),
        "n_unique_slides": len(set().union(*(set(m["records"]) for m in rows))),
        "method_order": [entry["run_id"] for entry in rows],
        "figures": generated,
        "normalized_measure": "total_processing_time_s / (native_pixels / 1000000)",
        "point_definition": "one valid slide-method timing record per panel",
        "box_definition": "25th-75th percentile, median, 1.5 IQR whiskers; no separate outlier markers",
    }
    (output / "timing_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot recorded segmentation runtimes without rerunning inference"
    )
    parser.add_argument(
        "--timings",
        type=Path,
        required=True,
        help="Benchmark output directory containing method/timings/*.json, or benchmark_timings.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-slides", type=int, default=1)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--exclude-first-call", action="store_true")
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg"), default=("png", "svg")
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    source = args.timings.expanduser()
    output = (
        args.output_dir
        or (source if source.is_dir() else source.parent) / "figures" / "timings"
    )
    result = generate_timing_figures(
        args.timings,
        output,
        min_slides=args.min_slides,
        matched=args.matched,
        exclude_first_call=args.exclude_first_call,
        formats=args.formats,
        dpi=args.dpi,
    )
    print(
        f"Timing figures created: {result['method_count']} methods, {result['n_unique_slides']} slides; {output}"
    )


if __name__ == "__main__":
    main()

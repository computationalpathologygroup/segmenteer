from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from segmenteer.io.asap import load_asap_xml
from segmenteer.io.predictions import read_prediction
from segmenteer.metrics.supervised import _union
from segmenteer.wsi.reader import NativeWSIReader
from .metadata import normalize_slide
from .performance import _method_name

CYAN = "#00AFDC"
AMBER = "#FFB000"
MAGENTA = "#DC267F"
DEFAULT_GT_ALPHA = 0.26
DEFAULT_TP_ALPHA = 0.32
DEFAULT_FN_ALPHA = 0.55
DEFAULT_FP_ALPHA = 0.50
SLIDE_SUFFIXES = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".qptiff"}
SELECTION_RULES = ("challenging", "disagreement", "under", "over")


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _read_scored_metrics(path):
    scores = defaultdict(dict)
    names = {}
    tolerances = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("evaluation_mode") != "supervised" or row.get(
                "evaluation_error"
            ):
                continue
            nsd = _number(row.get("nsd"))
            under = _number(row.get("under_segmentation_rate"))
            over = _number(row.get("over_segmentation_rate"))
            gt_area = _number(row.get("ground_truth_area_mm2"))
            tolerance = _number(row.get("boundary_tolerance_um"))
            if (
                nsd is None
                or under is None
                or over is None
                or gt_area is None
                or gt_area <= 0
                or tolerance is None
            ):
                continue
            if not 0 <= nsd <= 1 or not 0 <= under <= 1 or over < 0 or tolerance <= 0:
                raise ValueError(
                    "Metrics contain invalid NSD, under/over-segmentation rates, or NSD tolerance"
                )
            run_id = row.get("run_id", "").strip()
            filename = Path(row.get("filename", "")).name
            if (
                not run_id
                or Path(run_id).name != run_id
                or run_id in {".", ".."}
                or not filename
                or filename != row.get("filename", "")
            ):
                raise ValueError(
                    "Evaluation metrics must contain a valid basename-only run_id and filename"
                )
            slide = normalize_slide(filename)
            if slide in scores[run_id]:
                raise ValueError(
                    f"Duplicate successful evaluation for {run_id!r} and {slide!r}"
                )
            scores[run_id][slide] = {
                **row,
                "nsd_number": nsd,
                "under_number": under,
                "over_number": over,
                "gt_area_number": gt_area,
            }
            names[run_id] = _method_name(row.get("formal_name", ""), run_id)
            tolerances.add(round(tolerance, 9))
    if len(tolerances) > 1:
        raise ValueError(
            "NSD was evaluated at different boundary tolerances; rerun evaluation with a consistent tolerance"
        )
    return scores, names, next(iter(tolerances)) if tolerances else None


def select_qualitative_cases(
    metrics,
    *,
    min_method_slides=100,
    min_matched_slides=20,
    top_methods=3,
    categories=SELECTION_RULES,
):
    if min_method_slides < 1 or min_matched_slides < 4 or top_methods < 1:
        raise ValueError(
            "Minimum counts must be positive; at least four matched slides are required"
        )
    if (
        not categories
        or len(set(categories)) != len(categories)
        or any(item not in SELECTION_RULES for item in categories)
    ):
        raise ValueError(f"Categories must be unique selections from {SELECTION_RULES}")
    scores, names, tolerance = _read_scored_metrics(metrics)
    eligible = {
        rid: values
        for rid, values in scores.items()
        if len(values) >= min_method_slides
    }
    if len(eligible) < top_methods:
        raise ValueError(
            f"Only {len(eligible)} configurations have at least {min_method_slides} scores; need {top_methods}. Reduce --min-method-slides if appropriate"
        )
    shared = set.intersection(*(set(values) for values in eligible.values()))
    if len(shared) < min_matched_slides:
        raise ValueError(
            f"Only {len(shared)} common successfully evaluated slides across {len(eligible)} methods; need {min_matched_slides}. Change cohort selection before plotting"
        )
    ranking = sorted(
        eligible,
        key=lambda rid: (
            -float(np.mean([eligible[rid][slide]["nsd_number"] for slide in shared])),
            rid,
        ),
    )
    shown = ranking[:top_methods]
    entries = []
    for slide in sorted(shared):
        vals = [eligible[rid][slide] for rid in eligible]
        nsd = np.asarray([v["nsd_number"] for v in vals], dtype=float)
        gt_areas = np.asarray([v["gt_area_number"] for v in vals], dtype=float)
        entries.append(
            {
                "slide_id": slide,
                "filename": vals[0]["filename"],
                "challenging": float(np.quantile(nsd, 0.75)),
                "disagreement": float(np.quantile(nsd, 0.75) - np.quantile(nsd, 0.25)),
                "under": float(np.median([v["under_number"] for v in vals])),
                "over": float(np.median([v["over_number"] for v in vals])),
                "gt_area": float(np.median(gt_areas)),
                "mean_nsd": float(nsd.mean()),
            }
        )
    area_floor = float(np.quantile([item["gt_area"] for item in entries], 0.25))
    chosen = []
    used = set()
    for category in categories:
        candidates = [
            item
            for item in entries
            if item["slide_id"] not in used
            and (category not in {"under", "over"} or item["gt_area"] >= area_floor)
        ]
        if not candidates:
            raise ValueError(f"No available, unique slide for {category!r}")
        candidates.sort(
            key=lambda item: (
                item[category] if category == "challenging" else -item[category],
                item["slide_id"],
            )
        )
        selected = dict(candidates[0])
        selected["category"] = category
        selected["selection_statistic"] = selected[category]
        chosen.append(selected)
        used.add(selected["slide_id"])
    label_counts = Counter(names[rid] for rid in shown)
    method_info = []
    for rid in shown:
        display = names[rid]
        if label_counts[display] > 1:
            match = re.search(r"cfg=([0-9a-f]+)", rid)
            display += " · " + (match.group(1)[:6] if match else rid[-6:])
        method_info.append(
            {
                "run_id": rid,
                "name": display,
                "mean_nsd": float(
                    np.mean([eligible[rid][slide]["nsd_number"] for slide in shared])
                ),
                "n_matched": len(shared),
            }
        )
    return (
        chosen,
        method_info,
        {
            "matched_slides": len(shared),
            "eligible_methods": len(eligible),
            "nsd_tolerance_um": tolerance,
            "area_floor_mm2": area_floor,
            "categories": list(categories),
        },
    )


def _find_slides(root):
    slides = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.casefold() in SLIDE_SUFFIXES:
            key = path.stem.casefold()
            if key in slides:
                raise ValueError(
                    f"Ambiguous WSI stem {key!r}: {slides[key]} and {path}"
                )
            slides[key] = path
    return slides


def _annotation_path(root, stem):
    candidates = [Path(root) / (stem + suffix) for suffix in (".xml", ".geojson")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one XML or GeoJSON ground truth for {stem!r} in {root}; found {len(found)}"
        )
    return found[0]


def _read_geometry(path):
    payload = (
        load_asap_xml(path, groups=("tissue",))
        if path.suffix.casefold() == ".xml"
        else read_prediction(path)
    )
    return _union(payload)


def _thumbnail(path, max_size):
    reader = NativeWSIReader()
    slide = reader.read(path)
    (width, height), (mpp_x, mpp_y) = reader.get_size_and_mpp(slide)
    if min(width, height) <= 0:
        raise ValueError(f"Invalid WSI dimensions for {path}")
    downsample = max(1.0, width / max_size, height / max_size)
    image = reader.get_wsi_at_mpp(slide, (mpp_x * downsample, mpp_y * downsample))
    return np.asarray(image[..., :3], dtype=np.uint8), (width, height), (mpp_x, mpp_y)


def _geometry_mask(geom, slide_size, image_shape):
    import cv2
    from shapely.geometry import box

    width, height = slide_size
    rows, columns = image_shape[:2]
    mask = np.zeros((rows, columns), dtype=np.uint8)
    area = geom.intersection(box(0, 0, width, height))
    if area.is_empty:
        return mask.astype(bool)
    if area.geom_type == "Polygon":
        polygons = [area]
    elif area.geom_type == "MultiPolygon":
        polygons = list(area.geoms)
    elif area.geom_type == "GeometryCollection":
        polygons = [part for part in area.geoms if part.geom_type == "Polygon"]
    else:
        raise ValueError(f"Unexpected geometry type: {area.geom_type}")

    def raster_points(ring):
        coords = np.asarray(ring.coords, dtype=np.float64)
        x = np.round((coords[:, 0] + 0.5) * columns / width - 0.5)
        y = np.round((coords[:, 1] + 0.5) * rows / height - 0.5)
        return np.clip(np.column_stack((x, y)), -2_000_000, 2_000_000).astype(np.int32)

    for polygon in polygons:
        cv2.fillPoly(mask, [raster_points(polygon.exterior)], 255)
        for ring in polygon.interiors:
            cv2.fillPoly(mask, [raster_points(ring)], 0)
    return mask != 0


def _rgb(hex_string):
    return np.asarray(
        [int(hex_string[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32
    )


def _validate_opacities(gt_alpha, tp_alpha, fn_alpha, fp_alpha, outline_width):
    for name, alpha in (
        ("gt_alpha", gt_alpha),
        ("tp_alpha", tp_alpha),
        ("fn_alpha", fn_alpha),
        ("fp_alpha", fp_alpha),
    ):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if not isinstance(outline_width, int) or not 0 <= outline_width <= 5:
        raise ValueError("outline_width must be an integer between 0 and 5")


def _draw_outline(image, mask, color, thickness=1, *, shadow=False):
    import cv2

    if not mask.any() or thickness <= 0:
        return
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return
    if shadow:
        halo = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(halo, contours, -1, 255, thickness + 2, lineType=cv2.LINE_AA)
        strength = halo.astype(np.float32) * (0.45 / 255.0)
        image[:] = np.rint(image.astype(np.float32) * (1 - strength[..., None])).astype(
            np.uint8
        )
    cv2.drawContours(
        image,
        contours,
        -1,
        tuple(int(value) for value in _rgb(color)),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _overlay(
    image,
    truth,
    prediction=None,
    *,
    gt_alpha=DEFAULT_GT_ALPHA,
    tp_alpha=DEFAULT_TP_ALPHA,
    fn_alpha=DEFAULT_FN_ALPHA,
    fp_alpha=DEFAULT_FP_ALPHA,
    outline_width=1,
):
    _validate_opacities(gt_alpha, tp_alpha, fn_alpha, fp_alpha, outline_width)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("WSI thumbnail must be an RGB uint8 image")
    if truth.shape != image.shape[:2] or (
        prediction is not None and prediction.shape != truth.shape
    ):
        raise ValueError(
            "Ground truth and prediction masks must match WSI thumbnail dimensions"
        )
    truth = np.asarray(truth, dtype=bool)
    if prediction is None:
        layers = ((truth, CYAN, gt_alpha),)
    else:
        prediction = np.asarray(prediction, dtype=bool)
        layers = (
            (truth & prediction, CYAN, tp_alpha),
            (truth & ~prediction, AMBER, fn_alpha),
            (~truth & prediction, MAGENTA, fp_alpha),
        )
    result = image.astype(np.float32).copy()
    for mask, color, alpha in layers:
        if alpha and mask.any():
            result[mask] = (1 - alpha) * result[mask] + alpha * _rgb(color)
    result = np.rint(np.clip(result, 0, 255)).astype(np.uint8)
    if outline_width:
        if prediction is not None:
            _draw_outline(result, truth & ~prediction, AMBER, outline_width)
            _draw_outline(result, ~truth & prediction, MAGENTA, outline_width)
        _draw_outline(result, truth, CYAN, outline_width, shadow=True)
    return result


def _save_figure(
    images, methods, cases, destination, formats, dpi, slide_mpp, slide_sizes
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    nrows, ncols = len(cases), len(methods) + 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.12 * ncols + 0.3, 2.58 * nrows + 0.35),
        squeeze=False,
        facecolor="white",
    )
    fonts = {entry.name for entry in font_manager.fontManager.ttflist}
    with plt.rc_context(
        {
            "font.family": next(
                (
                    font
                    for font in ("Arial", "Liberation Sans", "DejaVu Sans")
                    if font in fonts
                ),
                "DejaVu Sans",
            ),
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    ):
        for row, case in enumerate(cases):
            for col in range(ncols):
                ax = axes[row, col]
                ax.imshow(images[row][col], interpolation="nearest", aspect="equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(
                        "Ground truth" if col == 0 else methods[col - 1]["name"],
                        fontsize=8.3,
                        color="black",
                        pad=6,
                    )
                if col == 0:
                    ax.text(
                        -0.055,
                        1.02,
                        "abcd"[row] if row < 4 else chr(ord("a") + row),
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=9.1,
                        weight="bold",
                        color="black",
                    )
                    size = slide_sizes[row]
                    mpp = slide_mpp[row]
                    width_mm = size[0] * mpp[0] / 1000
                    lengths = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
                    desired = min(lengths, key=lambda v: abs(v - 0.16 * width_mm))
                    if desired / width_mm < 0.35:
                        fraction = desired / width_mm
                        ax.plot(
                            [0.06, 0.06 + fraction],
                            [0.06, 0.06],
                            color="white",
                            linewidth=3.2,
                            transform=ax.transAxes,
                            solid_capstyle="butt",
                        )
                        ax.plot(
                            [0.06, 0.06 + fraction],
                            [0.06, 0.06],
                            color="black",
                            linewidth=1.7,
                            transform=ax.transAxes,
                            solid_capstyle="butt",
                        )
                        ax.text(
                            0.06,
                            0.086,
                            f"{desired:g} mm",
                            transform=ax.transAxes,
                            color="black",
                            ha="left",
                            va="bottom",
                            fontsize=6.8,
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "pad": 0.3,
                                "alpha": 0.7,
                            },
                        )
        fig.subplots_adjust(
            left=0.045, right=0.995, bottom=0.071, top=0.938, wspace=0.075, hspace=0.105
        )
        handles = [
            Line2D([0], [0], color=CYAN, linewidth=1.6, label="Ground-truth outline"),
            Patch(facecolor=CYAN, label="Correct tissue (TP)"),
            Patch(facecolor=AMBER, label="Missed tissue (FN)"),
            Patch(facecolor=MAGENTA, label="Excess prediction (FP)"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.009),
            ncol=4,
            fontsize=7.2,
            frameon=False,
            handlelength=1.6,
            columnspacing=2.3,
        )
        for extension in formats:
            fig.savefig(
                destination.with_suffix("." + extension),
                dpi=dpi if extension == "png" else None,
                bbox_inches="tight",
                pad_inches=0.045,
            )
    plt.close(fig)


def generate_qualitative_figures(
    *,
    metrics,
    predictions,
    ground_truth,
    data,
    output_dir,
    top_methods=3,
    min_method_slides=100,
    min_matched_slides=20,
    max_image_size=1000,
    categories=SELECTION_RULES,
    formats=("png", "svg"),
    dpi=250,
    gt_alpha=DEFAULT_GT_ALPHA,
    tp_alpha=DEFAULT_TP_ALPHA,
    fn_alpha=DEFAULT_FN_ALPHA,
    fp_alpha=DEFAULT_FP_ALPHA,
    outline_width=1,
):
    _validate_opacities(gt_alpha, tp_alpha, fn_alpha, fp_alpha, outline_width)
    if max_image_size < 128 or dpi < 50:
        raise ValueError("Image size must be at least 128 pixels and DPI at least 50")
    if not formats or set(formats) - {"png", "svg", "pdf"}:
        raise ValueError("Formats must be selected from png, svg, pdf")
    output = Path(output_dir)
    root = Path(predictions)
    gt_root = Path(ground_truth)
    slide_root = Path(data)
    for folder in (root, gt_root, slide_root):
        if not folder.is_dir():
            raise NotADirectoryError(folder)
    cases, methods, protocol = select_qualitative_cases(
        metrics,
        min_method_slides=min_method_slides,
        min_matched_slides=min_matched_slides,
        top_methods=top_methods,
        categories=categories,
    )
    slides = _find_slides(slide_root)
    image_inputs = []
    for case in cases:
        slide = case["slide_id"]
        if slide not in slides:
            raise FileNotFoundError(
                f"Original WSI not found for selected slide {case['filename']}"
            )
        stem = Path(case["filename"]).stem
        gt_path = _annotation_path(gt_root, stem)
        paths = []
        for method in methods:
            choices = [
                root / method["run_id"] / "predictions" / f"{stem}.{extension}"
                for extension in ("xml", "geojson")
            ]
            found = [path for path in choices if path.is_file()]
            if len(found) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one XML or GeoJSON prediction for {stem!r} and {method['run_id']!r}; found {len(found)}"
                )
            paths.append(found[0])
        image_inputs.append((slides[slide], gt_path, paths))
    output.mkdir(parents=True, exist_ok=True)
    all_images, spacings, dimensions = [], [], []
    for case, (wsi_path, gt_path, prediction_paths) in zip(cases, image_inputs):
        thumbnail, size, mpp = _thumbnail(wsi_path, max_image_size)
        gt = _geometry_mask(_read_geometry(gt_path), size, thumbnail.shape)
        row = [_overlay(thumbnail, gt, gt_alpha=gt_alpha, outline_width=outline_width)]
        for path in prediction_paths:
            prediction = _geometry_mask(_read_geometry(path), size, thumbnail.shape)
            row.append(
                _overlay(
                    thumbnail,
                    gt,
                    prediction,
                    tp_alpha=tp_alpha,
                    fn_alpha=fn_alpha,
                    fp_alpha=fp_alpha,
                    outline_width=outline_width,
                )
            )
        all_images.append(row)
        spacings.append(mpp)
        dimensions.append(size)
    target = output / "representative_segmentation"
    _save_figure(all_images, methods, cases, target, formats, dpi, spacings, dimensions)
    fields = [
        "panel",
        "case_type",
        "filename",
        "selection_rule",
        "selection_statistic",
        "mean_nsd",
        "median_under_rate",
        "median_over_rate",
        "gt_area_mm2",
        "source_wsi",
        "source_gt",
    ]
    with (output / "qualitative_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (case, (wsi, gt, _)) in enumerate(zip(cases, image_inputs)):
            writer.writerow(
                {
                    "panel": chr(ord("a") + i),
                    "case_type": case["category"],
                    "filename": case["filename"],
                    "selection_rule": {
                        "challenging": "lowest method-wise 75th percentile NSD",
                        "disagreement": "highest intermethod NSD IQR",
                        "under": "highest median FN/GT among GT-area-eligible slides",
                        "over": "highest median FP/GT among GT-area-eligible slides",
                    }[case["category"]],
                    "selection_statistic": case["selection_statistic"],
                    "mean_nsd": case["mean_nsd"],
                    "median_under_rate": case["under"],
                    "median_over_rate": case["over"],
                    "gt_area_mm2": case["gt_area"],
                    "source_wsi": str(wsi),
                    "source_gt": str(gt),
                }
            )
    with (output / "qualitative_method_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "panel",
                "filename",
                "method_rank",
                "run_id",
                "method",
                "global_matched_mean_nsd",
                "nsd",
                "dice",
                "under_segmentation_rate",
                "over_segmentation_rate",
            ]
        )
        records, _, _ = _read_scored_metrics(metrics)
        for i, case in enumerate(cases):
            for rank, method in enumerate(methods, 1):
                row = records[method["run_id"]][case["slide_id"]]
                writer.writerow(
                    [
                        chr(ord("a") + i),
                        case["filename"],
                        rank,
                        method["run_id"],
                        method["name"],
                        method["mean_nsd"],
                        row["nsd_number"],
                        row.get("dice"),
                        row["under_number"],
                        row["over_number"],
                    ]
                )
    manifest = {
        "figure": [str(target.with_suffix("." + ext)) for ext in formats],
        "source_metrics": str(Path(metrics).resolve()),
        "predictions": str(root.resolve()),
        "ground_truth": str(gt_root.resolve()),
        "data": str(slide_root.resolve()),
        "selection": protocol,
        "methods": methods,
        "case_selection": [
            {
                k: v
                for k, v in case.items()
                if k in ("filename", "category", "selection_statistic")
            }
            for case in cases
        ],
        "colors": {
            "ground_truth": CYAN,
            "correct_prediction": CYAN,
            "missed_tissue": AMBER,
            "excess_prediction": MAGENTA,
        },
        "opacity": {
            "ground_truth": gt_alpha,
            "true_positive": tp_alpha,
            "false_negative": fn_alpha,
            "false_positive": fp_alpha,
        },
        "outline_width_px": outline_width,
        "notes": [
            "Slide selection uses all eligible configurations on their common successful evaluation cohort.",
            "The same globally top-ranked methods are displayed for every slide.",
            "Original WSI is the background in every panel; ground truth is overlaid in the first column.",
            "Prediction panels depict correct tissue, missed tissue, and excess predicted tissue.",
            "Masks are rasterized for visualization at thumbnail resolution; saved evaluation metrics are not recomputed.",
        ],
    }
    (output / "qualitative_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest

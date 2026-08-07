"""Build a viewer index from official-results CSVs and selected predictions.

This loader is deliberately narrower than :mod:`app.loader`: it does not scan
all method folders under an outputs/ run.  Instead it treats the official
metrics CSVs as the source of truth and exposes only method/slide pairs that
were actually present in the selected official evaluation cohort.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.loader import IndexData, MethodInfo, WSIRecord
from app.loader import _generate_colors, _parse_mpp, _parse_params

_WSI_EXTENSIONS: tuple[str, ...] = (
    ".tiff", ".tif", ".svs", ".ndpi", ".qptiff", ".scn", ".mrxs",
)

HOLES_ARE_TISSUE = "holes_are_tissue"
HOLES_ARE_NOT_TISSUE = "holes_are_not_tissue"
HOLE_MODES = (HOLES_ARE_TISSUE, HOLES_ARE_NOT_TISSUE)
DEFAULT_HOLE_MODE = HOLES_ARE_NOT_TISSUE
DEFAULT_RESULTS_DIR = "official-results-8-July"
DEFAULT_OUTPUT_PREFIX = "metrics"

# Keep this order and naming aligned with the official manuscript outputs.
OFFICIAL_METHOD_SPECS: list[tuple[str, str]] = [
    ("trident_cpg__batch-size=8_holes-are-tissue=1_model-id=CPG_num-workers=0_target-mag=4__cfg=2bef911e6123", "CPG"),
    ("atlaspatch_sam2__checkpoint-path=None_device=cpu_mask-threshold=0.5_min-area=0__cfg=8e964bfe8daa", "AtlasPatch-SAM2"),
    ("trident_hest__batch-size=8_holes-are-tissue=1_model-id=HEST_num-workers=0_target-mag=10__cfg=ff39dab66a9b", "HEST"),
    ("trident_grandqc__batch-size=8_holes-are-tissue=1_model-id=GrandQC_num-workers=0_target-mag=1__cfg=8b9b772fc380", "GRANDQC"),
    ("trident_pathprofiler__batch-size=8_holes-are-tissue=1_model-id=PathProfiler_num-workers=0_target-mag=4__cfg=e5855f9f9e6a", "PathProfiler"),
    ("rtlucassen_slidesegmenter__min-area=0_mpp=7.04__cfg=70f008ac72e4", "SlideSegmenter"),
    ("fastsam_fastsam-x__conf=0.4_device=cpu_imgsz=1024_iou=0.9_min-area=0_model-name=FastSAM-x.pt_mpp=10_text-prompt=None__cfg=a07d5e6d7015", "FastSAM-x"),
    ("bigpicture__apply-hole-filling=1_confidence-threshold=0.8_device=cpu_dilate-mask=1_dilation-disk-size=32_min-area=0_mpp=8_select-largest-tissue-objects=0__cfg=a5319c27bb2c", "BigPicture"),
    ("entropy_masker__footprint=None_min-area=0_mpp=5__cfg=df2aff0aaab6", "EntropyMasker (mpp=5)"),
    ("entropy_masker__footprint=None_min-area=0_mpp=10__cfg=3804bdc5e9e0", "EntropyMasker (mpp=10)"),
    ("entropy_masker__footprint=None_min-area=0_mpp=20__cfg=b5d680d4d04e", "EntropyMasker (mpp=20)"),
    ("fesi__improved=0_min-area=0_mpp=5__cfg=4cd42299b1c8", "FESI (mpp=5)"),
    ("fesi__improved=0_min-area=0_mpp=10__cfg=4a5b3c4def6c", "FESI (mpp=10)"),
    ("fesi__improved=0_min-area=0_mpp=20__cfg=5cb283400625", "FESI (mpp=20)"),
    ("improved_fesi__improved=1_min-area=0_mpp=5__cfg=1817278ca330", "FESI improved (mpp=5)"),
    ("improved_fesi__improved=1_min-area=0_mpp=10__cfg=ee65ab73a479", "FESI improved (mpp=10)"),
    ("improved_fesi__improved=1_min-area=0_mpp=20__cfg=e22e644cf1ae", "FESI improved (mpp=20)"),
    ("hsv-threshold__lower=908103_min-area=0_mpp=5_upper=180255255__cfg=9ce66e397d91", "HSV (mpp=5)"),
    ("hsv-threshold__lower=908103_min-area=0_mpp=10_upper=180255255__cfg=cda10b7db15d", "HSV (mpp=10)"),
    ("hsv-threshold__lower=908103_min-area=0_mpp=20_upper=180255255__cfg=59747cd7a49d", "HSV (mpp=20)"),
    ("otsu_tissue__min-area=0_mpp=10__cfg=2c1a2af0bb29", "Otsu"),
    ("otsu__min-area=0_mpp=20__cfg=74c73a92dd68", "Otsu (mpp=20)"),
    ("li_tissue__min-area=0_mpp=20__cfg=2890821281f1", "Li (mpp=20)"),
    ("yen_tissue__min-area=0_mpp=20__cfg=1d6463db8975", "Yen (mpp=20)"),
]
METHOD_LABELS = dict(OFFICIAL_METHOD_SPECS)
OFFICIAL_METHOD_ORDER = [method_id for method_id, _ in OFFICIAL_METHOD_SPECS]
OFFICIAL_METHOD_RANK = {method_id: index for index, method_id in enumerate(OFFICIAL_METHOD_ORDER)}

METRIC_NAMES: tuple[str, ...] = (
    "dice",
    "iou",
    "hausdorff",
    "precision",
    "recall",
    "over_segmentation_rate",
    "under_segmentation_rate",
    "pixel_accuracy",
    "mae",
    "balanced_error_rate",
)


@dataclass(frozen=True)
class OfficialSource:
    path: Path
    hole_mode: str
    method_id: str
    method_label: str


def load_official_index(
    run_root: Path,
    metrics_dir: Path,
    *,
    data_dir: Path | None = None,
    ground_truth_dir: Path | None = None,
    hole_mode: str = DEFAULT_HOLE_MODE,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> IndexData:
    """Return the viewer index for a selected official-results cohort.

    Parameters
    ----------
    run_root:
        The original output run directory, for example ``outputs/20260701_155505``.
        Prediction polygons are read from ``<run_root>/<method>/predictions``.
    metrics_dir:
        Directory containing official CSVs named
        ``metrics__<hole_mode>__<method>.csv``.
    data_dir:
        Optional directory with source WSI files.  Only slides present in the
        selected official metrics are listed; this directory is used only to
        resolve image paths.
    ground_truth_dir:
        Optional GeoJSON ground-truth directory.  The app uses it for the
        ground-truth endpoint; the index marks a slide as having GT when the
        corresponding file exists or when the metrics row is evaluated.
    hole_mode:
        Usually ``holes_are_not_tissue`` for the official viewer.
    """
    run_root = Path(run_root)
    metrics_dir = Path(metrics_dir)
    data_dir = Path(data_dir) if data_dir else None
    ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None

    if hole_mode not in HOLE_MODES:
        raise ValueError(f"Unsupported hole mode: {hole_mode!r}")
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Official metrics directory does not exist: {metrics_dir}")

    sources = discover_official_sources(metrics_dir, hole_mode=hole_mode, output_prefix=output_prefix)
    # Important for interactive use on network volumes: do not scan --data at
    # startup.  Source WSI paths are resolved lazily by the server only when a
    # slide thumbnail/viewer is actually requested.

    # Retain only official methods whose prediction directory exists in this
    # output run.  This avoids surfacing stale/incomplete methods from the run
    # root and makes the metrics directory authoritative.
    valid_sources: list[OfficialSource] = []
    for source in sources:
        if (run_root / source.method_id / "predictions").is_dir():
            valid_sources.append(source)

    valid_sources = sorted(
        valid_sources,
        key=lambda source: (OFFICIAL_METHOD_RANK.get(source.method_id, 10_000), source.method_label.casefold()),
    )

    colors = _generate_colors(len(valid_sources))
    methods: dict[str, MethodInfo] = {}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    all_stems: set[str] = set()
    image_paths: dict[str, Path] = {}

    for index, source in enumerate(valid_sources):
        methods[source.method_id] = MethodInfo(
            run_id=source.method_id,
            name=source.method_label,
            mpp=_parse_mpp(source.method_id),
            params=_parse_params(source.method_id),
            color=colors[index],
        )

        prediction_dir = run_root / source.method_id / "predictions"
        available_prediction_stems = {path.stem for path in prediction_dir.glob("*.geojson")}
        for row in _read_csv_rows(source.path):
            parsed = _parse_official_metric_row(row)
            if parsed is None:
                continue
            stem, entry = parsed

            # The official metrics CSV is the cohort definition. Keep every
            # evaluated row in the index/summary so the dataset count matches
            # the official ground-truth cohort. Prediction overlays remain
            # gated separately and are offered only when the GeoJSON exists.
            entry["_prediction_available"] = stem in available_prediction_stems
            entry["official_metrics_file"] = str(source.path)
            entry["official_hole_mode"] = source.hole_mode
            scores.setdefault(stem, {})[source.method_id] = entry
            all_stems.add(stem)

            # Avoid thousands of per-row filesystem probes. If the CSV stores an
            # explicit image path, honour it; otherwise image paths are backfilled
            # once from the data-directory scan below.
            stored_image = str(row.get("image_path") or "").strip()
            if stored_image:
                image = _resolve_stored_image_path(stored_image, stem, data_dir)
                if image is not None:
                    image_paths[stem] = image

    # Do not backfill image paths by scanning --data here.  For speed, the
    # server resolves <data>/<stem>.<ext> lazily on the first WSI request.

    gt_stems = {path.stem for path in ground_truth_dir.glob("*.geojson")} if ground_truth_dir and ground_truth_dir.is_dir() else set()
    # The official viewer's dataset denominator must be the evaluated official
    # cohort, not every GeoJSON in the ground-truth directory.  The ground-truth
    # directory can contain extra annotations that were not part of the selected
    # official metrics run; those are intentionally *not* added to all_stems.
    # Ground-truth availability is still recorded below for the evaluated slides.

    wsis: list[WSIRecord] = []
    for stem in sorted(all_stems, key=str.casefold):
        gt_exists = stem in gt_stems
        wsis.append(
            WSIRecord(
                stem=stem,
                image_path=str(image_paths[stem]) if stem in image_paths else None,
                # Optimistic when --data is supplied: exact WSI path lookup is
                # lazy and cheap per slide, avoiding a full startup scan.
                has_wsi=(stem in image_paths) or bool(data_dir and data_dir.is_dir()),
                has_ground_truth=gt_exists or _has_any_supervised_metrics(scores.get(stem, {})),
                ground_truth_url=f"/api/ground-truth/{stem}",
                run_status="supervised",
            )
        )

    return IndexData(output_dir=str(run_root), wsis=wsis, methods=methods, scores=scores)


def discover_official_sources(
    metrics_dir: Path,
    *,
    hole_mode: str = DEFAULT_HOLE_MODE,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> list[OfficialSource]:
    pattern = re.compile(
        rf"^{re.escape(output_prefix)}__(?P<hole_mode>{HOLES_ARE_TISSUE}|{HOLES_ARE_NOT_TISSUE})__(?P<method>.+)\.csv$"
    )
    sources: list[OfficialSource] = []
    for path in sorted(metrics_dir.glob(f"{output_prefix}__{hole_mode}__*.csv"), key=lambda p: p.name.casefold()):
        match = pattern.match(path.name)
        if not match or match.group("hole_mode") != hole_mode:
            continue
        method_id = match.group("method")
        sources.append(
            OfficialSource(
                path=path,
                hole_mode=hole_mode,
                method_id=method_id,
                method_label=method_label(method_id),
            )
        )
    return sources


def method_label(method_id: str) -> str:
    known = METHOD_LABELS.get(method_id)
    if known is not None:
        return known
    family, _, parameters = method_id.partition("__")
    extras: list[str] = []
    for key in ("mpp", "model-id", "target-mag", "model-name"):
        match = re.search(rf"(?:^|_){re.escape(key)}=([^_]+)", parameters)
        if match:
            extras.append(f"{key}={match.group(1)}")
    label = family.replace("_", "-")
    return f"{label} ({', '.join(extras[:2])})" if extras else label


def _resolve_stored_image_path(stored: str | None, stem: str, data_dir: Path | None) -> Path | None:
    """Resolve explicit image paths without scanning the data directory."""
    if stored:
        candidate = Path(stored).expanduser()
        if candidate.is_file():
            return candidate
        if data_dir and not candidate.is_absolute():
            joined = data_dir / candidate
            if joined.is_file():
                return joined
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _parse_official_metric_row(row: dict[str, str]) -> tuple[str, dict[str, Any]] | None:
    filename = str(row.get("filename") or row.get("slide_id") or row.get("image_id") or "").strip()
    stem = Path(filename).stem
    if not stem:
        return None

    status = str(row.get("evaluation_status") or "").strip().casefold()
    if status in {"failed", "no_ground_truth", "skipped"} or status.startswith("failed"):
        return None

    supervised: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        value = _finite_float(row.get(metric))
        if value is not None:
            supervised[metric] = value

    # Official vector-evaluation CSVs can have ground_truth_available=False even
    # when evaluation_status=evaluated_vector and Dice/IoU are valid.  Numeric
    # metrics + non-failed status are therefore the inclusion criterion.
    if not supervised:
        return None

    supervised["pixel_metrics_computed"] = _parse_bool(row.get("pixel_metrics_computed"), default=False)
    entry: dict[str, Any] = {
        "metrics": {"supervised": supervised},
        "postrun_evaluation": {
            "status": row.get("evaluation_status") or "",
            "error": row.get("evaluation_error") or "",
        },
        "run_id": row.get("run_id") or row.get("segmentation_method") or "",
        "segmentation_method": row.get("segmentation_method") or row.get("run_id") or "",
    }
    for key in ("execution_time_s", "seconds_per_megapixel", "native_spacing", "magnification_used"):
        value = _finite_float(row.get(key))
        if value is not None:
            entry[key] = value
    return stem, entry


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().casefold()
    if not text:
        return default
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _has_any_supervised_metrics(per_method: dict[str, dict[str, Any]]) -> bool:
    return any(isinstance(entry.get("metrics", {}).get("supervised"), dict) for entry in per_method.values())

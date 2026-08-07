from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

try:
    from shapely import make_valid as _shapely_make_valid
except ImportError:  # Shapely < 2
    _shapely_make_valid = None


# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "xml_output" / "ground_truth_geojson"
OUTPUT_DIR = PROJECT_ROOT / "OFFICIAL_FULL"
METRICS_DB = OUTPUT_DIR / "official_metrics.sqlite3"
METRICS_CSV = OUTPUT_DIR / "official_metrics.csv"  # written only by the snapshot exporter

# Used only for the metadata-stratified tables/figures. If neither file exists,
# the core evaluation, Table 1, and Figures 1–3 are still generated.
ANNOTATION_CSV_CANDIDATES = (
    PROJECT_ROOT / "benchmark_data_annotation_latest.csv",
    PROJECT_ROOT / "benchmark_data_annotation.csv",
)

# Two entries below were supplied without the common run prefix. The resolver
# first tries each path exactly, then tries this directory, then searches by the
# method-directory basename below PROJECT_ROOT/outputs.
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "20260701_155505"

OFFICIAL_METHOD_SPECS: list[tuple[str, str]] = [
    ("outputs/20260701_155505/atlaspatch_sam2__checkpoint-path=None_device=cpu_mask-threshold=0.5_min-area=0__cfg=8e964bfe8daa", "AtlasPatch-SAM2"),
   
    ("outputs/20260701_155505/trident_hest__batch-size=8_holes-are-tissue=1_model-id=HEST_num-workers=0_target-mag=10__cfg=ff39dab66a9b", "HEST"),
    ("outputs/20260701_155505/trident_grandqc__batch-size=8_holes-are-tissue=1_model-id=GrandQC_num-workers=0_target-mag=1__cfg=8b9b772fc380", "GrandQC"),
    ("outputs/20260802_163446/trident_grandqc__batch-size=8_holes-are-tissue=0_model-id=GrandQC_num-workers=0_target-mag=1__cfg=9558576ae076", "GrandQC (holes are not tissue)"),

    ("outputs/20260701_155505/trident_pathprofiler__batch-size=8_holes-are-tissue=1_model-id=PathProfiler_num-workers=0_target-mag=4__cfg=e5855f9f9e6a", "PathProfiler"),

    ("outputs/20260701_155505/grandqc_tissue_detection_mpp10__checkpoint-path=None_confidence-threshold=0.5_device=cpu_min-area=0_mpp=10__cfg=9ffee21ddef1", "GrandQC (non-TRIDENT, mpp=10)"),

    ("outputs/20260701_155505/rtlucassen_slidesegmenter__min-area=0_mpp=7.04__cfg=70f008ac72e4", "SlideSegmenter"),
   
    ("outputs/20260701_155505/fastsam_fastsam-x__conf=0.4_device=cpu_imgsz=1024_iou=0.9_min-area=0_model-name=FastSAM-x.pt_mpp=10_text-prompt=None__cfg=a07d5e6d7015", "FastSAM-x (mpp=10)"),
    ("outputs/20260701_155505/fastsam_fastsam-x__conf=0.4_device=cpu_imgsz=1024_iou=0.9_min-area=0_model-name=FastSAM-x.pt_mpp=20_text-prompt=None__cfg=d53022d2cb81", "FastSAM-x (mpp=20)"),

    ("outputs/20260701_155505/bigpicture__apply-hole-filling=1_confidence-threshold=0.8_device=cpu_dilate-mask=1_dilation-disk-size=32_min-area=0_mpp=8_select-largest-tissue-objects=0__cfg=a5319c27bb2c", "BigPicture"),
    ("outputs/20260701_155505/bigpicture__apply-hole-filling=0_confidence-threshold=0.8_device=cpu_dilate-mask=1_dilation-disk-size=32_min-area=0_mpp=8_select-largest-tissue-objects=1__cfg=8204843ee298", "BigPicture (no hole filling, largest tissue only)"),
    ("outputs/20260701_155505/bigpicture__apply-hole-filling=1_confidence-threshold=0.8_device=cpu_dilate-mask=1_dilation-disk-size=32_min-area=0_mpp=8_select-largest-tissue-objects=1__cfg=8c08a617055c", "BigPicture (hole filling, largest tissue only)"),

    ("outputs/20260701_155505/entropy_masker__footprint=None_min-area=0_mpp=5__cfg=df2aff0aaab6", "EntropyMasker (mpp=5)"),
    ("outputs/20260701_155505/entropy_masker__footprint=None_min-area=0_mpp=10__cfg=3804bdc5e9e0", "EntropyMasker (mpp=10)"),
    ("outputs/20260701_155505/entropy_masker__footprint=None_min-area=0_mpp=20__cfg=b5d680d4d04e", "EntropyMasker (mpp=20)"),
  
    ("outputs/20260701_155505/fesi__improved=0_min-area=0_mpp=5__cfg=4cd42299b1c8", "FESI (mpp=5)"),
    ("outputs/20260701_155505/fesi__improved=0_min-area=0_mpp=10__cfg=4a5b3c4def6c", "FESI (mpp=10)"),
    ("outputs/20260701_155505/fesi__improved=0_min-area=0_mpp=20__cfg=5cb283400625", "FESI (mpp=20)"),
    
    ("outputs/20260701_155505/improved_fesi__improved=1_min-area=0_mpp=5__cfg=1817278ca330", "FESI improved (mpp=5)"),
    ("outputs/20260701_155505/improved_fesi__improved=1_min-area=0_mpp=10__cfg=ee65ab73a479", "FESI improved (mpp=10)"),
    ("outputs/20260701_155505/improved_fesi__improved=1_min-area=0_mpp=20__cfg=e22e644cf1ae", "FESI improved (mpp=20)"),

    ("outputs/20260701_155505/hsv-threshold__lower=908103_min-area=0_mpp=20_upper=180255255__cfg=59747cd7a49d", "HSV (mpp=20)"),
    ("outputs/20260701_155505/otsu_tissue__min-area=0_mpp=20__cfg=68469bb631ce", "Otsu (mpp=20)"),

    # NOT FINISHED? ("outputs/20260701_155505/hsv-threshold__lower=908103_min-area=0_mpp=5_upper=180255255__cfg=9ce66e397d91", "HSV (mpp=5)"),
   
    ("outputs/20260701_155505/otsu_tissue__min-area=0_mpp=10__cfg=2c1a2af0bb29", "Otsu (mpp=10)"),

    ("outputs/20260701_155505/li_tissue__min-area=0_mpp=20__cfg=2890821281f1", "Li (mpp=20)"),
    ("outputs/20260701_155505/yen_tissue__min-area=0_mpp=20__cfg=1d6463db8975", "Yen (mpp=20)"),

    ("outputs/20260701_155505/histomicstk_tissue_saliency__mask-type=HistomicsTKMaskType.SALI_min-area=0_mpp=20__cfg=0d5aca7db3af", "HistomicsTK (saliency, mpp=20)"),
    ("outputs/20260701_155505/histomicstk_tissue_simple__mask-type=HistomicsTKMaskType.SIMP_min-area=0_mpp=20__cfg=fe7200738085", "HistomicsTK (simple, mpp=20)"),

    ("outputs/20260701_155505/watershed_tissue_mindist100um__min-area=0_min-distance-um=100_mpp=20__cfg=f1b322b74275", "Watershed (mpp=20)"),

    ("outputs/20260701_155505/hsv-threshold__lower=908103_min-area=0_mpp=10_upper=180255255__cfg=cda10b7db15d", "HSV (mpp=10)"),
    ("outputs/20260802_164846/li_tissue__min-area=0_mpp=10__cfg=e6accc9073af", "Li (mpp=10)"),

    ("outputs/20260701_155505/histomicstk_saliency__mask-type=HistomicsTKMaskType.SALI_min-area=0_mpp=10__cfg=ad14a7751ad4", "HistomicsTK (saliency, mpp=10)"),
    # ("outputs/20260701_155505/histomicstk_saliency__mask-type=HistomicsTKMaskType.SALI_min-area=0_mpp=10__cfg=ad14a7751ad4", "HistomicsTK (saliency, mpp=10)"),

    # ("outputs/20260701_155505/yen_tissue__min-area=0_mpp=10__cfg=e3b4a270c130", "Yen (mpp=10)"),

    # ("outputs/20260701_155505/hsv-threshold__lower=908103_min-area=0_mpp=5_upper=180255255__cfg=9ce66e397d91", "HSV (mpp=5)"),

    # ("outputs/20260701_155505/otsu_tissue__min-area=0_mpp=5__cfg=3aa21d881333", "Otsu (mpp=5)"),
    # ("outputs/20260701_155505/li_tissue__min-area=0_mpp=5__cfg=fc357a40782b", "Li (mpp=5)"),
    # ("outputs/20260701_155505/yen_tissue__min-area=0_mpp=5__cfg=f4f36eab6d59", "Yen (mpp=5)"),


]

# Reporting controls.
STD_DDOF = 1                 # sample SD across WSI rows
TOP_K_DISTRIBUTIONS = 5      # Figures 2–3
TOP_K_STRATIFIED = 10        # Figures 4–7
MAX_SOURCE_COLUMNS = 9
MAX_ORGAN_COLUMNS = 15
MAX_STAINING_COLUMNS = 0     # 0 = show all in Figure 6
MIN_STAINING_N_FOR_FIGURE_7 = 3
FLOAT_DECIMALS = 8


# =============================================================================
# Output schema and plotting style
# =============================================================================

METRIC_COLUMNS = (
    "filename",
    "method_name",
    "formal_name",
    "dice_score",
    "iou",
    "oversegmentation",
    "undersegmentation",
)
NUMERIC_METRICS = ("dice_score", "iou", "oversegmentation", "undersegmentation")

# Conceptual method families used when a figure should show one globally best
# configured variant per method. The order here is only a completeness contract;
# figures remain ranked by mean Dice.
METHOD_FAMILY_ORDER = (
    "HEST",
    "SlideSegmenter",
    "AtlasPatch-SAM2",
    "PathProfiler",
    "EntropyMasker",
    "BigPicture",
    "FESI",
    "GrandQC",
    "FastSAM-x",
    "HSV",
    "Otsu",
    "Watershed",
    "Li",
    "Yen",
    "HistomicsTK",
)

# Learned / non-classical method families. In the one-best-variant figures,
# these labels explicitly report the resolution used by the selected run.
MPP_LABEL_METHOD_FAMILIES = frozenset({
    "EntropyMasker",
    "FESI",
    "FastSAM-x",
    "HSV",
    "Otsu",
    "Watershed",
    "Li",
    "Yen",
    "HistomicsTK",
})

MM = 1 / 25.4
COL_W_2 = 180 * MM
INK = "#0f172a"
MID = "#64748b"
ACC = "#2563eb"
ACC_D = "#1d4ed8"
ACC_L = "#dbeafe"
SUPPORT_CMAP = LinearSegmentedColormap.from_list(
    "support", ["#e9f2ff", "#dbeafe", "#bfdbfe", "#60a5fa", "#1d4ed8"], N=256
)
SUPPORT_CMAP.set_bad("#f1f3f5")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.0,
    }
)


@dataclass(frozen=True)
class ResolvedMethod:
    raw_path: str
    method_name: str
    formal_name: str
    predictions_dir: Path
    official_rank: int


@dataclass(frozen=True)
class Metadata:
    data_source: str
    organ: str
    staining: str


# =============================================================================
# Basic helpers
# =============================================================================


def _print(message: str = "") -> None:
    print(message, flush=True)


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.{FLOAT_DECIMALS}f}"


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _compact_label(label: str, *, remove_configuration: bool = False) -> str:
    label = str(label).strip()
    if remove_configuration:
        label = label.split(" (", 1)[0].strip()
    if " (" in label:
        head, tail = label.split(" (", 1)
        return f"{head}\n({tail}"
    replacements = {
        "AtlasPatch-SAM2": "AtlasPatch\nSAM2",
        "SlideSegmenter": "Slide\nSegmenter",
        "PathProfiler": "Path\nProfiler",
    }
    if label in replacements:
        return replacements[label]
    if len(label) > 18 and "-" in label:
        return label.replace("-", "-\n", 1)
    if len(label) > 18 and " " in label:
        left, right = label.split(" ", 1)
        return f"{left}\n{right}"
    return label


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _slide_key(value: str) -> str:
    name = Path(str(value).strip()).name.casefold()
    known = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".qptiff", ".geojson", ".json"}
    while Path(name).suffix.casefold() in known:
        name = Path(name).stem
    return re.sub(r"[^a-z0-9]+", "", name)


def _sample_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= STD_DDOF:
        return float("nan")
    return float(np.std(values, ddof=STD_DDOF))


def _mean_sd(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(array)), _sample_sd(array), int(len(array))


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "pad_inches": 0}
    for extension in ("pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{extension}"), **common)
    figure.savefig(stem.with_suffix(".png"), dpi=300, **common)
    try:
        figure.savefig(
            stem.with_suffix(".tiff"), dpi=600,
            pil_kwargs={"compression": "tiff_lzw"}, **common,
        )
    except (TypeError, OSError):
        figure.savefig(stem.with_suffix(".tiff"), dpi=600, **common)


# =============================================================================
# Method and input discovery
# =============================================================================


def _prediction_dir_from_candidate(candidate: Path) -> Path | None:
    if not candidate.is_dir():
        return None
    if candidate.name == "predictions":
        return candidate
    nested = candidate / "predictions"
    return nested if nested.is_dir() else None


def _resolve_method(raw_path: str, formal_name: str, rank: int) -> ResolvedMethod | None:
    supplied = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.extend((PROJECT_ROOT / supplied, DEFAULT_RUN_DIR / supplied.name))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        predictions = _prediction_dir_from_candidate(candidate)
        if predictions is not None:
            method_dir = predictions.parent
            return ResolvedMethod(raw_path, method_dir.name, formal_name, predictions, rank)

    # Last fallback: exact basename search under outputs. This is intentionally
    # exact and deterministic; it never chooses a partial-name match.
    outputs_root = PROJECT_ROOT / "outputs"
    if outputs_root.is_dir():
        matches = sorted(
            (path for path in outputs_root.rglob(supplied.name) if path.is_dir()),
            key=lambda path: str(path).casefold(),
        )
        valid = [path for path in matches if _prediction_dir_from_candidate(path) is not None]
        if len(valid) == 1:
            predictions = _prediction_dir_from_candidate(valid[0])
            assert predictions is not None
            return ResolvedMethod(raw_path, valid[0].name, formal_name, predictions, rank)
        if len(valid) > 1:
            _print(f"WARNING: ambiguous method path for {formal_name}; found {len(valid)} exact basename matches. Skipping.")
            for match in valid:
                _print(f"  - {match}")
            return None

    _print(f"WARNING: prediction directory not found for {formal_name}: {raw_path}")
    return None


def _resolve_methods() -> list[ResolvedMethod]:
    """Resolve every configured method, failing loudly on any missing directory.

    Silent omission is unsafe for an official benchmark: a typo in one method
    path would otherwise reduce the denominator while still producing figures.
    """
    methods: list[ResolvedMethod] = []
    unresolved: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for rank, (raw_path, formal_name) in enumerate(OFFICIAL_METHOD_SPECS):
        resolved = _resolve_method(raw_path, formal_name, rank)
        if resolved is None:
            unresolved.append((formal_name, raw_path))
            continue
        if resolved.method_name in seen_names:
            raise RuntimeError(f"Duplicate resolved method directory: {resolved.method_name}")
        seen_names.add(resolved.method_name)
        methods.append(resolved)

    if unresolved:
        details = "\n".join(
            f"  - {formal_name}: {raw_path}"
            for formal_name, raw_path in unresolved
        )
        raise RuntimeError(
            "One or more configured methods could not be resolved. "
            "No evaluation or figures were generated.\n" + details
        )
    return methods


# =============================================================================
# GeoJSON geometry and metrics
# =============================================================================


def _polygonal_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    parts: list[Polygon] = []
    for member in getattr(geometry, "geoms", []):
        parts.extend(_polygonal_parts(member))
    return parts


def _repair_geometry(geometry: Any) -> Any:
    if geometry is None or geometry.is_empty or geometry.is_valid:
        return geometry
    try:
        repaired = _shapely_make_valid(geometry) if _shapely_make_valid else geometry.buffer(0)
    except Exception:
        repaired = geometry.buffer(0)
    return repaired


def _load_union(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("GeoJSON root is not an object")

    if payload.get("type") == "FeatureCollection":
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError("FeatureCollection.features is not a list")
        raw_geometries = [feature.get("geometry") for feature in features if isinstance(feature, dict)]
    elif payload.get("type") == "Feature":
        raw_geometries = [payload.get("geometry")]
    else:
        raw_geometries = [payload]

    polygons: list[Polygon] = []
    for raw in raw_geometries:
        if not isinstance(raw, dict):
            continue
        try:
            geometry = _repair_geometry(shape(raw))
        except Exception:
            continue
        polygons.extend(_polygonal_parts(geometry))

    if not polygons:
        return GeometryCollection()
    merged = _repair_geometry(unary_union(polygons))
    final_parts = _polygonal_parts(merged)
    if not final_parts:
        return GeometryCollection()
    return unary_union(final_parts)


def _compute_metrics(prediction: Any, ground_truth: Any) -> dict[str, float]:
    pred_empty = prediction is None or prediction.is_empty
    gt_empty = ground_truth is None or ground_truth.is_empty
    if pred_empty and gt_empty:
        return {"dice_score": 1.0, "iou": 1.0, "oversegmentation": 0.0, "undersegmentation": 0.0}
    if gt_empty:
        return {"dice_score": 0.0, "iou": 0.0, "oversegmentation": float("nan"), "undersegmentation": 0.0}
    if pred_empty:
        return {"dice_score": 0.0, "iou": 0.0, "oversegmentation": 0.0, "undersegmentation": 1.0}

    intersection = prediction.intersection(ground_truth).area
    pred_area = prediction.area
    gt_area = ground_truth.area
    union_area = pred_area + gt_area - intersection
    fp_area = max(0.0, pred_area - intersection)
    fn_area = max(0.0, gt_area - intersection)
    return {
        "dice_score": float(2.0 * intersection / (pred_area + gt_area)) if pred_area + gt_area > 0 else 1.0,
        "iou": float(intersection / union_area) if union_area > 0 else 1.0,
        "oversegmentation": float(fp_area / gt_area) if gt_area > 0 else float("nan"),
        "undersegmentation": float(fn_area / gt_area) if gt_area > 0 else float("nan"),
    }


# =============================================================================
# SQLite checkpoint and concise CSV snapshot
# =============================================================================

_DB_SCHEMA_VERSION = 1


def _connect_database() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(METRICS_DB, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            filename TEXT NOT NULL,
            method_name TEXT NOT NULL,
            formal_name TEXT NOT NULL,
            dice_score REAL,
            iou REAL,
            oversegmentation REAL,
            undersegmentation REAL,
            ground_truth_size INTEGER NOT NULL,
            ground_truth_mtime_ns INTEGER NOT NULL,
            prediction_size INTEGER NOT NULL,
            prediction_mtime_ns INTEGER NOT NULL,
            computed_at_utc TEXT NOT NULL,
            PRIMARY KEY (filename, method_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
        (str(_DB_SCHEMA_VERSION),),
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_method ON metrics(method_name)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_filename ON metrics(filename)"
    )
    connection.commit()
    return connection


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _load_resume_index(
    connection: sqlite3.Connection,
    methods: Sequence[ResolvedMethod],
) -> dict[tuple[str, str], sqlite3.Row]:
    if not methods:
        return {}
    placeholders = ",".join("?" for _ in methods)
    rows = connection.execute(
        f"""
        SELECT filename, method_name, formal_name,
               ground_truth_size, ground_truth_mtime_ns,
               prediction_size, prediction_mtime_ns
        FROM metrics
        WHERE method_name IN ({placeholders})
        """,
        tuple(method.method_name for method in methods),
    )
    return {(row["filename"], row["method_name"]): row for row in rows}


def _row_matches_inputs(
    row: sqlite3.Row,
    *,
    ground_truth_signature: tuple[int, int],
    prediction_signature: tuple[int, int],
) -> bool:
    return (
        int(row["ground_truth_size"]) == ground_truth_signature[0]
        and int(row["ground_truth_mtime_ns"]) == ground_truth_signature[1]
        and int(row["prediction_size"]) == prediction_signature[0]
        and int(row["prediction_mtime_ns"]) == prediction_signature[1]
    )


def _upsert_metric(
    connection: sqlite3.Connection,
    *,
    filename: str,
    method: ResolvedMethod,
    metrics: dict[str, float],
    ground_truth_signature: tuple[int, int],
    prediction_signature: tuple[int, int],
) -> None:
    connection.execute(
        """
        INSERT INTO metrics (
            filename, method_name, formal_name,
            dice_score, iou, oversegmentation, undersegmentation,
            ground_truth_size, ground_truth_mtime_ns,
            prediction_size, prediction_mtime_ns,
            computed_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filename, method_name) DO UPDATE SET
            formal_name = excluded.formal_name,
            dice_score = excluded.dice_score,
            iou = excluded.iou,
            oversegmentation = excluded.oversegmentation,
            undersegmentation = excluded.undersegmentation,
            ground_truth_size = excluded.ground_truth_size,
            ground_truth_mtime_ns = excluded.ground_truth_mtime_ns,
            prediction_size = excluded.prediction_size,
            prediction_mtime_ns = excluded.prediction_mtime_ns,
            computed_at_utc = excluded.computed_at_utc
        """,
        (
            filename,
            method.method_name,
            method.formal_name,
            _finite_or_none(metrics["dice_score"]),
            _finite_or_none(metrics["iou"]),
            _finite_or_none(metrics["oversegmentation"]),
            _finite_or_none(metrics["undersegmentation"]),
            ground_truth_signature[0],
            ground_truth_signature[1],
            prediction_signature[0],
            prediction_signature[1],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    # Each completed WSI is a durable checkpoint. WAL avoids rewriting all
    # previous rows, unlike the former CSV checkpoint.
    connection.commit()


def _delete_metric(
    connection: sqlite3.Connection,
    *,
    filename: str,
    method_name: str,
) -> None:
    connection.execute(
        "DELETE FROM metrics WHERE filename = ? AND method_name = ?",
        (filename, method_name),
    )
    connection.commit()


def _current_database_rows(
    connection: sqlite3.Connection,
    methods: Sequence[ResolvedMethod],
    ground_truth_names: set[str],
) -> list[dict[str, str]]:
    if not methods:
        return []
    placeholders = ",".join("?" for _ in methods)
    db_rows = connection.execute(
        f"""
        SELECT filename, method_name, formal_name,
               dice_score, iou, oversegmentation, undersegmentation
        FROM metrics
        WHERE method_name IN ({placeholders})
        """,
        tuple(method.method_name for method in methods),
    )
    rank = {method.method_name: method.official_rank for method in methods}
    formal = {method.method_name: method.formal_name for method in methods}
    rows: list[dict[str, str]] = []
    for row in db_rows:
        if row["filename"] not in ground_truth_names:
            continue
        rows.append(
            {
                "filename": str(row["filename"]),
                "method_name": str(row["method_name"]),
                "formal_name": formal.get(str(row["method_name"]), str(row["formal_name"])),
                "dice_score": "nan" if row["dice_score"] is None else _format_float(float(row["dice_score"])),
                "iou": "nan" if row["iou"] is None else _format_float(float(row["iou"])),
                "oversegmentation": "nan" if row["oversegmentation"] is None else _format_float(float(row["oversegmentation"])),
                "undersegmentation": "nan" if row["undersegmentation"] is None else _format_float(float(row["undersegmentation"])),
            }
        )
    rows.sort(
        key=lambda row: (
            rank.get(row["method_name"], 10_000),
            row["filename"].casefold(),
            row["filename"],
        )
    )
    return rows


def _write_metrics_snapshot(rows: Sequence[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp = METRICS_CSV.with_suffix(METRICS_CSV.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, METRICS_CSV)


def _ground_truth_paths() -> list[Path]:
    if not GROUND_TRUTH_DIR.is_dir():
        raise NotADirectoryError(f"Ground-truth directory not found: {GROUND_TRUTH_DIR}")
    paths = sorted(GROUND_TRUTH_DIR.glob("*.geojson"), key=lambda path: path.name.casefold())
    if not paths:
        raise FileNotFoundError(f"No GeoJSON ground truth files found in: {GROUND_TRUTH_DIR}")
    return paths


def evaluate_all(methods: Sequence[ResolvedMethod]) -> None:
    ground_truth_paths = _ground_truth_paths()
    if not methods:
        raise RuntimeError("None of the configured prediction directories could be resolved.")

    connection = _connect_database()
    try:
        # Keep display names current without recomputing unchanged geometry.
        for method in methods:
            connection.execute(
                "UPDATE metrics SET formal_name = ? WHERE method_name = ?",
                (method.formal_name, method.method_name),
            )
        connection.commit()

        resume_index = _load_resume_index(connection, methods)
        total = len(methods) * len(ground_truth_paths)
        completed = 0
        computed = skipped = missing = failed = invalidated = 0
        gt_cache: dict[Path, Any] = {}
        gt_signatures = {path: _file_signature(path) for path in ground_truth_paths}

        _print(f"Ground truth: {GROUND_TRUTH_DIR}")
        _print(f"Database:     {METRICS_DB}")
        _print(f"Methods:      {len(methods)}")
        _print(f"WSI labels:   {len(ground_truth_paths)}")
        _print(f"DB rows:      {connection.execute('SELECT COUNT(*) FROM metrics').fetchone()[0]}")
        _print()

        ground_truth_name_set = {path.name for path in ground_truth_paths}
        for method_index, method in enumerate(methods, start=1):
            _print(f"[{method_index}/{len(methods)}] {method.formal_name}")
            _print(f"  {method.predictions_dir}")
            prediction_names = {path.name for path in method.predictions_dir.glob("*.geojson")}
            extra = prediction_names - ground_truth_name_set
            if extra:
                _print(f"  NOTE: {len(extra)} prediction file(s) have no matching ground truth and are ignored.")

            for gt_path in ground_truth_paths:
                completed += 1
                key = (gt_path.name, method.method_name)
                prefix = f"  [{completed:>5}/{total}] {gt_path.name}"
                pred_path = method.predictions_dir / gt_path.name

                if not pred_path.is_file():
                    missing += 1
                    if key in resume_index:
                        _delete_metric(
                            connection,
                            filename=gt_path.name,
                            method_name=method.method_name,
                        )
                        resume_index.pop(key, None)
                        invalidated += 1
                    _print(f"{prefix}  MISSING")
                    continue

                gt_signature = gt_signatures[gt_path]
                pred_signature = _file_signature(pred_path)
                existing = resume_index.get(key)
                if existing is not None and _row_matches_inputs(
                    existing,
                    ground_truth_signature=gt_signature,
                    prediction_signature=pred_signature,
                ):
                    skipped += 1
                    _print(f"{prefix}  SKIP")
                    continue

                if existing is not None:
                    _delete_metric(
                        connection,
                        filename=gt_path.name,
                        method_name=method.method_name,
                    )
                    resume_index.pop(key, None)
                    invalidated += 1

                started = time.perf_counter()
                try:
                    ground_truth = gt_cache.get(gt_path)
                    if ground_truth is None:
                        ground_truth = _load_union(gt_path)
                        gt_cache[gt_path] = ground_truth
                    prediction = _load_union(pred_path)
                    metrics = _compute_metrics(prediction, ground_truth)
                    _upsert_metric(
                        connection,
                        filename=gt_path.name,
                        method=method,
                        metrics=metrics,
                        ground_truth_signature=gt_signature,
                        prediction_signature=pred_signature,
                    )
                    # Store a minimal in-memory resume entry for any duplicate
                    # configured pair encountered during this process.
                    resume_index[key] = connection.execute(
                        """
                        SELECT filename, method_name, formal_name,
                               ground_truth_size, ground_truth_mtime_ns,
                               prediction_size, prediction_mtime_ns
                        FROM metrics
                        WHERE filename = ? AND method_name = ?
                        """,
                        key,
                    ).fetchone()
                    computed += 1
                    elapsed = time.perf_counter() - started
                    _print(
                        f"{prefix}  OK  Dice={metrics['dice_score']:.4f}  "
                        f"IoU={metrics['iou']:.4f}  {elapsed:.2f}s"
                    )
                except Exception as exc:
                    failed += 1
                    _print(f"{prefix}  ERROR  {type(exc).__name__}: {exc}")

        current_rows = len(
            _current_database_rows(
                connection,
                methods,
                {path.name for path in ground_truth_paths},
            )
        )
        total_database_rows = int(
            connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        )
        _print()
        _print(
            f"Evaluation complete: computed={computed}, skipped={skipped}, "
            f"missing={missing}, invalidated={invalidated}, failed={failed}, "
            f"current_config_rows={current_rows}, database_rows_total={total_database_rows}"
        )
    finally:
        connection.close()


def load_current_rows(
    methods: Sequence[ResolvedMethod],
) -> list[dict[str, str]]:
    """Return the current configured database rows without writing a CSV."""
    ground_truth_paths = _ground_truth_paths()
    connection = _connect_database()
    try:
        return _current_database_rows(
            connection,
            methods,
            {path.name for path in ground_truth_paths},
        )
    finally:
        connection.close()


def export_metrics_snapshot(methods: Sequence[ResolvedMethod]) -> Path:
    """Export the current configured SQLite view to the concise CSV snapshot."""
    rows = load_current_rows(methods)
    _write_metrics_snapshot(rows)
    _print(f"CSV snapshot refreshed: {METRICS_CSV} ({len(rows)} rows)")
    return METRICS_CSV


def print_method_coverage(methods: Sequence[ResolvedMethod]) -> None:
    """Print current prediction and SQLite coverage for every configured method."""
    ground_truth_paths = _ground_truth_paths()
    ground_truth_names = {path.name for path in ground_truth_paths}
    connection = _connect_database()
    try:
        counts = {
            str(row["method_name"]): int(row["n"])
            for row in connection.execute(
                "SELECT method_name, COUNT(*) AS n FROM metrics GROUP BY method_name"
            )
        }
    finally:
        connection.close()

    _print("\nPer-method coverage")
    _print("-------------------")
    for method in methods:
        prediction_names = {
            path.name for path in method.predictions_dir.glob("*.geojson")
        }
        matched_predictions = len(prediction_names & ground_truth_names)
        missing_predictions = len(ground_truth_names - prediction_names)
        stored_rows = counts.get(method.method_name, 0)
        status = "COMPLETE" if missing_predictions == 0 and stored_rows >= len(ground_truth_names) else "PARTIAL"
        _print(
            f"{status:8}  {method.formal_name}: "
            f"predictions={matched_predictions}/{len(ground_truth_names)}, "
            f"database_rows={stored_rows}, missing={missing_predictions}"
        )


# =============================================================================
# Records, ordering, and tables
# =============================================================================


def _numeric_rows(rows: Sequence[dict[str, str]], methods: Sequence[ResolvedMethod]) -> list[dict[str, Any]]:
    allowed = {method.method_name for method in methods}
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method_name") not in allowed:
            continue
        parsed.append(
            {
                "filename": row["filename"],
                "method_name": row["method_name"],
                "formal_name": row["formal_name"],
                **{metric: _safe_float(row.get(metric)) for metric in NUMERIC_METRICS},
            }
        )
    return parsed


def _group_by_method(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method_name"]].append(record)
    return dict(grouped)


def _method_order_by_dice(records: Sequence[dict[str, Any]], methods: Sequence[ResolvedMethod]) -> list[str]:
    grouped = _group_by_method(records)
    official_rank = {method.method_name: method.official_rank for method in methods}
    formal = {method.method_name: method.formal_name for method in methods}

    def key(method_name: str) -> tuple[float, int, str]:
        mean, _, n = _mean_sd(record["dice_score"] for record in grouped[method_name])
        return (mean if n else -math.inf, -official_rank.get(method_name, 10_000), formal.get(method_name, method_name))

    return sorted(grouped, key=key, reverse=True)


def _summary_row(method_name: str, method_records: Sequence[dict[str, Any]], formal_name: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "method_name": method_name,
        "formal_name": formal_name,
        "n_wsi": len(method_records),
    }
    for metric in NUMERIC_METRICS:
        mean, sd, _ = _mean_sd(record[metric] for record in method_records)
        prefix = metric
        output[f"{prefix}_mean"] = mean
        output[f"{prefix}_std"] = sd
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (_format_float(value) if isinstance(value, float) else value)
                    for field, value in row.items()
                }
            )


def _format_mean_sd(mean: Any, sd: Any) -> str:
    m = _safe_float(mean)
    s = _safe_float(sd)
    if not math.isfinite(m):
        return "--"
    return f"{m:.3f} $\\pm$ {s:.3f}" if math.isfinite(s) else f"{m:.3f}"


def write_summary_table(records: Sequence[dict[str, Any]], methods: Sequence[ResolvedMethod], tables_dir: Path) -> None:
    grouped = _group_by_method(records)
    method_order = _method_order_by_dice(records, methods)
    formal = {method.method_name: method.formal_name for method in methods}
    rows = [_summary_row(name, grouped[name], formal.get(name, name)) for name in method_order]
    fields = [
        "method_name", "formal_name", "n_wsi",
        "dice_score_mean", "dice_score_std",
        "iou_mean", "iou_std",
        "oversegmentation_mean", "oversegmentation_std",
        "undersegmentation_mean", "undersegmentation_std",
    ]
    _write_csv(tables_dir / "table_1_method_summary.csv", rows, fields)

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Whole-slide tissue-segmentation performance. Values are mean $\pm$ standard deviation across evaluated WSI files.}",
        r"\label{tab:official-full-performance}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & $n$ & Dice & IoU & Over-segmentation & Under-segmentation \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            f"{_latex_escape(str(row['formal_name']))} & {row['n_wsi']} & "
            f"{_format_mean_sd(row['dice_score_mean'], row['dice_score_std'])} & "
            f"{_format_mean_sd(row['iou_mean'], row['iou_std'])} & "
            f"{_format_mean_sd(row['oversegmentation_mean'], row['oversegmentation_std'])} & "
            f"{_format_mean_sd(row['undersegmentation_mean'], row['undersegmentation_std'])} \\\\" 
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (tables_dir / "table_1_method_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


# =============================================================================
# Core Figures 1–3
# =============================================================================


def _metric_values(records: Sequence[dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray([record.get(metric, float("nan")) for record in records], dtype=float)
    return values[np.isfinite(values)]


def figure_1(records: Sequence[dict[str, Any]], methods: Sequence[ResolvedMethod], figures_dir: Path) -> None:
    grouped = _group_by_method(records)
    order = _method_order_by_dice(records, methods)
    order = [name for name in order if len(_metric_values(grouped[name], "dice_score"))]
    if not order:
        return
    height_mm = max(68, 7.0 * len(order) + 26)
    figure = plt.figure(figsize=(COL_W_2, height_mm * MM))
    axis = figure.add_axes([0.34, 0.075, 0.645, 0.905])
    y = np.arange(len(order))[::-1]
    formal = {method.method_name: method.formal_name for method in methods}
    means, sds = [], []
    for name in order:
        mean, sd, _ = _mean_sd(_metric_values(grouped[name], "dice_score"))
        means.append(mean)
        sds.append(sd)
    means_array = np.asarray(means)
    sds_array = np.asarray(sds)
    low = np.maximum(0.0, means_array - np.nan_to_num(sds_array))
    high = np.minimum(1.0, means_array + np.nan_to_num(sds_array))
    for yi, mean, lo, hi in zip(y, means_array, low, high):
        axis.plot([lo, hi], [yi, yi], color=ACC, lw=1.1, solid_capstyle="round")
        axis.scatter([mean], [yi], s=23, color=ACC_D, linewidths=0, zorder=3)
    axis.set_xlim(0.0, 1.0)
    axis.set_yticks(y)
    axis.set_yticklabels(
        [f"{_compact_label(formal.get(name, name))} (n={len(grouped[name])})" for name in order],
        fontsize=5.1,
    )
    axis.set_xlabel("Mean Dice ± SD", fontsize=6.2)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    _save_figure(figure, figures_dir / "figure_1_all_methods_dice_ranking")
    plt.close(figure)


def _draw_boxplot(axis: plt.Axes, data: Sequence[np.ndarray], labels: Sequence[str], ylabel: str, *, ylim: tuple[float, float] | None = None) -> None:
    positions = np.arange(1, len(data) + 1)
    box = axis.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True, showfliers=False,
        medianprops={"color": INK, "linewidth": 1.1},
        whiskerprops={"color": INK, "linewidth": 0.6},
        capprops={"color": INK, "linewidth": 0.6},
        boxprops={"edgecolor": INK, "linewidth": 0.6},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(ACC_L)
    for position, values in zip(positions, data):
        rng = np.random.default_rng(int(position))
        axis.scatter(
            np.full(len(values), position) + rng.uniform(-0.10, 0.10, size=len(values)),
            values, s=5, color=ACC, alpha=0.15, linewidths=0,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=5.2)
    axis.set_ylabel(ylabel, fontsize=6.1)
    axis.tick_params(axis="x", length=0)
    axis.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        axis.set_ylim(*ylim)


def figure_2_and_3(records: Sequence[dict[str, Any]], methods: Sequence[ResolvedMethod], figures_dir: Path) -> None:
    grouped = _group_by_method(records)
    order = _method_order_by_dice(records, methods)
    selected = [name for name in order if len(_metric_values(grouped[name], "dice_score"))][:TOP_K_DISTRIBUTIONS]
    if not selected:
        return
    formal = {method.method_name: method.formal_name for method in methods}
    labels = [f"{_compact_label(formal.get(name, name))}\n(n={len(grouped[name])})" for name in selected]

    figure, axes = plt.subplots(1, 2, figsize=(COL_W_2, 58 * MM), squeeze=False)
    for axis, metric, label in zip(axes[0], ("dice_score", "iou"), ("Dice", "IoU (Jaccard)")):
        _draw_boxplot(axis, [_metric_values(grouped[name], metric) for name in selected], labels, label, ylim=(0.0, 1.0))
    figure.tight_layout(pad=0.15)
    _save_figure(figure, figures_dir / "figure_2_top_k_overlap_boxplots")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(COL_W_2, 58 * MM), squeeze=False)
    for axis, metric, label in zip(
        axes[0],
        ("oversegmentation", "undersegmentation"),
        ("Over-segmentation rate", "Under-segmentation rate"),
    ):
        values = [_metric_values(grouped[name], metric) for name in selected]
        _draw_boxplot(axis, values, labels, label)
        concatenated = np.concatenate(values) if values else np.array([])
        if len(concatenated):
            upper = max(0.05, float(np.percentile(concatenated, 99) * 1.10))
            clipped = int(np.sum(concatenated > upper))
            axis.set_ylim(0.0, upper)
            if clipped:
                axis.text(0.98, 0.98, f"{clipped} outlier(s) above axis range", transform=axis.transAxes, ha="right", va="top", fontsize=5.0, color=MID)
    figure.tight_layout(pad=0.15)
    _save_figure(figure, figures_dir / "figure_3_top_k_failure_mode_boxplots")
    plt.close(figure)


# =============================================================================
# Optional metadata tables and Figures 4–7
# =============================================================================


def _find_annotation_csv() -> Path | None:
    return next((path for path in ANNOTATION_CSV_CANDIDATES if path.is_file()), None)


def _canonical_source(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text or "Unknown"


def _canonical_organ(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return "Unknown"
    normal = text.casefold()
    corrections = {"lymph nodes": "lymph node", "mesentry": "mesentery"}
    return corrections.get(normal, normal)


def _canonical_staining(value: Any) -> str:
    text = " ".join(str(value or "").replace("_", " ").strip().split())
    return text.upper() if text else "Unknown"


def _display_label(value: str) -> str:
    if value == "Unknown":
        return value
    if value.isupper() and len(value) <= 12:
        return value
    return value.replace("_", " ").title()


def _read_metadata(path: Path) -> dict[str, Metadata]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        normalised = {re.sub(r"[^a-z0-9]", "", column.casefold()): column for column in header}

        def column(*names: str) -> str:
            for name in names:
                found = normalised.get(re.sub(r"[^a-z0-9]", "", name.casefold()))
                if found:
                    return found
            raise ValueError(f"annotation CSV is missing one of: {', '.join(names)}")

        slide_col = column("slide_id", "filename", "file_name", "slide", "wsi")
        source_col = column("data_source", "source", "dataset")
        organ_col = column("organ", "tissue", "site")
        staining_col = column("staining", "stain")
        lookup: dict[str, Metadata] = {}
        for row in reader:
            key = _slide_key(row.get(slide_col, ""))
            if not key:
                continue
            lookup[key] = Metadata(
                _canonical_source(row.get(source_col)),
                _canonical_organ(row.get(organ_col)),
                _canonical_staining(row.get(staining_col)),
            )
    return lookup


def _join_metadata(records: Sequence[dict[str, Any]], lookup: dict[str, Metadata]) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for record in records:
        metadata = lookup.get(_slide_key(record["filename"]))
        if metadata is None:
            continue
        joined.append({**record, "data_source": metadata.data_source, "organ": metadata.organ, "staining": metadata.staining})
    return joined


def _stratified_rows(joined: Sequence[dict[str, Any]], field: str, methods: Sequence[ResolvedMethod]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in joined:
        grouped[(record["method_name"], record[field])].append(record)
    method_order = _method_order_by_dice(joined, methods)
    strata = sorted({record[field] for record in joined}, key=lambda value: (value == "Unknown", str(value).casefold()))
    formal = {method.method_name: method.formal_name for method in methods}
    rows: list[dict[str, Any]] = []
    for method_name in method_order:
        for stratum in strata:
            subset = grouped.get((method_name, stratum), [])
            if not subset:
                continue
            row: dict[str, Any] = {
                "method_name": method_name,
                "formal_name": formal.get(method_name, method_name),
                field: stratum,
                "n_wsi": len({record["filename"] for record in subset}),
            }
            for metric in NUMERIC_METRICS:
                mean, sd, _ = _mean_sd(record[metric] for record in subset)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = sd
            rows.append(row)
    return rows


def _write_stratified_table(joined: Sequence[dict[str, Any]], field: str, stem: str, methods: Sequence[ResolvedMethod], tables_dir: Path) -> None:
    rows = _stratified_rows(joined, field, methods)
    fields = [
        "method_name", "formal_name", field, "n_wsi",
        "dice_score_mean", "dice_score_std", "iou_mean", "iou_std",
        "oversegmentation_mean", "oversegmentation_std",
        "undersegmentation_mean", "undersegmentation_std",
    ]
    _write_csv(tables_dir / f"{stem}.csv", rows, fields)
    tex = [
        r"\begin{landscape}",
        r"\begin{longtable}{llrcccc}",
        rf"\caption{{Whole-slide tissue-segmentation performance by {_latex_escape(field.replace('_', ' '))}. Values are mean $\pm$ standard deviation.}}\\",
        rf"\label{{tab:{_safe_suffix(stem)}}}\\",
        r"\toprule",
        f"Method & {_latex_escape(field.replace('_', ' ').title())} & $n$ & Dice & IoU & Over-segmentation & Under-segmentation " + r"\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        f"Method & {_latex_escape(field.replace('_', ' ').title())} & $n$ & Dice & IoU & Over-segmentation & Under-segmentation " + r"\\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        tex.append(
            f"{_latex_escape(str(row['formal_name']))} & {_latex_escape(_display_label(str(row[field])))} & {row['n_wsi']} & "
            f"{_format_mean_sd(row['dice_score_mean'], row['dice_score_std'])} & "
            f"{_format_mean_sd(row['iou_mean'], row['iou_std'])} & "
            f"{_format_mean_sd(row['oversegmentation_mean'], row['oversegmentation_std'])} & "
            f"{_format_mean_sd(row['undersegmentation_mean'], row['undersegmentation_std'])} \\\\" 
        )
    tex += [r"\bottomrule", r"\end{longtable}", r"\end{landscape}"]
    (tables_dir / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def _method_family_label(formal_name: str) -> str:
    """Return the conceptual benchmark method family for a configured variant."""
    label = str(formal_name).strip()
    for family in METHOD_FAMILY_ORDER:
        if label == family or label.startswith(f"{family} ("):
            return family
    # FESI's improved implementation is still a variant of the FESI family.
    if label.startswith("FESI improved"):
        return "FESI"
    return label.split(" (", 1)[0].strip()


def _format_mpp(value: float) -> str:
    """Format microns-per-pixel compactly without unnecessary trailing zeros."""
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _configured_mpp(method: ResolvedMethod) -> str | None:
    """Return the selected run's configured MPP when it can be recovered.

    Explicit ``mpp=`` values take precedence. TRIDENT runs store target
    magnification rather than MPP; using the standard 40x = 0.25 µm/px scale,
    their equivalent resolution is 10 / target_mag µm/px.
    """
    searchable = f"{method.formal_name} {method.raw_path} {method.method_name}"
    explicit = re.search(r"(?:^|[^a-z0-9])mpp\s*=\s*([0-9]+(?:\.[0-9]+)?)", searchable, re.IGNORECASE)
    if explicit:
        return _format_mpp(float(explicit.group(1)))

    target_mag = re.search(r"target-mag\s*=\s*([0-9]+(?:\.[0-9]+)?)", searchable, re.IGNORECASE)
    if target_mag:
        magnification = float(target_mag.group(1))
        if magnification > 0:
            return _format_mpp(10.0 / magnification)
    return None


def _best_method_family_display_label(method: ResolvedMethod) -> str:
    """Return the row label for a selected conceptual-family winner.

    For the method families where resolution is part of the comparison, show
    only the conceptual family name and the selected ``mpp`` value.  Variant
    qualifiers such as ``simple``, ``saliency``, or ``improved`` are omitted.
    All other families are displayed using the plain conceptual family name.
    """
    family = _method_family_label(method.formal_name)
    if family in MPP_LABEL_METHOD_FAMILIES:
        mpp = _configured_mpp(method)
        return f"{family} (mpp={mpp})" if mpp is not None else family
    return family


def _ranked_methods(
    joined: Sequence[dict[str, Any]],
    methods: Sequence[ResolvedMethod],
    *,
    top_k: int,
    best_configuration: bool = False,
    best_method_family: bool = False,
    best_bigpicture_only: bool = False,
) -> list[str]:
    grouped = _group_by_method(joined)
    order = _method_order_by_dice(joined, methods)
    formal = {method.method_name: method.formal_name for method in methods}
    if best_configuration:
        # Preserve the original directory-prefix grouping used by Figure 7 v2.
        winners: dict[str, str] = {}
        for name in order:
            family = name.partition("__")[0]
            if family not in winners:
                winners[family] = name
        order = [name for name in order if winners.get(name.partition("__")[0]) == name]
    if best_method_family:
        # `order` is already descending by global mean Dice, so the first
        # configured variant encountered for each conceptual family is its winner.
        winners = {}
        for name in order:
            family = _method_family_label(formal.get(name, name))
            if family not in winners:
                winners[family] = name
        order = [
            name for name in order
            if winners.get(_method_family_label(formal.get(name, name))) == name
        ]
    if best_bigpicture_only:
        bigpictures = [name for name in order if name.startswith("bigpicture__")]
        best = bigpictures[0] if bigpictures else None
        order = [name for name in order if not name.startswith("bigpicture__") or name == best]
    return [name for name in order if name in grouped][:top_k]


def _strata_for_figure(joined: Sequence[dict[str, Any]], field: str, max_columns: int, *, min_n: int = 1, pool_other: bool = False) -> tuple[list[str], dict[str, str], Counter[str]]:
    slide_sets: dict[str, set[str]] = defaultdict(set)
    for record in joined:
        if record[field] != "Unknown":
            slide_sets[str(record[field])].add(record["filename"])
    counts = Counter({key: len(value) for key, value in slide_sets.items()})
    ordered = sorted(counts, key=lambda key: (-counts[key], key.casefold()))
    mapping = {stratum: stratum for stratum in ordered}

    if pool_other:
        kept = [stratum for stratum in ordered if counts[stratum] >= min_n]
        pooled = [stratum for stratum in ordered if counts[stratum] < min_n]
        if max_columns > 0 and len(kept) > max_columns:
            pooled.extend(kept[max_columns:])
            kept = kept[:max_columns]
        if pooled:
            other = "Other stains" if field == "staining" else "Other"
            for stratum in pooled:
                mapping[stratum] = other
            kept.append(other)
            counts[other] = len({record["filename"] for record in joined if mapping.get(str(record[field]), str(record[field])) == other})
        return kept, mapping, counts

    kept = [stratum for stratum in ordered if counts[stratum] >= min_n]
    if max_columns > 0:
        kept = kept[:max_columns]
    return kept, mapping, counts


def _heatmap_matrix(joined: Sequence[dict[str, Any]], method_names: Sequence[str], strata: Sequence[str], field: str, mapper: dict[str, str]) -> np.ndarray:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in joined:
        mapped = mapper.get(str(record[field]), str(record[field]))
        value = record["dice_score"]
        if record["method_name"] in method_names and mapped in strata and math.isfinite(value):
            buckets[(record["method_name"], mapped)].append(value)
    matrix = np.full((len(method_names), len(strata)), np.nan)
    for row, method_name in enumerate(method_names):
        for column, stratum in enumerate(strata):
            values = buckets.get((method_name, stratum), [])
            if values:
                matrix[row, column] = float(np.mean(values))
    return matrix


def _draw_dice_heatmap(
    joined: Sequence[dict[str, Any]], methods: Sequence[ResolvedMethod], field: str, stem: str,
    figures_dir: Path, *, max_columns: int, top_k: int, min_n: int = 1,
    pool_other: bool = False, best_configuration: bool = False,
    best_method_family: bool = False, best_bigpicture_only: bool = False,
    include_failure_panel: bool = False,
) -> None:
    method_names = _ranked_methods(
        joined, methods, top_k=top_k,
        best_configuration=best_configuration,
        best_method_family=best_method_family,
        best_bigpicture_only=best_bigpicture_only,
    )
    strata, mapper, counts = _strata_for_figure(joined, field, max_columns, min_n=min_n, pool_other=pool_other)
    if not method_names or not strata:
        _print(f"Skipping {stem}: insufficient metadata-matched rows.")
        return
    matrix = _heatmap_matrix(joined, method_names, strata, field, mapper)
    if not np.isfinite(matrix).any():
        return

    formal = {method.method_name: method.formal_name for method in methods}
    method_lookup = {method.method_name: method for method in methods}
    grouped = _group_by_method(joined)
    n_rows, n_cols = matrix.shape
    height_mm = max(58.0, 7.8 * n_rows + 24.0)
    heatmap_width_mm = max(82.0, 7.2 * n_cols)
    left_mm, gap_mm, mean_width_mm, right_mm = 28.0, 1.0, 13.0, 3.0
    error_width_mm = 27.0 if include_failure_panel else 0.0
    error_gap_mm = 3.0 if include_failure_panel else 0.0
    width_mm = left_mm + heatmap_width_mm + gap_mm + mean_width_mm + error_gap_mm + error_width_mm + right_mm
    figure = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    bottom = 0.245
    height = 0.725
    axis = figure.add_axes([left_mm / width_mm, bottom, heatmap_width_mm / width_mm, height])
    mean_axis = figure.add_axes([(left_mm + heatmap_width_mm + gap_mm) / width_mm, bottom, mean_width_mm / width_mm, height])
    error_axis = None
    if include_failure_panel:
        error_axis = figure.add_axes([
            (left_mm + heatmap_width_mm + gap_mm + mean_width_mm + error_gap_mm) / width_mm,
            bottom, error_width_mm / width_mm, height,
        ])

    axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=SUPPORT_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_xticks(np.arange(n_cols))
    axis.set_xticklabels(
        [f"{_display_label(stratum)}\n(n={counts[stratum]})" for stratum in strata],
        rotation=42, ha="right", rotation_mode="anchor", fontsize=5.2,
    )
    axis.set_yticks(np.arange(n_rows))
    row_labels = []
    for name in method_names:
        label = formal.get(name, name)
        if best_method_family and name in method_lookup:
            label = _best_method_family_display_label(method_lookup[name])
        else:
            label = _compact_label(label, remove_configuration=best_configuration)
        row_labels.append(label)
    axis.set_yticklabels(
        [_compact_label(label) for label in row_labels],
        fontsize=5.2, linespacing=0.92,
    )
    axis.tick_params(axis="x", length=0, pad=2)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row in range(n_rows):
        for column in range(n_cols):
            value = matrix[row, column]
            axis.text(
                column, row, "-" if not np.isfinite(value) else f"{value:.2f}",
                ha="center", va="center", fontsize=5.0,
                color=("#64748b" if not np.isfinite(value) else ("white" if value >= 0.58 else INK)),
            )

    means, sds = [], []
    for name in method_names:
        mean, sd, _ = _mean_sd(record["dice_score"] for record in grouped[name])
        means.append(mean)
        sds.append(sd)
    means_array = np.asarray(means)
    sds_array = np.asarray(sds)
    y = np.arange(n_rows)
    # Use the exact same 0–1 value-to-colour mapping as the Dice heatmap.
    # A method with mean Dice d therefore has the same colour as a heatmap
    # cell with Dice d, rather than being reduced to one of two fixed blues.
    mean_bar_colors = [
        SUPPORT_CMAP(float(np.clip(value, 0.0, 1.0)))
        if np.isfinite(value) else SUPPORT_CMAP(np.ma.masked)
        for value in means_array
    ]
    mean_axis.barh(
        y, means_array, height=0.34, color=mean_bar_colors, edgecolor="none"
    )
    finite = np.isfinite(sds_array)
    lower_endpoints = np.full(n_rows, np.nan)
    upper_endpoints = np.full(n_rows, np.nan)
    if finite.any():
        lower = np.minimum(sds_array[finite], means_array[finite])
        upper = np.minimum(sds_array[finite], 1.0 - means_array[finite])
        lower_endpoints[finite] = means_array[finite] - lower
        upper_endpoints[finite] = means_array[finite] + upper
        mean_axis.errorbar(
            means_array[finite], y[finite],
            xerr=np.vstack([lower, upper]), fmt="none",
            ecolor="black", elinewidth=0.45,
            capsize=1.4, capthick=0.45, zorder=3,
        )

    # Place each value inside its bar initially, then measure the rendered text
    # in data coordinates.  A fixed threshold is unreliable because the mean
    # panel is deliberately narrow: the same four-character label occupies a
    # much larger fraction of the x-axis than it would in a conventional plot.
    # If the horizontal SD segment intersects the measured label box (with a
    # small visual margin), move the value above the bar.
    mean_axis.set_xlim(0.0, 1.0)
    mean_axis.set_ylim(axis.get_ylim())
    value_labels: list[tuple[int, float, Any]] = []
    for row_index, (yi, value) in enumerate(zip(y, means_array)):
        if not np.isfinite(value):
            continue
        inside_x = 0.05 if value >= 0.16 else min(0.98, value + 0.03)
        label = mean_axis.text(
            inside_x, yi, f"{value:.2f}", va="center", ha="left", fontsize=5.0,
            color="white" if value >= 0.58 else INK, zorder=4,
        )
        value_labels.append((row_index, value, label))

    # Matplotlib needs one draw before text extents are available.  The final
    # save redraws the figure after any labels have been repositioned.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverse = mean_axis.transData.inverted()
    for row_index, value, label in value_labels:
        if not (
            np.isfinite(lower_endpoints[row_index])
            and np.isfinite(upper_endpoints[row_index])
        ):
            continue
        bbox_pixels = label.get_window_extent(renderer=renderer).expanded(1.12, 1.0)
        x_left = float(inverse.transform((bbox_pixels.x0, bbox_pixels.y0))[0])
        x_right = float(inverse.transform((bbox_pixels.x1, bbox_pixels.y1))[0])
        whisker_left = float(lower_endpoints[row_index])
        whisker_right = float(upper_endpoints[row_index])
        whisker_crosses_label = max(x_left, whisker_left) <= min(x_right, whisker_right)
        if whisker_crosses_label:
            # Keep displaced values aligned to the same left edge as values
            # that remain inside their bars; only the vertical position changes.
            label.set_position((0.05, y[row_index] - 0.23))
            label.set_ha("left")
            label.set_color(INK)
            label.set_zorder(5)
            label.set_clip_on(False)
    mean_axis.set_yticks([])
    mean_axis.set_xticks([0.0, 1.0])
    mean_axis.set_xlabel("Mean Dice", fontsize=5.2, labelpad=1)
    mean_axis.spines[["top", "right", "left"]].set_visible(False)

    if error_axis is not None:
        over = np.asarray([100 * _mean_sd(record["oversegmentation"] for record in grouped[name])[0] for name in method_names])
        under = np.asarray([100 * _mean_sd(record["undersegmentation"] for record in grouped[name])[0] for name in method_names])
        finite_values = np.concatenate([over[np.isfinite(over)], under[np.isfinite(under)]])
        limit = 10.0 if not len(finite_values) else max(10.0, math.ceil(float(np.percentile(finite_values, 95)) / 5.0) * 5.0)
        error_axis.barh(y, -under, height=0.30, color="#9ca3af", edgecolor="none")
        error_axis.barh(y, over, height=0.30, color="#4b5563", edgecolor="none")
        error_axis.set_xlim(-limit, limit)
        error_axis.set_ylim(axis.get_ylim())
        error_axis.set_yticks([])
        error_axis.set_xticks([-limit, 0, limit])
        error_axis.set_xticklabels([f"{limit:.0f}%", "0", f"{limit:.0f}%"], fontsize=5.0)
        error_axis.set_xlabel("Segmentation rate (%)", fontsize=5.2, labelpad=1)
        # Keep the directional headers inside the plotting area, immediately
        # below the upper end of the shared y-axis.  Placing them above the axes
        # allowed tight-bbox export to crop the top of the glyphs.
        error_axis.text(
            0.25, 0.992, "Under", transform=error_axis.transAxes,
            ha="center", va="top", fontsize=5.0, clip_on=True,
        )
        error_axis.text(
            0.75, 0.992, "Over", transform=error_axis.transAxes,
            ha="center", va="top", fontsize=5.0, clip_on=True,
        )

        # Match the original publication figure: whole-percentage labels are
        # placed inside bars only when there is enough room; short-bar labels
        # are positioned outside with a minimum clearance from x=0.
        label_offset = max(0.030 * limit, 0.8)
        outside_min_distance_from_zero = max(0.160 * limit, 7.5)
        min_inside_percent = 18.0

        def label_fits_inside_bar(value: float, label: str) -> bool:
            required_width = max(min_inside_percent, 2.25 * len(label), 10.0)
            return value >= required_width

        def outside_under_x(value: float) -> float:
            return -max(value + label_offset, outside_min_distance_from_zero)

        def outside_over_x(value: float) -> float:
            return max(value + label_offset, outside_min_distance_from_zero)

        for row_index, value in enumerate(under):
            if not np.isfinite(value):
                continue
            yi = y[row_index]
            label = f"{value:.0f}%"
            if label_fits_inside_bar(float(value), label):
                error_axis.text(
                    -0.50 * value, yi, label,
                    ha="center", va="center", fontsize=5.0,
                    color="white", zorder=3, clip_on=True,
                )
            else:
                error_axis.text(
                    outside_under_x(float(value)), yi, label,
                    ha="right", va="center", fontsize=5.0,
                    color=INK, zorder=4, clip_on=False,
                )

        for row_index, value in enumerate(over):
            if not np.isfinite(value):
                continue
            yi = y[row_index]
            label = f"{value:.0f}%"
            if label_fits_inside_bar(float(value), label):
                error_axis.text(
                    0.50 * value, yi, label,
                    ha="center", va="center", fontsize=5.0,
                    color="white", zorder=3, clip_on=True,
                )
            else:
                error_axis.text(
                    outside_over_x(float(value)), yi, label,
                    ha="left", va="center", fontsize=5.0,
                    color=INK, zorder=4, clip_on=False,
                )

        error_axis.spines["left"].set_position(("data", 0.0))
        error_axis.spines["left"].set_linewidth(0.4)
        error_axis.spines[["top", "right"]].set_visible(False)

    _save_figure(figure, figures_dir / stem)
    plt.close(figure)




def _remove_obsolete_staining_outputs(tables_dir: Path, figures_dir: Path) -> None:
    """Remove superseded separate/per-stain outputs from earlier script versions."""
    for directory in (
        figures_dir / "figure_11_all_methods_error_rates_by_staining",
        figures_dir / "per_stain",
        tables_dir / "per_stain",
    ):
        if directory.exists():
            shutil.rmtree(directory)
            _print(f"Removed obsolete output directory: {directory}")

    # Figure 8 previously duplicated the exhaustive staining heatmap without
    # the mean-Dice and failure-rate panels. The integrated Figure 10 is now
    # the sole exhaustive all-method/all-staining figure.
    for extension in ("pdf", "svg", "png", "tiff"):
        path = figures_dir / f"figure_8_all_methods_dice_by_staining_all.{extension}"
        if path.exists():
            path.unlink()
            _print(f"Removed obsolete output file: {path}")

def metadata_outputs(records: Sequence[dict[str, Any]], methods: Sequence[ResolvedMethod], tables_dir: Path, figures_dir: Path) -> None:
    annotation_csv = _find_annotation_csv()
    if annotation_csv is None:
        _print("Metadata-stratified outputs skipped: no benchmark_data_annotation_latest.csv or benchmark_data_annotation.csv found.")
        return
    _print(f"Metadata: {annotation_csv}")
    lookup = _read_metadata(annotation_csv)
    joined = _join_metadata(records, lookup)
    unique_all = len({_slide_key(record["filename"]) for record in records})
    unique_joined = len({_slide_key(record["filename"]) for record in joined})
    _print(f"Metadata match: {unique_joined}/{unique_all} unique WSI IDs")
    if not joined:
        return

    _remove_obsolete_staining_outputs(tables_dir, figures_dir)

    _write_stratified_table(joined, "data_source", "table_s1_performance_by_source", methods, tables_dir)
    _write_stratified_table(joined, "organ", "table_s2_performance_by_organ", methods, tables_dir)
    _write_stratified_table(joined, "staining", "table_s3_performance_by_staining", methods, tables_dir)

    _draw_dice_heatmap(joined, methods, "data_source", "figure_4_top_methods_dice_by_source", figures_dir, max_columns=MAX_SOURCE_COLUMNS, top_k=TOP_K_STRATIFIED)
    _draw_dice_heatmap(joined, methods, "organ", "figure_5_top_methods_dice_by_organ", figures_dir, max_columns=MAX_ORGAN_COLUMNS, top_k=TOP_K_STRATIFIED)
    _draw_dice_heatmap(joined, methods, "staining", "figure_6_top_methods_dice_by_staining", figures_dir, max_columns=MAX_STAINING_COLUMNS, top_k=TOP_K_STRATIFIED)

    # Exhaustive organ heatmap. Staining is represented by one integrated
    # all-method figure below, rather than separate or duplicate outputs.
    _print("Generating exhaustive heatmap: all methods × all organs...")
    _draw_dice_heatmap(
        joined, methods, "organ",
        "figure_9_all_methods_dice_by_organ_all",
        figures_dir, max_columns=0, top_k=len(methods),
    )

    _print("Generating one integrated all-method/all-staining figure (Dice heatmap + mean Dice + under/over bars)...")
    _draw_dice_heatmap(
        joined, methods, "staining",
        "figure_10_all_methods_dice_and_error_rates_by_staining_all",
        figures_dir, max_columns=0, top_k=len(methods),
        include_failure_panel=True,
    )

    _print("Generating Figure 10 variant: best global-Dice configuration per conceptual method (15 rows)...")
    best_family_methods = _ranked_methods(
        joined, methods,
        top_k=len(METHOD_FAMILY_ORDER),
        best_method_family=True,
    )
    selected_families = {
        _method_family_label(next(
            method.formal_name for method in methods if method.method_name == method_name
        ))
        for method_name in best_family_methods
    }
    missing_families = [family for family in METHOD_FAMILY_ORDER if family not in selected_families]
    if missing_families:
        _print(
            "WARNING: best-variant Figure 10 has fewer than 15 rows because no "
            f"metadata-matched records were available for: {', '.join(missing_families)}"
        )
    _draw_dice_heatmap(
        joined, methods, "staining",
        "figure_10_v9_best_variant_per_method_dice_and_error_rates_by_staining_all",
        figures_dir, max_columns=0, top_k=len(METHOD_FAMILY_ORDER),
        best_method_family=True, include_failure_panel=True,
    )

    _print(
        "Generating compact Figure 10 variant: best global-Dice configuration "
        "per conceptual method, with low-support stains pooled as Other stains..."
    )
    _draw_dice_heatmap(
        joined, methods, "staining",
        "figure_10_v9_best_variant_per_method_dice_and_error_rates_by_staining_ge3_other",
        figures_dir, max_columns=MAX_STAINING_COLUMNS,
        top_k=len(METHOD_FAMILY_ORDER),
        min_n=MIN_STAINING_N_FOR_FIGURE_7, pool_other=True,
        best_method_family=True, include_failure_panel=True,
    )
    _draw_dice_heatmap(
        joined, methods, "staining", "figure_7_top_methods_dice_and_error_rates_by_staining_all", figures_dir,
        max_columns=MAX_STAINING_COLUMNS, top_k=TOP_K_STRATIFIED, include_failure_panel=True,
    )
    _draw_dice_heatmap(
        joined, methods, "staining", "figure_7_top_methods_dice_and_error_rates_by_staining_ge3_other", figures_dir,
        max_columns=MAX_STAINING_COLUMNS, top_k=TOP_K_STRATIFIED,
        min_n=MIN_STAINING_N_FOR_FIGURE_7, pool_other=True, include_failure_panel=True,
    )
    _draw_dice_heatmap(
        joined, methods, "staining", "figure_7_v2_top_methods_dice_and_error_rates_by_staining_ge3_other", figures_dir,
        max_columns=MAX_STAINING_COLUMNS, top_k=TOP_K_STRATIFIED,
        min_n=MIN_STAINING_N_FOR_FIGURE_7, pool_other=True,
        best_configuration=True, include_failure_panel=True,
    )
    _draw_dice_heatmap(
        joined, methods, "staining", "figure_7_v3_top_methods_dice_and_error_rates_by_staining_ge3_other", figures_dir,
        max_columns=MAX_STAINING_COLUMNS, top_k=TOP_K_STRATIFIED,
        min_n=MIN_STAINING_N_FOR_FIGURE_7, pool_other=True,
        best_bigpicture_only=True, include_failure_panel=True,
    )


# =============================================================================
# Main
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate official Segmenteer outputs and render publication outputs."
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="skip GeoJSON evaluation and regenerate tables and figures from SQLite",
    )
    return parser


def render_outputs_from_database(methods: Sequence[ResolvedMethod]) -> None:
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_current_rows(methods)
    records = _numeric_rows(raw_rows, methods)
    if not records:
        raise SystemExit(
            "No completed numeric rows are available in the current SQLite view; "
            "figures and tables were not generated."
        )

    _print("\nGenerating tables and figures...")
    write_summary_table(records, methods, tables_dir)
    figure_1(records, methods, figures_dir)
    figure_2_and_3(records, methods, figures_dir)
    metadata_outputs(records, methods, tables_dir, figures_dir)

    _print("\nOutputs refreshed.")
    _print(f"Database: {METRICS_DB}")
    _print(f"Tables:   {tables_dir}")
    _print(f"Figures:  {figures_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _print("OFFICIAL_FULL evaluation")
    _print("========================")
    methods = _resolve_methods()
    _print(f"Resolved {len(methods)}/{len(OFFICIAL_METHOD_SPECS)} configured methods.")
    for method in methods:
        _print(f"  {method.formal_name}: {method.predictions_dir}")
    _print()

    if not args.reports_only:
        evaluate_all(methods)
    else:
        _print("Reports-only mode: geometry evaluation skipped; rebuilding tables and figures from SQLite.")

    print_method_coverage(methods)
    render_outputs_from_database(methods)


if __name__ == "__main__":
    main()

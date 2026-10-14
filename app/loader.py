"""Output-first loader for the decoupled Segmenteer viewer.

Predictions define what the viewer can browse. Ground truth and evaluator
metrics are optional, independent inputs layered on top of the runner output.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

_WSI_EXTENSIONS: tuple[str, ...] = (
    ".tiff", ".tif", ".svs", ".ndpi", ".qptiff", ".scn", ".mrxs",
)

SUPERVISED_METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "over_segmentation_rate",
    "under_segmentation_rate",
)
UNSUPERVISED_METRICS = (
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


@dataclass(frozen=True)
class MethodInfo:
    run_id: str
    name: str
    label: str
    details: str
    backend: str | None
    model_id: str | None
    mpp: float | None
    target_mag: float | None
    params: dict[str, str]
    color: str


@dataclass(frozen=True)
class WSIRecord:
    stem: str
    image_path: str | None
    has_wsi: bool
    has_ground_truth: bool
    ground_truth_url: str | None
    run_status: str


@dataclass(frozen=True)
class IndexData:
    output_dir: str
    wsis: list[WSIRecord]
    methods: dict[str, MethodInfo]
    scores: dict[str, dict[str, dict[str, Any]]]
    metrics_available: bool
    available_metrics: list[str]
    ground_truth_available: bool
    viewer_build: str = "20260807_18"


def load_index(
    run_root: Path | str,
    *,
    metrics_file: Path | str | None = None,
    data_dir: Path | str | None = None,
    ground_truth_dir: Path | str | None = None,
) -> IndexData:
    """Build a viewer index from runner output plus optional evaluator inputs."""
    run_root = Path(run_root).expanduser().resolve()
    metrics_path = Path(metrics_file).expanduser().resolve() if metrics_file else None
    data_root = Path(data_dir).expanduser().resolve() if data_dir else None
    gt_root = Path(ground_truth_dir).expanduser().resolve() if ground_truth_dir else None

    if not run_root.is_dir():
        raise FileNotFoundError(f"Runner output directory does not exist: {run_root}")
    if metrics_path is not None and not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")
    if gt_root is not None and not gt_root.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_root}")

    method_dirs = [
        path for path in sorted(run_root.iterdir(), key=lambda p: p.name.casefold())
        if path.is_dir() and (path / "predictions").is_dir()
    ]
    run_ids = [method_dir.name for method_dir in method_dirs]
    colors = _method_colors(run_ids)
    methods: dict[str, MethodInfo] = {}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    stems: set[str] = set(_manifest_stems(run_root / "dataset_manifest.json"))

    for method_dir in method_dirs:
        run_id = method_dir.name
        metadata = _method_metadata(method_dir)
        methods[run_id] = MethodInfo(
            run_id=run_id,
            name=metadata["name"],
            label=metadata["label"],
            details=metadata["details"],
            backend=metadata["backend"],
            model_id=metadata["model_id"],
            mpp=metadata["mpp"],
            target_mag=metadata["target_mag"],
            params=metadata["params"],
            color=colors[run_id],
        )
        name = metadata["name"]
        for prediction in sorted((method_dir / "predictions").glob("*.geojson"), key=lambda p: p.name.casefold()):
            stem = prediction.stem
            stems.add(stem)
            scores.setdefault(stem, {})[run_id] = {
                "_prediction_available": True,
                "metrics": {},
                "run_id": run_id,
                "segmentation_method": name,
            }

    if metrics_path is not None:
        for row in _read_metrics(metrics_path):
            parsed = _parse_metric_row(row)
            if parsed is None:
                continue
            stem, run_id, entry = parsed
            # Evaluator artifacts may annotate existing runner slides, but they
            # must never create slides in the viewer. Runner output remains the
            # authoritative cohort.
            if stem not in stems:
                continue
            if run_id not in methods:
                resolved = _resolve_metric_method(row, methods)
                if resolved is None:
                    # Metrics cannot create a viewer method; predictions remain the
                    # authoritative runner artifact.
                    continue
                run_id = resolved
            base = scores.setdefault(stem, {}).setdefault(
                run_id,
                {
                    "_prediction_available": (run_root / run_id / "predictions" / f"{stem}.geojson").is_file(),
                    "metrics": {},
                    "run_id": run_id,
                    "segmentation_method": methods[run_id].name,
                },
            )
            base["metrics"] = entry.get("metrics", {})
            base["evaluation_mode"] = entry.get("evaluation_mode")
            base["evaluation_error"] = entry.get("evaluation_error")

    # Evaluated-mode contract: once a metrics file contributes at least one
    # usable evaluator result, only methods represented by usable metric values
    # are exposed by the viewer. This keeps prediction controls, summary tables,
    # and Dice sorting aligned with the evaluated method cohort instead of
    # showing runner methods as rows full of dashes. Prediction-only mode
    # (metrics_path is None) continues to expose every runner method.
    if metrics_path is not None:
        evaluated_run_ids = {
            run_id
            for per_slide in scores.values()
            for run_id, entry in per_slide.items()
            if run_id in methods and _has_usable_metrics(entry)
        }
        if evaluated_run_ids:
            methods = {
                run_id: method
                for run_id, method in methods.items()
                if run_id in evaluated_run_ids
            }
            for stem in list(scores):
                scores[stem] = {
                    run_id: entry
                    for run_id, entry in scores[stem].items()
                    if run_id in evaluated_run_ids
                }

    gt_stems = {
        path.stem for path in gt_root.glob("*.geojson")
    } if gt_root is not None else set()

    wsis: list[WSIRecord] = []
    for stem in sorted(stems, key=str.casefold):
        image_path = _resolve_wsi_path(data_root, stem)
        per_method = scores.get(stem, {})
        has_supervised = any(_metric_group(entry, "supervised") for entry in per_method.values())
        has_unsupervised = any(_metric_group(entry, "unsupervised") for entry in per_method.values())
        has_gt = stem in gt_stems
        status = (
            "supervised" if has_supervised
            else "unsupervised" if has_unsupervised
            else "ground_truth_available" if has_gt
            else "prediction_only"
        )
        wsis.append(
            WSIRecord(
                stem=stem,
                image_path=str(image_path) if image_path else None,
                has_wsi=image_path is not None,
                has_ground_truth=has_gt,
                ground_truth_url=f"/api/ground-truth/{stem}" if has_gt else None,
                run_status=status,
            )
        )

    ordered_metric_keys = list(SUPERVISED_METRICS) + list(UNSUPERVISED_METRICS)
    available_metrics = [
        key
        for key in ordered_metric_keys
        if any(
            any(
                _finite_float(group.get(key)) is not None
                for group in (
                    entry.get("metrics", {}).get("supervised", {}),
                    entry.get("metrics", {}).get("unsupervised", {}),
                )
                if isinstance(group, dict)
            )
            for per_method in scores.values()
            for entry in per_method.values()
            if isinstance(entry, dict)
        )
    ]
    metrics_available = bool(available_metrics)

    return IndexData(
        output_dir=str(run_root),
        wsis=wsis,
        methods=methods,
        scores=scores,
        metrics_available=metrics_available,
        available_metrics=available_metrics,
        ground_truth_available=bool(gt_stems),
    )


def _manifest_stems(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    slides = payload.get("slides") if isinstance(payload, dict) else None
    if not isinstance(slides, list):
        return []
    return [Path(str(name)).stem for name in slides if str(name).strip()]


def _read_method_config(method_dir: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load((method_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _method_metadata(method_dir: Path) -> dict[str, Any]:
    """Return UI metadata with the underlying model first.

    Wrapper class names are implementation details. In particular, a saved
    ``TRIDENTSegmenter`` configuration is labelled using ``params.model_id``
    (PathProfiler, HEST, GrandQC, CPG, ...). Resolution metadata is displayed
    only when it is explicitly saved in ``config.yaml``: ``target_mag`` stays
    magnification, and ``mpp`` is never inferred from magnification.
    """
    run_id = method_dir.name
    config = _read_method_config(method_dir)
    config_params = config.get("params")
    config_params = config_params if isinstance(config_params, dict) else {}
    run_params = _parse_params(run_id)

    class_path = str(config.get("class") or "").strip()
    class_name = class_path.rsplit(".", 1)[-1] if class_path else ""
    configured_display = _clean_text(config.get("display_name"))
    configured_name = _clean_text(config.get("name") or config.get("method_name"))

    model_id = _first_text(
        config.get("model_id"),
        config.get("model-id"),
        config_params.get("model_id"),
        config_params.get("model-id"),
        config_params.get("model_name"),
        config_params.get("model-name"),
        config_params.get("underlying_model"),
        config_params.get("underlying_model_id"),
        _deep_config_value(config, {"model_id", "model-id", "model_name", "model-name", "underlying_model", "underlying_model_id"}),
        run_params.get("model-id"),
        run_params.get("model-name"),
    )
    wrapper_hints = " ".join(filter(None, (class_name, configured_display, configured_name, run_id))).casefold()
    inferred_trident_model = _trident_model_from_run_id(run_id)
    is_trident = "trident" in wrapper_hints or inferred_trident_model is not None
    if is_trident and (not model_id or model_id.casefold() in {"unknown", "trident", "tridentsegmenter"}):
        model_id = inferred_trident_model
    if model_id:
        model_id = _normalise_model_label(model_id)

    # MPP is intentionally config-only. Never infer it from target_mag or from
    # the run-directory name: the viewer may display only explicitly saved MPP.
    mpp = _first_float(
        config.get("mpp"),
        config_params.get("mpp"),
        _deep_config_value(config, {"mpp"}),
    )
    target_mag = _first_float(
        config.get("target_mag"),
        config.get("target-mag"),
        config_params.get("target_mag"),
        config_params.get("target-mag"),
        _deep_config_value(config, {"target_mag", "target-mag"}),
        run_params.get("target-mag"),
    )
    if is_trident:
        primary_name = model_id or _strip_segmenter_suffix(configured_display or configured_name or class_name or "TRIDENT")
        backend = "TRIDENT"
    else:
        backend = None
        if model_id and _looks_like_model_filename(model_id):
            primary_name = Path(model_id).stem
        elif configured_display and not _is_generic_segmenter_label(configured_display):
            primary_name = _strip_wrapper_prefix(configured_display)
        elif configured_name and not _is_generic_segmenter_label(configured_name):
            primary_name = _strip_wrapper_prefix(configured_name)
        elif class_name:
            primary_name = _humanize_identifier(_strip_segmenter_suffix(class_name))
        else:
            primary_name = _humanize_run_base(run_id)

    primary_name = primary_name.strip() or _humanize_run_base(run_id)

    detail_parts: list[str] = []
    if backend:
        detail_parts.append(backend)
    if target_mag is not None:
        detail_parts.append(f"target {_fmt_number(target_mag)}×")
    if mpp is not None:
        detail_parts.append(f"{_fmt_number(mpp)} µm/px")

    if model_id and not is_trident:
        model_label = Path(model_id).name if _looks_like_model_filename(model_id) else model_id
        if model_label.casefold() not in primary_name.casefold():
            detail_parts.insert(0, f"model {model_label}")

    display_params = _display_params(config_params, run_params)
    details = " · ".join(detail_parts)
    if display_params:
        details = " · ".join(part for part in (details, _format_params(display_params)) if part)

    compact_resolution: list[str] = []
    if backend:
        compact_resolution.append(backend)
    if target_mag is not None:
        compact_resolution.append(f"{_fmt_number(target_mag)}×")
    if mpp is not None:
        compact_resolution.append(f"{_fmt_number(mpp)} µm/px")
    label = primary_name
    if compact_resolution:
        label += " (" + ", ".join(compact_resolution) + ")"

    return {
        "name": primary_name,
        "label": label,
        "details": details,
        "backend": backend,
        "model_id": model_id,
        "mpp": mpp,
        "target_mag": target_mag,
        "params": display_params,
    }


def _deep_config_value(value: Any, keys: set[str]) -> Any:
    """Return the first matching metadata value from nested saved config mappings."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in keys and child not in (None, ""):
                return child
        for child in value.values():
            found = _deep_config_value(child, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _deep_config_value(child, keys)
            if found not in (None, ""):
                return found
    return None


def _is_generic_segmenter_label(value: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value)).casefold()
    return cleaned.endswith("segmenter") or cleaned in {"segmenter", "trident", "tridentsegmenter"}


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    return value or None


def _first_text(*values: Any) -> str | None:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _strip_segmenter_suffix(value: str) -> str:
    return re.sub(r"Segmenter$", "", value, flags=re.IGNORECASE).strip()


def _strip_wrapper_prefix(value: str) -> str:
    value = re.sub(r"^TRIDENT[\s:_-]+", "", value, flags=re.IGNORECASE)
    return _strip_segmenter_suffix(value)


def _humanize_identifier(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _humanize_run_base(run_id: str) -> str:
    base = run_id.split("__", 1)[0]
    if base.casefold().startswith("trident_"):
        model = _trident_model_from_run_id(run_id)
        if model:
            return model
    return _humanize_identifier(base)


def _trident_model_from_run_id(run_id: str) -> str | None:
    base = run_id.split("__", 1)[0]
    if not base.casefold().startswith("trident_"):
        return None
    raw = base[len("trident_"):]
    return _normalise_model_label(raw)


def _normalise_model_label(value: str) -> str:
    cleaned = Path(str(value)).stem.replace("_", " ").strip()
    known = {
        "hest": "HEST",
        "grandqc": "GrandQC",
        "pathprofiler": "PathProfiler",
        "path profiler": "PathProfiler",
        "cpg": "CPG",
        "fastsam x": "FastSAM-x",
        "fastsam-x": "FastSAM-x",
    }
    return known.get(cleaned.casefold(), _humanize_identifier(cleaned))


def _looks_like_model_filename(value: str) -> bool:
    return Path(value).suffix.casefold() in {".pt", ".pth", ".onnx", ".ckpt", ".safetensors"}


def _fmt_number(value: float) -> str:
    return str(int(round(value))) if math.isclose(value, round(value), abs_tol=1e-9) else f"{value:g}"


def _display_params(config_params: dict[str, Any], run_params: dict[str, str]) -> dict[str, str]:
    """Return readable configuration parameters without wrapper internals."""
    merged: dict[str, Any] = dict(run_params)
    for key, value in config_params.items():
        merged[key.replace("_", "-")] = value

    hidden = {
        "segmenter", "reader", "model-id", "model-name", "mpp", "target-mag",
        "segmenteer-version", "cfg",
    }
    output: dict[str, str] = {}
    for key in sorted(merged):
        normalised = key.replace("_", "-")
        if normalised in hidden:
            continue
        value = merged[key]
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
            rendered = str(value)
        else:
            continue
        if len(rendered) > 80:
            continue
        output[normalised] = rendered
    return output


def _format_params(params: dict[str, str]) -> str:
    return " · ".join(f"{key}={value}" for key, value in params.items())

def _read_metrics(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            table_names = {
                str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "metrics" not in table_names:
                raise ValueError(f"SQLite metrics file has no 'metrics' table: {path}")
            for row in connection.execute("SELECT * FROM metrics"):
                yield dict(row)
        finally:
            connection.close()
        return

    if suffix != ".csv":
        raise ValueError("Metrics file must be .csv, .sqlite, .sqlite3, or .db")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _parse_metric_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    filename = str(row.get("filename") or row.get("slide_id") or row.get("image_id") or "").strip()
    stem = Path(filename).stem
    run_id = str(row.get("run_id") or row.get("method_id") or "").strip()
    if not run_id:
        # Reference evaluator variants used method_name for the method directory.
        run_id = str(row.get("method_name") or row.get("segmentation_method") or "").strip()
    if not stem or not run_id:
        return None

    supervised_sources = {
        "dice": row.get("dice") if row.get("dice") not in (None, "") else row.get("dice_score"),
        "iou": row.get("iou"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "over_segmentation_rate": row.get("over_segmentation_rate") if row.get("over_segmentation_rate") not in (None, "") else row.get("oversegmentation"),
        "under_segmentation_rate": row.get("under_segmentation_rate") if row.get("under_segmentation_rate") not in (None, "") else row.get("undersegmentation"),
    }
    supervised = {
        key: value
        for key, raw in supervised_sources.items()
        if (value := _finite_float(raw)) is not None
    }
    unsupervised = {
        key: value
        for key in UNSUPERVISED_METRICS
        if (value := _finite_float(row.get(key))) is not None
    }
    metrics: dict[str, Any] = {}
    if supervised:
        metrics["supervised"] = supervised
    if unsupervised:
        metrics["unsupervised"] = unsupervised
    return stem, run_id, {
        "metrics": metrics,
        "evaluation_mode": str(row.get("evaluation_mode") or row.get("evaluation_status") or ""),
        "evaluation_error": str(row.get("evaluation_error") or ""),
    }


def _resolve_metric_method(row: dict[str, Any], methods: dict[str, MethodInfo]) -> str | None:
    """Resolve evaluator rows to runner methods without letting metrics define methods."""
    direct_candidates = [
        row.get("run_id"), row.get("method_id"), row.get("method_name"), row.get("segmentation_method"),
    ]
    for candidate in direct_candidates:
        text = str(candidate or "").strip()
        if text in methods:
            return text

    labels = {
        str(row.get("formal_name") or "").strip().casefold(),
        str(row.get("display_name") or "").strip().casefold(),
    }
    labels.discard("")
    if not labels:
        return None
    matches = [
        run_id for run_id, method in methods.items()
        if method.name.casefold() in labels or method.label.casefold() in labels
    ]
    return matches[0] if len(matches) == 1 else None


def _metric_group(entry: dict[str, Any], group: str) -> bool:
    return isinstance(entry.get("metrics", {}).get(group), dict) and bool(entry["metrics"][group])


def _has_usable_metrics(entry: dict[str, Any]) -> bool:
    """Return whether an evaluator row contributes any usable metric value."""
    metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
    if not isinstance(metrics, dict):
        return False
    for group in ("supervised", "unsupervised"):
        values = metrics.get(group, {})
        if isinstance(values, dict) and any(_finite_float(value) is not None for value in values.values()):
            return True
    return False


def _resolve_wsi_path(data_dir: Path | None, stem: str) -> Path | None:
    if data_dir is None or not data_dir.is_dir():
        return None
    for extension in _WSI_EXTENSIONS:
        candidate = data_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


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


def _parse_params(run_id: str) -> dict[str, str]:
    _, separator, tail = run_id.partition("__")
    if not separator:
        return {}
    params: dict[str, str] = {}
    for piece in tail.split("_"):
        if "=" in piece:
            key, value = piece.split("=", 1)
            if key != "cfg":
                params[key] = value
    return params


_RESERVED_LAYER_COLORS = {
    "#2563eb",  # fixed ground-truth blue
}

# Ordered for perceptual separation, not by hue.  The sequence deliberately
# jumps between red/green/purple/orange/teal/magenta/olive families so adjacent
# method variants do not receive near-identical colours.  The GT blue family is
# reserved and absent from this prediction palette.
_DISTINCT_METHOD_PALETTE = (
    # Farthest-point sampled in CIELAB from a broad sRGB candidate set.
    # The fixed GT blue family is excluded. The order itself is the assignment
    # sequence, so every method in one experiment receives a unique colour and
    # early colours are maximally separated from one another.
    "#db2424", "#14eb14", "#144224", "#ebcc52", "#eb14ad", "#14ebeb",
    "#522452", "#42ad52", "#ad7052", "#9e14eb", "#eb80ad", "#eb8014",
    "#8f1433", "#addb14", "#707014", "#428f8f", "#db80eb", "#eb1470",
    "#523324", "#801480", "#14ebad", "#14eb70", "#ebbd80", "#8fad33",
    "#eb7070", "#148052", "#eb33eb", "#9e3314", "#52bd14", "#ad2470",
    "#ad8014", "#337014", "#52bd9e", "#9e5270", "#525224", "#eb8052",
    "#ad14ad", "#245252", "#8fdb61", "#cc2442", "#eb61bd", "#704214",
    "#809e52", "#9e529e", "#612433", "#ad9e52", "#ebad14", "#cccc14",
    "#80eb14", "#db5214", "#5214eb", "#ad6114", "#14bdcc", "#702414",
    "#eb9e80", "#eb9e52", "#ad5252", "#70eb8f", "#ad9e14", "#8f7033",
    "#db5280", "#eb61eb", "#527033", "#33bd80",
)


def _method_colors(run_ids: list[str]) -> dict[str, str]:
    """Assign a distinct prediction colour to every method in this experiment.

    Assignment is deterministic for a fixed experiment and intentionally uses
    the palette in a globally high-separation order. No colour is hashed or recycled.
    The palette was precomputed by farthest-point sampling in CIELAB space from
    a broad sRGB candidate set while excluding the fixed ground-truth blue family.
    """
    ordered = sorted(run_ids, key=lambda value: (value.casefold(), value))
    if len(ordered) > len(_DISTINCT_METHOD_PALETTE):
        raise ValueError(
            f"Viewer distinct-colour palette supports {len(_DISTINCT_METHOD_PALETTE)} methods; "
            f"found {len(ordered)}. Extend _DISTINCT_METHOD_PALETTE before displaying this run."
        )
    return {
        run_id: _DISTINCT_METHOD_PALETTE[index]
        for index, run_id in enumerate(ordered)
    }


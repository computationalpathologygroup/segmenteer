from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from segmenteer.benchmark.indexes import (
    CURRENT_ARTIFACT_SCHEMA_VERSION,
    iter_completed_artifacts,
)
from segmenteer.benchmark.locking import output_index_lock
from segmenteer.benchmark.resume import fingerprint, image_fingerprint
from segmenteer.benchmark.reporting import (
    score_to_dataset_csv_row,
    write_dataset_results_csv_rows,
)

if TYPE_CHECKING:
    from segmenteer.benchmark.runner import BenchmarkResult

__all__ = [
    "EnsembleOutputWriter",
    "load_ensemble_members",
    "make_ensemble_run_id",
]

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return float(obj)
    return obj


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace a text artifact so a killed run leaves no valid half-file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, data: object) -> None:
    _write_text_atomic(path, json.dumps(_sanitize(data), indent=2))


# ---------------------------------------------------------------------------
# Canonical filenames inside every member directory
# ---------------------------------------------------------------------------

PREDICTIONS_DIR = "predictions"
GROUND_TRUTH_DIR = "ground_truth"
EVAL_DIR = "eval"
EVAL_SCORES_DIR = "eval/scores"
EVAL_HEATMAPS_DIR = "eval/heatmaps"
EVAL_ERRORS_DIR = "eval/errors"
EVAL_ERROR_MAPS_DIR = "eval/error_maps"
EVAL_COMPLETED_DIR = "eval/completed"
CONFIG_FILE = "config.yaml"
MANIFEST_FILE = "ensemble_manifest.json"


# ---------------------------------------------------------------------------
# Ensemble run_id construction
# ---------------------------------------------------------------------------


def make_ensemble_run_id(
    member_run_ids: list[str],
    strategy: str = "majority",
    threshold: float = 0.5,
) -> str:
    """Return a filesystem-safe run_id for a combined ensemble prediction.

    Follows the same ``<name>__<param>=<val>`` convention as individual method
    run_ids so the viewer and writer treat ensemble results identically.

    Example
    -------
    >>> make_ensemble_run_id(["fesi__mpp=20", "otsu__mpp=5"], threshold=0.5)
    'ensemble__members=2_strategy=majority_threshold=0.5'
    """
    n = len(member_run_ids)
    # Format threshold: strip trailing zero decimals (0.5 → "0.5", 1.0 → "1")
    thr_str = f"{threshold:g}"
    return f"ensemble__members={n}_strategy={strategy}_threshold={thr_str}"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class EnsembleOutputWriter:
    """Saves each benchmark result as a self-contained ensemble member.

    Works as a drop-in ``result_callback`` for :class:`BenchmarkRunner`::

        writer  = seg.EnsembleOutputWriter(output_dir)
        runner  = seg.BenchmarkRunner(reporter=reporter, result_callback=writer)

        # single image
        results = runner.run_multiple(segmenters, path)
        writer.finalize(results)

        # dataset
        all_results = runner.run_dataset(segmenters, images)
        writer.finalize_dataset(all_results)

    Parameters
    ----------
    output_dir:
        Root directory for the entire run.
    heatmap_mpp:
        Resolution used when rendering the heatmap overlay.
    heatmap_max_size:
        Longest edge of the heatmap thumbnail in pixels.
    heatmap_alpha:
        Opacity of the segmentation overlay (0–1).
    """

    def __init__(
        self,
        output_dir: Path | str,
        image_path: Path | str | None = None,
        heatmap_mpp: float = 10.0,
        heatmap_max_size: int = 1024,
        heatmap_alpha: float = 0.4,
        save_heatmaps: bool = False,
        index_refresh_interval: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.image_path = Path(image_path) if image_path else None
        self.heatmap_mpp = heatmap_mpp
        self.heatmap_max_size = heatmap_max_size
        self.heatmap_alpha = heatmap_alpha
        # Heatmaps require an additional WSI read. Keep them opt-in so the
        # benchmark path writes only predictions plus metadata by default.
        self.save_heatmaps = save_heatmaps
        if index_refresh_interval < 1:
            raise ValueError("index_refresh_interval must be at least 1.")
        # Dataset root indexes are derived by scanning all committed artifacts.
        # Refreshing them after every slide becomes quadratic on large cohorts;
        # workflows use a modest batch interval while direct writer use remains
        # immediately visible by default for backwards compatibility.
        self.index_refresh_interval = int(index_refresh_interval)
        self._pending_dataset_index_updates = 0

        self._members: list[dict] = []  # flat list for single-image runs
        # run_id → list[per-image entry] for dataset runs
        self._run_members: dict[str, list[dict]] = defaultdict(list)
        # run_id → method class name (for manifest header)
        self._run_names: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Callable interface — plug directly into result_callback
    # ------------------------------------------------------------------

    def __call__(self, result: BenchmarkResult) -> None:
        if result.failed:
            return  # skip writing empty output for failed runs
        # ``save_member`` owns batched dataset-index refreshes so direct calls
        # and callback-driven calls have identical semantics.
        self.save_member(result)

    # ------------------------------------------------------------------
    # Core save logic
    # ------------------------------------------------------------------

    def save_member(self, result: BenchmarkResult) -> Path:
        """Persist prediction plus lightweight reproducibility metadata.

        Ground truth is copied once when available so post-run evaluation can
        derive metrics from saved GeoJSON.  Polygon metrics, visual-QA layers,
        and heatmaps are deliberately not produced unless explicitly requested.
        """
        from segmenteer.io.loader import save_geojson

        image_path: Path | None = result.image_path or self.image_path
        is_dataset = self.image_path is None and result.image_path is not None
        image_stem = image_path.stem if image_path else "image"

        method_dir = self.output_dir / result.run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        scores_dir = method_dir / EVAL_SCORES_DIR
        heatmaps_dir = method_dir / EVAL_HEATMAPS_DIR
        predictions_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)
        if self.save_heatmaps:
            heatmaps_dir.mkdir(parents=True, exist_ok=True)

        # 1. Prediction GeoJSON ------------------------------------------------
        annotations_rel = f"{result.run_id}/{PREDICTIONS_DIR}/{image_stem}.geojson"
        save_geojson(result.geojson, predictions_dir / f"{image_stem}.geojson")

        # 2. Segmenter configuration -----------------------------------------
        # A configuration mismatch is invalidated before this writer is called
        # during resume.  Rewriting the current config here also repairs a
        # partially written config file from an interrupted fresh run.
        config_rel = f"{result.run_id}/{CONFIG_FILE}"
        config_path = method_dir / CONFIG_FILE
        config_fingerprint: str | None = None
        if result.segmenter_config is not None:
            import yaml

            config_fingerprint = fingerprint(result.segmenter_config)
            _write_text_atomic(
                config_path,
                yaml.safe_dump(
                    result.segmenter_config,
                    default_flow_style=False,
                    sort_keys=False,
                ),
            )

        # 3. Ground-truth reference (no evaluation in the benchmark path) -----
        # Ground truth is persisted once so a later post-run evaluator can work
        # entirely from this output folder.  Deliberately do not create TP/FP/FN
        # layers or error maps here: those require geometry operations and are
        # evaluation artifacts, not inference artifacts.
        ground_truth_rel: str | None = None
        if result.ground_truth_geojson is not None:
            ground_truth_dir = self.output_dir / GROUND_TRUTH_DIR
            ground_truth_dir.mkdir(parents=True, exist_ok=True)
            ground_truth_path = ground_truth_dir / f"{image_stem}.geojson"
            if not ground_truth_path.exists():
                save_geojson(result.ground_truth_geojson, ground_truth_path)
            ground_truth_rel = f"{GROUND_TRUTH_DIR}/{image_stem}.geojson"

        # 4. Score JSON ---------------------------------------------------------
        eval_data: dict = {
            "artifact_schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
            "method_name": result.method_name,
            "run_id": result.run_id,
            "image_stem": image_stem,
            "image_path": str(image_path) if image_path else None,
            "execution_time_s": result.execution_time,
            "seconds_per_pixel": result.seconds_per_pixel,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_mode": (
                "ground_truth_available"
                if result.ground_truth_geojson is not None
                else "prediction_only"
            ),
            "ground_truth_available": result.ground_truth_geojson is not None,
            "segmenter_config": result.segmenter_config,
            "segmenter_config_fingerprint": config_fingerprint,
            "source_image_fingerprint": image_fingerprint(image_path) if image_path else None,
            "native_spacing_um_per_px": list(result.native_spacing) if result.native_spacing else None,
            "magnification_used": result.magnification_used,
            "ground_truth_fingerprint": (
                fingerprint(result.ground_truth_geojson)
                if result.ground_truth_geojson is not None
                else None
            ),
        }
        if ground_truth_rel:
            eval_data["ground_truth"] = {
                "path": ground_truth_rel,
                "coordinate_space": "level_0_pixels",
            }
        scores_rel = f"{result.run_id}/{EVAL_SCORES_DIR}/{image_stem}.json"
        _write_json(scores_dir / f"{image_stem}.json", eval_data)

        # 5. Optional prediction heatmap --------------------------------------
        heatmap_rel = None
        if self.save_heatmaps and image_path is not None:
            from segmenteer.visualization.heatmaps import save_heatmap_thumbnail

            heatmap_rel = f"{result.run_id}/{EVAL_HEATMAPS_DIR}/{image_stem}.png"
            try:
                save_heatmap_thumbnail(
                    image=image_path,
                    geojson_data=result.geojson,
                    output_path=heatmaps_dir / f"{image_stem}.png",
                    mpp=self.heatmap_mpp,
                    max_size=self.heatmap_max_size,
                    alpha=self.heatmap_alpha,
                )
            except Exception as exc:
                heatmap_rel = None
                print(f"  ! prediction heatmap skipped for {image_stem}: {type(exc).__name__}: {exc}")

        # 6. Completion marker --------------------------------------------------
        # This file is written only after prediction + score persistence.  It
        # lets later resumes distinguish a valid completed item from an
        # interruption that occurred mid-write.  Legacy folders without this
        # marker are still accepted via config.yaml validation.
        completed_dir = method_dir / EVAL_COMPLETED_DIR
        completed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            completed_dir / f"{image_stem}.json",
            {
                "artifact_schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
                "run_id": result.run_id,
                "image_stem": image_stem,
                "segmenter_config_fingerprint": config_fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # 7. Register in manifest lists ---------------------------------------
        entry: dict = {
            "name": result.method_name,
            "run_id": result.run_id,
            "path": result.run_id,
            "image_stem": image_stem,
            "image": str(image_path) if image_path else None,
            "annotations": annotations_rel,
            "scores": scores_rel,
            "config": config_rel,
            "heatmap": heatmap_rel,
        }
        if ground_truth_rel:
            entry["ground_truth"] = ground_truth_rel
        if is_dataset:
            self._run_members[result.run_id].append(entry)
            self._run_names[result.run_id] = result.method_name
        else:
            self._members.append(entry)

        return method_dir

    # ------------------------------------------------------------------
    # Ensemble prediction — treated as a first-class method
    # ------------------------------------------------------------------

    def save_ensemble_result(
        self,
        image_path: Path | str,
        geojson: dict,
        member_run_ids: list[str],
        *,
        strategy: str = "majority",
        threshold: float = 0.5,
        ground_truth_geojson: dict | None = None,
    ) -> Path:
        """Save an ensemble prediction without evaluating it inline.

        The same post-run ``evaluate_outputs.py`` workflow used for individual
        methods can evaluate this prediction later.  This avoids treating an
        ensemble write as a hidden, expensive metric pass.
        """
        from segmenteer.io.loader import save_geojson

        image_path = Path(image_path)
        image_stem = image_path.stem
        run_id = make_ensemble_run_id(member_run_ids, strategy=strategy, threshold=threshold)

        method_dir = self.output_dir / run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        scores_dir = method_dir / EVAL_SCORES_DIR
        predictions_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)

        save_geojson(geojson, predictions_dir / f"{image_stem}.geojson")

        config_path = method_dir / "config.json"
        ensemble_config = {
            "type": "ensemble",
            "strategy": strategy,
            "threshold": threshold,
            "n_members": len(member_run_ids),
            "member_run_ids": member_run_ids,
        }
        if not config_path.exists():
            _write_json(config_path, ensemble_config)
        config_fingerprint = fingerprint(ensemble_config)

        ground_truth_rel: str | None = None
        if ground_truth_geojson is not None:
            ground_truth_dir = self.output_dir / GROUND_TRUTH_DIR
            ground_truth_dir.mkdir(parents=True, exist_ok=True)
            gt_path = ground_truth_dir / f"{image_stem}.geojson"
            if not gt_path.exists():
                save_geojson(ground_truth_geojson, gt_path)
            ground_truth_rel = f"{GROUND_TRUTH_DIR}/{image_stem}.geojson"

        # Read only level-0 metadata.  No polygon metrics or mask rasterisation
        # occur here.
        native_spacing = None
        try:
            from segmenteer.core.base import get_wsi_reader

            reader = get_wsi_reader()
            wsi = reader.read(str(image_path))
            x_mpp, y_mpp = reader.get_mpp(wsi, 0)
            native_spacing = [float(x_mpp), float(y_mpp)]
        except Exception:
            pass

        eval_data: dict = {
            "artifact_schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
            "method_name": "ensemble",
            "run_id": run_id,
            "image_stem": image_stem,
            "image_path": str(image_path),
            "strategy": strategy,
            "threshold": threshold,
            "member_run_ids": member_run_ids,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_mode": (
                "ground_truth_available" if ground_truth_geojson is not None else "prediction_only"
            ),
            "ground_truth_available": ground_truth_geojson is not None,
            "native_spacing_um_per_px": native_spacing,
            "magnification_used": None,
            "segmenter_config_fingerprint": config_fingerprint,
        }
        if ground_truth_rel:
            eval_data["ground_truth"] = {
                "path": ground_truth_rel,
                "coordinate_space": "level_0_pixels",
            }
        _write_json(scores_dir / f"{image_stem}.json", eval_data)

        completed_dir = method_dir / EVAL_COMPLETED_DIR
        completed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            completed_dir / f"{image_stem}.json",
            {
                "artifact_schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
                "run_id": run_id,
                "image_stem": image_stem,
                "segmenter_config_fingerprint": config_fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        if self.save_heatmaps:
            from segmenteer.visualization.heatmaps import save_heatmap_thumbnail

            heatmaps_dir = method_dir / EVAL_HEATMAPS_DIR
            heatmaps_dir.mkdir(parents=True, exist_ok=True)
            try:
                save_heatmap_thumbnail(
                    image=image_path,
                    geojson_data=geojson,
                    output_path=heatmaps_dir / f"{image_stem}.png",
                    mpp=self.heatmap_mpp,
                    max_size=self.heatmap_max_size,
                    alpha=self.heatmap_alpha,
                )
            except Exception as exc:
                print(f"  ! prediction heatmap skipped for {image_stem}: {type(exc).__name__}: {exc}")

        if self.image_path is None:
            self._pending_dataset_index_updates += 1
            if self._pending_dataset_index_updates >= self.index_refresh_interval:
                self.refresh_dataset_indexes()
                self._pending_dataset_index_updates = 0
        return method_dir

    # ------------------------------------------------------------------
    # Finalise — single image
    # ------------------------------------------------------------------

    def finalize(
        self,
        results: list[BenchmarkResult] | None = None,
        image_path: Path | str | None = None,
    ) -> Path:
        """Write ``ensemble_manifest.json`` for a single-image run."""
        manifest: dict = {
            "segmenteer_version": _segmenteer_version(),
            "mode": "single",
            "image": str(image_path) if image_path else None,
            "run_dir": str(self.output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_members": len(self._members),
            "members": self._members,
        }
        if results:
            manifest["summary"] = _build_summary(results)

        manifest_path = self.output_dir / MANIFEST_FILE
        _write_json(manifest_path, manifest)
        return manifest_path

    # ------------------------------------------------------------------
    # Finalise — dataset
    # ------------------------------------------------------------------

    def _load_dataset_metadata(self) -> dict | None:
        """Read the stable dataset contract written by the workflow, if any."""
        path = self.output_dir / "dataset_manifest.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _scan_dataset_members(
        self,
    ) -> tuple[
        dict[str, list[dict]],
        dict[str, str],
        list[dict[str, str]],
        dict[tuple[str, str], dict],
    ]:
        """Rebuild all derived indexes from committed artifacts on disk."""
        groups: dict[str, list[dict]] = defaultdict(list)
        names: dict[str, str] = {}
        rows: list[dict[str, str]] = []
        scores: dict[tuple[str, str], dict] = {}

        for artifact in iter_completed_artifacts(self.output_dir):
            run_id = artifact.run_id
            image_stem = artifact.image_stem
            score = artifact.score
            config_filename = (
                CONFIG_FILE
                if (artifact.method_dir / CONFIG_FILE).is_file()
                else "config.json"
            )
            entry: dict = {
                "name": str(score.get("method_name") or run_id),
                "run_id": run_id,
                "path": run_id,
                "image_stem": image_stem,
                "image": score.get("image_path"),
                "annotations": f"{run_id}/{PREDICTIONS_DIR}/{image_stem}.geojson",
                "scores": f"{run_id}/{EVAL_SCORES_DIR}/{image_stem}.json",
                "config": f"{run_id}/{config_filename}",
                "heatmap": (
                    f"{run_id}/{EVAL_HEATMAPS_DIR}/{image_stem}.png"
                    if (artifact.method_dir / EVAL_HEATMAPS_DIR / f"{image_stem}.png").is_file()
                    else None
                ),
            }
            ground_truth = score.get("ground_truth")
            if isinstance(ground_truth, dict) and ground_truth.get("path"):
                entry["ground_truth"] = ground_truth["path"]
                layers = score.get("evaluation_layers")
                if isinstance(layers, dict):
                    entry["evaluation_layers"] = {
                        key: layers.get(key)
                        for key in ("true_positive", "false_positive", "false_negative")
                        if layers.get(key)
                    }
                    entry["error_map"] = layers.get("error_map")

            groups[run_id].append(entry)
            names[run_id] = entry["name"]
            rows.append(score_to_dataset_csv_row(score))
            scores[(run_id, image_stem)] = score

        self._run_members = defaultdict(list, groups)
        self._run_names = names
        return groups, names, rows, scores

    @staticmethod
    def _summary_from_score(score: dict) -> dict | None:
        """Convert persisted score metadata into a lightweight manifest row."""
        if not isinstance(score, dict):
            return None
        row: dict = {
            "run_id": score.get("run_id"),
            "method": score.get("method_name"),
            "evaluation_mode": score.get("evaluation_mode", "prediction_only"),
            "execution_time_s": score.get("execution_time_s", 0.0),
            "native_spacing_um_per_px": score.get("native_spacing_um_per_px"),
            "magnification_used": score.get("magnification_used"),
        }
        # Legacy output folders may already contain metrics; surface those
        # values without requiring new runs to calculate any.
        metrics = score.get("metrics")
        if isinstance(metrics, dict):
            unsupervised = metrics.get("unsupervised")
            if isinstance(unsupervised, dict):
                for key in ("num_objects", "coverage_ratio", "mean_area", "mean_compactness", "mean_solidity"):
                    if key in unsupervised:
                        row[key] = unsupervised[key]
            supervised = metrics.get("supervised")
            if isinstance(supervised, dict):
                for key in ("dice", "iou", "precision", "recall"):
                    if key in supervised:
                        row[key] = supervised[key]
        return row

    def _write_dataset_manifest(
        self,
        groups: dict[str, list[dict]],
        names: dict[str, str],
        scores: dict[tuple[str, str], dict],
        dataset_metadata: dict | None,
    ) -> Path:
        members_block: list[dict] = []
        for run_id in sorted(groups, key=str.casefold):
            image_entries = sorted(groups[run_id], key=lambda entry: entry["image_stem"].casefold())
            member_entry: dict = {
                "run_id": run_id,
                "name": names.get(run_id, run_id),
                "n_images": len(image_entries),
                "images": image_entries,
            }
            summary_rows = [
                summary
                for entry in image_entries
                if (summary := self._summary_from_score(scores[(run_id, entry["image_stem"])]))
                is not None
            ]
            if summary_rows:
                member_entry["summary"] = summary_rows
            members_block.append(member_entry)

        all_image_stems = sorted(
            {entry["image_stem"] for image_entries in groups.values() for entry in image_entries},
            key=str.casefold,
        )
        manifest: dict = {
            "segmenteer_version": _segmenteer_version(),
            "mode": "dataset",
            "run_dir": str(self.output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_methods": len(groups),
            "n_images": len(all_image_stems),
            "images": all_image_stems,
            "members": members_block,
        }
        if dataset_metadata is not None:
            manifest["dataset"] = dataset_metadata

        manifest_path = self.output_dir / MANIFEST_FILE
        _write_json(manifest_path, manifest)
        return manifest_path

    def refresh_dataset_indexes(self, dataset_metadata: dict | None = None) -> Path:
        """Atomically rebuild shared ``results.csv`` and the dataset manifest.

        The root lock is deliberately held only during this cheap disk scan and
        replacement of derived files.  Model inference and method-owned writes
        remain fully parallel across distinct ``run_id`` directories.
        """
        with output_index_lock(self.output_dir):
            groups, names, rows, scores = self._scan_dataset_members()
            write_dataset_results_csv_rows(rows, self.output_dir / "results.csv")
            metadata = dataset_metadata if dataset_metadata is not None else self._load_dataset_metadata()
            manifest = self._write_dataset_manifest(groups, names, scores, metadata)
        self._pending_dataset_index_updates = 0
        return manifest

    def finalize_dataset(
        self,
        all_results: dict[Path, list[BenchmarkResult]] | None = None,
        dataset_metadata: dict | None = None,
    ) -> Path:
        """Rebuild the complete dataset indexes from committed output artifacts."""
        del all_results  # Derived indexes intentionally do not trust local process state.
        return self.refresh_dataset_indexes(dataset_metadata=dataset_metadata)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def member_dirs(self) -> list[Path]:
        """All saved member directories in insertion order."""
        all_entries = list(self._members)
        for img_entries in self._run_members.values():
            all_entries.extend(img_entries)
        return [self.output_dir / e["path"] for e in all_entries]


# ---------------------------------------------------------------------------
# Consumer-side loader
# ---------------------------------------------------------------------------


def load_ensemble_members(
    manifest_path: Path | str,
    image_stem: str | None = None,
) -> list[dict]:
    """Load ensemble members from a manifest produced by :class:`EnsembleOutputWriter`.

    Parameters
    ----------
    manifest_path:
        Path to ``ensemble_manifest.json``.
    image_stem:
        For *dataset* manifests: restrict to one image (by stem).
        When ``None`` all images are loaded.

    Returns
    -------
    List of member dicts, each with:

    - ``name``       – method class name
    - ``run_id``     – unique hyperparam-aware id
    - ``geojson``    – loaded FeatureCollection
    - ``metrics``    – ``{"unsupervised": …, "supervised": …}``
    - ``metadata``   – timing, image_path, etc.
    - ``member_dir`` – absolute :class:`~pathlib.Path`

    Example — single image::

        members = seg.load_ensemble_members("outputs/1903_1542/ensemble_manifest.json")

    Example — dataset, one image at a time::

        for image in images:
            members = seg.load_ensemble_members(manifest, image_stem=image.stem)
            geojsons = [m["geojson"] for m in members]
            # → pass to your ensemble fusion function
    """
    manifest_path = Path(manifest_path)
    run_dir = manifest_path.parent

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    mode = manifest.get("mode", "single")

    # Collect raw manifest entries
    raw_entries: list[dict] = []
    if mode == "dataset":
        # members[] is method-first; images[] inside each member is per-image.
        # If image_stem is given, return one entry per method for that image.
        # If not given, return every entry across all methods and images.
        for method_block in manifest.get("members", []):
            for img_entry in method_block.get("images", []):
                if image_stem is not None and img_entry.get("image_stem") != image_stem:
                    continue
                raw_entries.append(img_entry)
    else:
        raw_entries = manifest.get("members", [])

    members = []
    for entry in raw_entries:
        member_dir = run_dir / entry["path"]

        with (run_dir / entry["annotations"]).open(encoding="utf-8") as fh:
            geojson_data = json.load(fh)

        # New layout: eval/scores/<stem>.json + eval/heatmaps/<stem>.png.
        # Compat: previous layout used a single "eval" key or separate metrics/metadata.
        if "scores" in entry:
            with (run_dir / entry["scores"]).open(encoding="utf-8") as fh:
                eval_data = json.load(fh)
            metrics = eval_data.get("metrics", {})
            metadata = {k: v for k, v in eval_data.items() if k != "metrics"}
        elif "eval" in entry:
            with (run_dir / entry["eval"]).open(encoding="utf-8") as fh:
                eval_data = json.load(fh)
            metrics = eval_data.get("metrics", {})
            metadata = {k: v for k, v in eval_data.items() if k != "metrics"}
        else:
            with (run_dir / entry["metrics"]).open(encoding="utf-8") as fh:
                metrics = json.load(fh)
            with (run_dir / entry["metadata"]).open(encoding="utf-8") as fh:
                metadata = json.load(fh)

        members.append(
            {
                "name": entry["name"],
                "run_id": entry.get("run_id", entry["name"]),
                "geojson": geojson_data,
                "metrics": metrics,
                "metadata": metadata,
                "member_dir": member_dir,
            }
        )

    return members


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_summary(results: list[BenchmarkResult]) -> list[dict]:
    """Build a metadata-first summary without triggering metric computation."""
    rows = []
    for result in results:
        row: dict = {
            "run_id": result.run_id,
            "method": result.method_name,
            "evaluation_mode": (
                "ground_truth_available"
                if result.ground_truth_geojson is not None
                else "prediction_only"
            ),
            "execution_time_s": result.execution_time,
            "native_spacing_um_per_px": result.native_spacing,
            "magnification_used": result.magnification_used,
        }
        if result.unsupervised_metrics is not None:
            unsupervised = result.unsupervised_metrics
            row.update(
                {
                    "num_objects": unsupervised.num_objects,
                    "coverage_ratio": unsupervised.coverage_ratio,
                    "mean_area": unsupervised.mean_area,
                    "mean_compactness": unsupervised.mean_compactness,
                    "mean_solidity": unsupervised.mean_solidity,
                }
            )
        if result.supervised_metrics is not None:
            supervised = result.supervised_metrics
            row.update(
                {
                    "dice": supervised.dice,
                    "iou": supervised.iou,
                    "precision": supervised.precision,
                    "recall": supervised.recall,
                }
            )
        rows.append(row)
    return rows


def _segmenteer_version() -> str:
    try:
        from importlib.metadata import version

        return version("segmenteer")
    except Exception:
        return "unknown"

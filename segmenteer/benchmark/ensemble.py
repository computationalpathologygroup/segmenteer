"""Ensemble-ready output writer for benchmark runs.

Output layout (single image and dataset runs share the same structure)
----------------------------------------------------------------------
    <output_dir>/
        ensemble_manifest.json
        results.csv / results.json
        thumbnails/
            <image_stem>.png
        <run_id>/                      # one dir per method+hyperparams
            config.yaml                # segmenter config – written once
            predictions/
                <image_stem>.geojson   # same stem as the source image file
            eval/
                scores/
                    <image_stem>.json  # metrics + metadata merged
                heatmaps/
                    <image_stem>.png   # heatmap overlay

``run_id`` encodes the method class-name *and* its hyperparameters, so two
instances of the same class with different params live in separate directories
and never overwrite each other.

Every member directory is self-contained: a downstream ensembler only needs
the manifest to discover and load all members.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(_sanitize(data), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Canonical filenames inside every member directory
# ---------------------------------------------------------------------------

PREDICTIONS_DIR = "predictions"
EVAL_DIR = "eval"
EVAL_SCORES_DIR = "eval/scores"
EVAL_HEATMAPS_DIR = "eval/heatmaps"
CONFIG_FILE = "config.yaml"
MANIFEST_FILE = "ensemble_manifest.json"
THUMBNAILS_DIR = "thumbnails"


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
    ) -> None:
        self.output_dir = Path(output_dir)
        self.image_path = Path(image_path) if image_path else None
        self.heatmap_mpp = heatmap_mpp
        self.heatmap_max_size = heatmap_max_size
        self.heatmap_alpha = heatmap_alpha

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
        self.save_member(result)

    # ------------------------------------------------------------------
    # Core save logic
    # ------------------------------------------------------------------

    def save_member(self, result: BenchmarkResult) -> Path:
        """Persist *result* as a self-contained member directory.

        Layout::

            <output_dir>/<run_id>/
                config.yaml                  ← written once per method
                predictions/
                    <image_stem>.geojson
                eval/
                    scores/
                        <image_stem>.json    ← metrics + metadata merged
                    heatmaps/
                        <image_stem>.png     ← heatmap overlay
        """
        from segmenteer.io.loader import save_geojson
        from segmenteer.visualization.heatmaps import save_heatmap_thumbnail

        image_path: Path = result.image_path or self.image_path
        is_dataset = result.image_path is not None
        image_stem = image_path.stem if image_path else "image"

        method_dir = self.output_dir / result.run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        scores_dir = method_dir / EVAL_SCORES_DIR
        heatmaps_dir = method_dir / EVAL_HEATMAPS_DIR
        predictions_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)
        heatmaps_dir.mkdir(parents=True, exist_ok=True)

        # 1. Annotations → predictions/<image_stem>.geojson --------------------
        annotations_rel = f"{result.run_id}/{PREDICTIONS_DIR}/{image_stem}.geojson"
        save_geojson(result.geojson, predictions_dir / f"{image_stem}.geojson")

        # 1b. Prompt → prompts/<image_stem>.txt (only for contextual segmenters)
        if result.prompt is not None:
            prompts_dir = method_dir / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (prompts_dir / f"{image_stem}.txt").write_text(result.prompt, encoding="utf-8")

        # 2. Config YAML → <run_id>/config.yaml (once per method) --------------
        config_rel = f"{result.run_id}/{CONFIG_FILE}"
        config_path = method_dir / CONFIG_FILE
        if result.segmenter_config is not None and not config_path.exists():
            import yaml  # lazy import – only needed at save time

            config_path.write_text(
                yaml.dump(
                    result.segmenter_config, default_flow_style=False, sort_keys=False
                ),
                encoding="utf-8",
            )

        # 3. Scores → eval/scores/<image_stem>.json (metrics + metadata merged)
        eval_data: dict = {
            "method_name": result.method_name,
            "run_id": result.run_id,
            "image_stem": image_stem,
            "image_path": str(image_path) if image_path else None,
            "execution_time_s": result.execution_time,
            "seconds_per_pixel": result.seconds_per_pixel,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "unsupervised": asdict(result.unsupervised_metrics),
            },
        }
        if result.supervised_metrics is not None:
            eval_data["metrics"]["supervised"] = asdict(result.supervised_metrics)
        scores_rel = f"{result.run_id}/{EVAL_SCORES_DIR}/{image_stem}.json"
        _write_json(scores_dir / f"{image_stem}.json", eval_data)

        # 4. Heatmap → eval/heatmaps/<image_stem>.png --------------------------
        heatmap_rel = f"{result.run_id}/{EVAL_HEATMAPS_DIR}/{image_stem}.png"
        if image_path is not None:
            save_heatmap_thumbnail(
                image=image_path,
                geojson_data=result.geojson,
                output_path=heatmaps_dir / f"{image_stem}.png",
                mpp=self.heatmap_mpp,
                max_size=self.heatmap_max_size,
                alpha=self.heatmap_alpha,
            )

        # 5. Register in manifest lists ----------------------------------------
        entry = {
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
        """Save the combined ensemble prediction as a first-class method directory.

        The output follows the exact same layout as ``save_member``::

            <output_dir>/<ensemble_run_id>/
                config.json                  ← lists member run_ids + strategy
                predictions/
                    <image_stem>.geojson
                eval/
                    scores/
                        <image_stem>.json    ← metrics + ensemble metadata
                    heatmaps/
                        <image_stem>.png

        This means the viewer, loader, and any downstream tool that searches
        for ``<run_id>/predictions/*.geojson`` will pick up the ensemble
        result automatically — no special-casing needed.

        Parameters
        ----------
        image_path:
            Source WSI used to compute metrics and the heatmap overlay.
        geojson:
            Combined FeatureCollection produced by your ensemble strategy.
        member_run_ids:
            Ordered list of member ``run_id``s that were fused (used to build
            the ``run_id`` slug and recorded in ``config.json``).
        strategy:
            Human-readable fusion strategy name (e.g. ``"majority"``,
            ``"weighted"``).
        threshold:
            Vote threshold or confidence cut-off used during fusion.
        ground_truth_geojson:
            Optional ground-truth FeatureCollection for supervised metrics.
        """
        from segmenteer.io.loader import save_geojson
        from segmenteer.metrics.supervised import compute_all_supervised_metrics
        from segmenteer.metrics.unsupervised import compute_unsupervised_metrics
        from segmenteer.visualization.heatmaps import save_heatmap_thumbnail

        image_path = Path(image_path)
        image_stem = image_path.stem
        run_id = make_ensemble_run_id(member_run_ids, strategy=strategy, threshold=threshold)

        method_dir = self.output_dir / run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        scores_dir = method_dir / EVAL_SCORES_DIR
        heatmaps_dir = method_dir / EVAL_HEATMAPS_DIR
        for d in (predictions_dir, scores_dir, heatmaps_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1. Predictions -------------------------------------------------------
        save_geojson(geojson, predictions_dir / f"{image_stem}.geojson")

        # 2. Config (JSON, not YAML — no segmenter class to serialise) ----------
        config_path = method_dir / "config.json"
        if not config_path.exists():
            _write_json(config_path, {
                "type": "ensemble",
                "strategy": strategy,
                "threshold": threshold,
                "n_members": len(member_run_ids),
                "member_run_ids": member_run_ids,
            })

        # 3. Metrics -----------------------------------------------------------
        from monai.data.wsi_reader import WSIReader
        from segmenteer.core.base import WSI_READER

        reader = WSIReader(WSI_READER)
        wsi = reader.read(str(image_path))
        w, h = reader.get_size(wsi, 0)
        area = w * h

        unsupervised = compute_unsupervised_metrics(geojson, area)
        supervised = (
            compute_all_supervised_metrics(geojson, ground_truth_geojson, (w, h))
            if ground_truth_geojson is not None
            else None
        )

        eval_data: dict = {
            "method_name": "ensemble",
            "run_id": run_id,
            "image_stem": image_stem,
            "image_path": str(image_path),
            "strategy": strategy,
            "threshold": threshold,
            "member_run_ids": member_run_ids,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {"unsupervised": asdict(unsupervised)},
        }
        if supervised is not None:
            eval_data["metrics"]["supervised"] = asdict(supervised)
        _write_json(scores_dir / f"{image_stem}.json", eval_data)

        # 4. Heatmap -----------------------------------------------------------
        save_heatmap_thumbnail(
            image=image_path,
            geojson_data=geojson,
            output_path=heatmaps_dir / f"{image_stem}.png",
            mpp=self.heatmap_mpp,
            max_size=self.heatmap_max_size,
            alpha=self.heatmap_alpha,
        )

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

    def finalize_dataset(
        self,
        all_results: dict[Path, list[BenchmarkResult]] | None = None,
    ) -> Path:
        """Write ``ensemble_manifest.json`` for a dataset run.

        The manifest is **method-first**: each top-level ``members`` entry
        represents one algorithm and lists its per-image predictions.
        This mirrors the on-disk layout ``<run_id>/predictions/<stem>.geojson``
        and makes it straightforward to load one method's outputs or all
        methods for a single image via :func:`load_ensemble_members`.
        """
        members_block = []
        for run_id, image_entries in self._run_members.items():
            member_entry: dict = {
                "run_id": run_id,
                "name": self._run_names.get(run_id, run_id),
                "n_images": len(image_entries),
                "images": image_entries,
            }
            if all_results:
                # collect per-image summary rows for this run_id
                summary_rows = []
                for img_path, img_result_list in all_results.items():
                    matching = [r for r in img_result_list if r.run_id == run_id]
                    summary_rows.extend(_build_summary(matching))
                if summary_rows:
                    member_entry["summary"] = summary_rows
            members_block.append(member_entry)

        all_image_stems = sorted(
            {e["image_stem"] for imgs in self._run_members.values() for e in imgs}
        )
        manifest: dict = {
            "segmenteer_version": _segmenteer_version(),
            "mode": "dataset",
            "run_dir": str(self.output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_methods": len(self._run_members),
            "n_images": len(all_image_stems),
            "images": all_image_stems,
            "members": members_block,
        }

        manifest_path = self.output_dir / MANIFEST_FILE
        _write_json(manifest_path, manifest)
        return manifest_path

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
    rows = []
    for r in results:
        u = r.unsupervised_metrics
        row: dict = {
            "run_id": r.run_id,
            "method": r.method_name,
            "execution_time_s": r.execution_time,
            "num_objects": u.num_objects,
            "coverage_ratio": u.coverage_ratio,
            "mean_area": u.mean_area,
            "mean_compactness": u.mean_compactness,
            "mean_solidity": u.mean_solidity,
        }
        if r.supervised_metrics is not None:
            s = r.supervised_metrics
            row["dice"] = s.dice
            row["iou"] = s.iou
            row["precision"] = s.precision
            row["recall"] = s.recall
        rows.append(row)
    return rows


def _segmenteer_version() -> str:
    try:
        from importlib.metadata import version

        return version("segmenteer")
    except Exception:
        return "unknown"

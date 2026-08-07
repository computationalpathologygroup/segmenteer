"""High-level benchmark workflows used by :mod:`run.py`."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable
from typing import Optional

from segmenteer.benchmark.console import BenchmarkReporter
from segmenteer.benchmark.ensemble import EnsembleOutputWriter
from segmenteer.benchmark.locking import claim_method_locks, dataset_metadata_lock
from segmenteer.benchmark.reporting import export_results_csv, export_results_json
from segmenteer.benchmark.resume import DatasetResumeStore
from segmenteer.benchmark.runner import BenchmarkRunner, make_run_ids
from segmenteer.io import create_timestamped_output_dir, write_json_atomic
from segmenteer.io.loader import save_geojson


def _persist_ground_truths(output_dir: Path, ground_truths: dict[Path, dict] | None) -> None:
    """Persist paired GT before models run so the viewer can always render it."""
    if not ground_truths:
        return
    ground_truth_dir = output_dir / "ground_truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    for image_path, geojson in ground_truths.items():
        if geojson and geojson.get("features"):
            target = ground_truth_dir / f"{Path(image_path).stem}.geojson"
            # The parent launcher commits the shared cohort first. Child shards
            # then avoid duplicating 555 GeoJSON writes before inference begins.
            if not target.exists():
                save_geojson(geojson, target)


def run_single_image(
    segmenters: list,
    path: Path,
    ground_truth: Optional[dict] = None,
    compute_pixel_metrics: bool = False,
    save_heatmaps: bool = False,
    output_root: str | Path = "outputs",
) -> Path:
    """Benchmark segmenters on one WSI and return the created output folder."""
    path = Path(path)
    output_dir = create_timestamped_output_dir(output_root)
    if ground_truth:
        _persist_ground_truths(output_dir, {path: ground_truth})

    reporter = BenchmarkReporter()
    writer = EnsembleOutputWriter(
        output_dir, image_path=path, save_heatmaps=save_heatmaps
    )
    # ``compute_pixel_metrics`` is retained for API compatibility but metrics
    # are intentionally deferred to evaluate_outputs.py.
    runner = BenchmarkRunner(reporter=reporter, result_callback=writer)
    results = runner.run_multiple(segmenters, path, ground_truth_geojson=ground_truth)

    export_results_csv(results, output_dir / "results.csv")
    export_results_json(results, output_dir / "results.json")
    manifest = writer.finalize(results, image_path=path)
    reporter.print_summary(results)
    saved_items = ["results.csv", "results.json", manifest.name] + [f"{result.run_id}/" for result in results]
    reporter.print_saved(output_dir, saved_items)
    return output_dir


def _persist_dataset_manifest(output_dir: Path, dataset_metadata: dict | None) -> Path | None:
    """Create or validate the immutable dataset contract for an output folder."""
    if dataset_metadata is None:
        return None
    path = output_dir / "dataset_manifest.json"
    with dataset_metadata_lock(output_dir):
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Existing dataset manifest is unreadable: {path}. "
                    "Use a different output directory."
                ) from exc
            if existing != dataset_metadata:
                raise ValueError(
                    "The output directory belongs to a different dataset or run policy. "
                    f"Use a different directory instead of mixing contracts: {output_dir}"
                )
        else:
            write_json_atomic(path, dataset_metadata)
    return path


def _resolve_dataset_output_dir(
    *,
    output_root: str | Path,
    output_dir: str | Path | None,
) -> Path:
    """Return a new timestamped folder or safely create/join a named folder.

    A caller that supplies ``output_dir`` always gets create-or-join semantics.
    Valid existing slide/method artifacts are reused automatically, while
    distinct method configurations remain concurrently writable through their
    independent method locks.
    """
    if output_dir is None:
        return create_timestamped_output_dir(output_root)

    selected = Path(output_dir).expanduser()
    if selected.exists() and not selected.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {selected}")
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def prepare_dataset_output(
    *,
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
    ground_truths: dict[Path, dict] | None = None,
    dataset_metadata: dict | None = None,
) -> Path:
    """Create/join a dataset output folder and commit its immutable contract.

    This lightweight operation performs no inference.  It lets a launcher set
    up one shared output folder before several disjoint slide shards begin
    writing the same method configuration concurrently.
    """
    selected = _resolve_dataset_output_dir(
        output_root=output_root,
        output_dir=output_dir,
    )
    _persist_dataset_manifest(selected, dataset_metadata)
    _persist_ground_truths(selected, ground_truths)
    return selected


def run_dataset(
    segmenters: list,
    images: list[Path],
    ground_truths: Optional[dict[Path, dict]] = None,
    compute_pixel_metrics: bool = False,
    save_heatmaps: bool = False,
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
    dataset_metadata: dict | None = None,
    output_ready_callback: Callable[[Path], None] | None = None,
    dataset_index_refresh_interval: int = 16,
    shared_method_workers: bool = False,
) -> Path:
    """Benchmark segmenters across WSIs and return the output folder.

    By default a collision-safe timestamped folder is created beneath
    ``output_root``. Supplying ``output_dir`` always creates or joins that
    folder safely: valid slide × method artifacts are reused, while distinct
    method configurations can run concurrently in the same directory. Each
    process claims only its selected method IDs; a duplicate method/configuration
    is rejected instead of racing on its files. ``shared_method_workers=True``
    is reserved for the built-in launcher, whose workers receive disjoint slide
    shards for the same configuration. Missing, failed, partial, corrupt, or
    incompatible artifacts are recomputed automatically.
    """
    output_dir = prepare_dataset_output(
        output_root=output_root,
        output_dir=output_dir,
        ground_truths=ground_truths,
        dataset_metadata=dataset_metadata,
    )
    dataset_manifest = output_dir / "dataset_manifest.json" if dataset_metadata is not None else None
    if output_ready_callback is not None:
        output_ready_callback(output_dir)

    run_ids = make_run_ids(segmenters)
    # Claim only the method directories this process owns.  This keeps
    # inference parallel for distinct methods while protecting configuration
    # invalidation and all per-method artifacts from duplicate workers.
    with claim_method_locks(output_dir, run_ids, shared=shared_method_workers):
        reporter = BenchmarkReporter()
        writer = EnsembleOutputWriter(
            output_dir,
            save_heatmaps=save_heatmaps,
            index_refresh_interval=dataset_index_refresh_interval,
        )
        # Reuse validation is unconditional. A named output directory is
        # always safe to join, whether it was created moments ago or contains
        # prior work from an interrupted or concurrent process.
        resume_store = DatasetResumeStore(output_dir)
        # ``compute_pixel_metrics`` is retained for API compatibility but metrics
        # are intentionally deferred to evaluate_outputs.py.
        runner = BenchmarkRunner(reporter=reporter, result_callback=writer)
        all_results = runner.run_dataset(
            segmenters,
            images,
            ground_truths=ground_truths,
            resume_store=resume_store,
            run_ids=run_ids,
        )
        # This scans all committed method outputs under a short root lock, so
        # the final CSV/manifest is the complete union across active workers.
        manifest = writer.finalize_dataset(all_results, dataset_metadata=dataset_metadata)

        for image_path, results in all_results.items():
            ground_truth_available = any(
                result.ground_truth_geojson is not None for result in results
            )
            reporter.print_image_header(
                image_path,
                0,
                0,
                evaluation_mode=(
                    "ground truth available; metrics deferred"
                    if ground_truth_available
                    else "prediction-only"
                ),
            )
            reporter.print_summary(results)

        stats = resume_store.stats
        if stats.reused or stats.invalid_artifacts or stats.invalidated_methods:
            print(
                "Output reuse summary: "
                f"reused {stats.reused} completed slide/method result(s); "
                f"recomputed or repaired {stats.invalid_artifacts} invalid result(s)."
            )
        if stats.invalidated_methods:
            print(
                "Configuration changed; cleared and reran every selected output for: "
                + ", ".join(stats.invalidated_methods)
            )

    saved_items = ["results.csv", manifest.name, "evaluate_outputs.py (post-run metrics)"]
    if dataset_manifest is not None:
        saved_items.append(dataset_manifest.name)
    if ground_truths:
        saved_items.append("ground_truth/")
    reporter.print_saved(output_dir, saved_items)
    return output_dir

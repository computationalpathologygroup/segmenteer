"""Low-level inference-only workflows used by the public directory runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from segmenteer.benchmark.console import BenchmarkReporter
from segmenteer.benchmark.locking import claim_method_locks, dataset_metadata_lock
from segmenteer.benchmark.output import PredictionOutputWriter
from segmenteer.benchmark.resume import DatasetResumeStore
from segmenteer.benchmark.runner import BenchmarkRunner, make_run_ids
from segmenteer.io.atomic import write_json_atomic
from segmenteer.io.utils import create_timestamped_output_dir


def _resolve_dataset_output_dir(
    *, output_root: str | Path, output_dir: str | Path | None
) -> Path:
    if output_dir is None:
        return create_timestamped_output_dir(output_root)
    selected = Path(output_dir).expanduser()
    if selected.exists() and not selected.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {selected}")
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def _persist_dataset_manifest(output_dir: Path, images: list[Path]) -> Path:
    """Record the union of slide names discovered by cooperative runner workers."""
    discovered = {Path(image).name for image in images}
    path = output_dir / "dataset_manifest.json"
    with dataset_metadata_lock(output_dir):
        existing_names: set[str] = set()
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Existing dataset manifest is unreadable: {path}") from exc
            if isinstance(existing, dict) and isinstance(existing.get("slides"), list):
                existing_names = {
                    str(name) for name in existing["slides"] if str(name).strip()
                }
        names = sorted(existing_names | discovered, key=str.casefold)
        write_json_atomic(path, {"n_slides": len(names), "slides": names})
    return path


def prepare_dataset_output(
    *,
    images: list[Path] | None = None,
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
) -> Path:
    """Create/join an experiment directory and persist its runner manifest."""
    selected = _resolve_dataset_output_dir(
        output_root=output_root, output_dir=output_dir
    )
    if images is not None:
        _persist_dataset_manifest(selected, [Path(image) for image in images])
    return selected


def run_single_image(
    segmenters: list,
    path: Path,
    output_root: str | Path = "outputs",
) -> Path:
    """Run segmentation methods on one WSI and save prediction/config artifacts."""
    path = Path(path)
    output_dir = create_timestamped_output_dir(output_root)
    _persist_dataset_manifest(output_dir, [path])

    reporter = BenchmarkReporter()
    writer = PredictionOutputWriter(output_dir)
    runner = BenchmarkRunner(reporter=reporter, result_callback=writer)
    results = runner.run_multiple(segmenters, path)

    reporter.print_summary(results)
    reporter.print_saved(
        output_dir,
        ["dataset_manifest.json", "<method>/config.yaml", "<method>/predictions/"],
    )
    return output_dir


def run_dataset(
    segmenters: list,
    images: list[Path],
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
    output_ready_callback: Callable[[Path], None] | None = None,
    shared_method_workers: bool = False,
) -> Path:
    """Run methods across a dataset, independent of evaluator and viewer."""
    images = [Path(image) for image in images]
    output_dir = prepare_dataset_output(
        images=images, output_root=output_root, output_dir=output_dir
    )
    if output_ready_callback is not None:
        output_ready_callback(output_dir)

    run_ids = make_run_ids(segmenters)
    reporter = BenchmarkReporter()
    with claim_method_locks(output_dir, run_ids, shared=shared_method_workers):
        writer = PredictionOutputWriter(output_dir)
        resume_store = DatasetResumeStore(output_dir)
        runner = BenchmarkRunner(reporter=reporter, result_callback=writer)
        all_results = runner.run_dataset(
            segmenters,
            images,
            resume_store=resume_store,
            run_ids=run_ids,
        )

        combined_results = [
            result for image_results in all_results.values() for result in image_results
        ]
        reporter.print_summary(combined_results)

        stats = resume_store.stats
        if stats.reused or stats.invalid_artifacts or stats.invalidated_methods:
            print(
                "Output reuse summary: "
                f"reused {stats.reused} prediction(s); "
                f"recomputed/repaired {stats.invalid_artifacts} invalid prediction(s)."
            )

    reporter.print_saved(
        output_dir,
        ["dataset_manifest.json", "<method>/config.yaml", "<method>/predictions/"],
    )
    return output_dir

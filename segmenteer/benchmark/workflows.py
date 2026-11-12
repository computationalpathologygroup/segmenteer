from __future__ import annotations
import json
from collections.abc import Callable
from pathlib import Path
from segmenteer.benchmark.console import BenchmarkReporter
from segmenteer.benchmark.locking import claim_method_locks, dataset_metadata_lock
from segmenteer.benchmark.output import PredictionOutputWriter
from segmenteer.benchmark.resume import DatasetResumeStore
from segmenteer.benchmark.timing import rebuild_timing_csv
from segmenteer.benchmark.runner import BenchmarkRunner, make_run_ids
from segmenteer.benchmark.slide_identity import validate_unique_slide_stems
from segmenteer.io.atomic import write_json_atomic
from segmenteer.io.predictions import normalize_prediction_format
from segmenteer.io.utils import DEFAULT_EXPERIMENT_NAME


def _resolve_dataset_output_dir(
    *, output_root: str | Path, output_dir: str | Path | None
) -> Path:
    selected = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else Path(output_root).expanduser() / DEFAULT_EXPERIMENT_NAME
    )
    if selected.exists() and (not selected.is_dir()):
        raise NotADirectoryError(
            f"Output path exists but is not a directory: {selected}"
        )
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def _persist_dataset_manifest(
    output_dir: Path, images: list[Path], output_format: str = "geojson"
) -> Path:
    output_format = normalize_prediction_format(output_format)
    validate_unique_slide_stems(images, context="requested dataset cohort")
    discovered = {Path(image).name for image in images}
    path = output_dir / "dataset_manifest.json"
    with dataset_metadata_lock(output_dir):
        existing_names: set[str] = set()
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Existing dataset manifest is unreadable: {path}"
                ) from exc
            if isinstance(existing, dict) and isinstance(existing.get("slides"), list):
                existing_names = {
                    str(name) for name in existing["slides"] if str(name).strip()
                }
            saved_format = (
                existing.get("prediction_format", "geojson")
                if isinstance(existing, dict)
                else "geojson"
            )
            if saved_format != output_format:
                raise ValueError(
                    f"Experiment {output_dir} already uses {saved_format} predictions; select a new --output-dir to use {output_format}."
                )
        names = sorted(existing_names | discovered, key=str.casefold)
        validate_unique_slide_stems(names, context=f"experiment {output_dir}")
        write_json_atomic(
            path,
            {
                "n_slides": len(names),
                "slides": names,
                "prediction_format": output_format,
            },
        )
    return path


def prepare_dataset_output(
    *,
    images: list[Path] | None = None,
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
    output_format: str = "geojson",
) -> Path:
    if images is not None:
        validate_unique_slide_stems(images, context="requested dataset cohort")
    selected = _resolve_dataset_output_dir(
        output_root=output_root, output_dir=output_dir
    )
    if images is not None:
        _persist_dataset_manifest(
            selected, [Path(image) for image in images], output_format=output_format
        )
    return selected


def run_single_image(
    segmenters: list,
    path: Path,
    output_root: str | Path = "outputs",
    output_format: str = "geojson",
    output_dir: str | Path | None = None,
) -> Path:
    path = Path(path)
    output_dir = _resolve_dataset_output_dir(
        output_root=output_root, output_dir=output_dir
    )
    _persist_dataset_manifest(output_dir, [path], output_format=output_format)
    reporter = BenchmarkReporter()
    writer = PredictionOutputWriter(output_dir, output_format=output_format)
    runner = BenchmarkRunner(reporter=reporter, result_callback=writer)
    results = runner.run_multiple(segmenters, path)
    summaries = rebuild_timing_csv(output_dir)
    reporter.print_summary(summaries, expected_slides=1)
    saved = ["dataset_manifest.json", "<method>/config.yaml", "<method>/predictions/"]
    if any(path.is_dir() for path in output_dir.glob("*/timings")):
        saved.append("<method>/timings/")
    saved.extend(["benchmark_timings.csv", "benchmark_summary.csv"])
    if reporter is not None:
        reporter.print_saved(output_dir, saved)
    return output_dir


def run_dataset(
    segmenters: list,
    images: list[Path],
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
    output_ready_callback: Callable[[Path], None] | None = None,
    shared_method_workers: bool = False,
    output_format: str = "geojson",
) -> Path:
    images = [Path(image) for image in images]
    validate_unique_slide_stems(images, context="requested dataset cohort")
    output_dir = prepare_dataset_output(
        images=images,
        output_root=output_root,
        output_dir=output_dir,
        output_format=output_format,
    )
    if output_ready_callback is not None:
        output_ready_callback(output_dir)
    run_ids = make_run_ids(segmenters)
    reporter = None if shared_method_workers else BenchmarkReporter()
    with claim_method_locks(output_dir, run_ids, shared=shared_method_workers):
        writer = PredictionOutputWriter(output_dir, output_format=output_format)
        resume_store = DatasetResumeStore(output_dir, output_format=output_format)
        runner = BenchmarkRunner(reporter=reporter, result_callback=writer)
        all_results = runner.run_dataset(
            segmenters,
            images,
            resume_store=resume_store,
            run_ids=run_ids,
            retain_geometries=False,
        )
        if not shared_method_workers:
            summaries = rebuild_timing_csv(output_dir)
            reporter.print_summary(summaries, expected_slides=len(images))
        stats = resume_store.stats
        if not shared_method_workers and (
            stats.reused or stats.invalid_artifacts or stats.invalidated_methods
        ):
            print(
                f"Output reuse: {stats.reused} reused, {stats.invalid_artifacts} rebuilt."
            )
    saved = ["dataset_manifest.json", "<method>/config.yaml", "<method>/predictions/"]
    if any(path.is_dir() for path in output_dir.glob("*/timings")):
        saved.append("<method>/timings/")
    if not shared_method_workers:
        saved.extend(["benchmark_timings.csv", "benchmark_summary.csv"])
    if reporter is not None:
        reporter.print_saved(output_dir, saved)
    return output_dir

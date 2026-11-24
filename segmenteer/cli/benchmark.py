"""Declarative multi-method inference, reusing the benchmark runner and its resume store.

No annotation files are required. Evaluation is deliberately a separate command.
"""
from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml

from segmenteer.benchmark.runner import make_run_ids
from segmenteer.cli.commands import (
    _UniqueKeySafeLoader,
    _make_segmenter,
    _normalise_params,
    _validate_atlas_params,
    _validate_constructor,
    _workers_arg,
)
from segmenteer.cli.datasets import load_dataset
from segmenteer.cli.registry import METHODS, TRIDENT_MODEL_IDS, get_class
from segmenteer.io.utils import DEFAULT_EXPERIMENT_NAME
from segmenteer.runner import (
    SLIDE_ORDERS, _child_environment, _prepare_parallel_resume,
    _resolve_workers, _shard_images, _uses_accelerator,
)


def load_benchmark_configs(source: str | Path) -> list[tuple[str, dict]]:
    """Parse and strictly validate a list of named method configurations."""
    source = Path(source).expanduser().resolve()
    try:
        payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load benchmark YAML {source}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"methods"}:
        raise ValueError("Benchmark YAML must have exactly one top-level key: methods.")
    entries = payload["methods"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Benchmark YAML 'methods' must be a nonempty list.")
    result = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not set(entry) <= {"name", "params"} or "name" not in entry:
            raise ValueError(f"Method entry {index} must contain name and optional params only.")
        name = entry["name"]
        if not isinstance(name, str) or name not in METHODS:
            raise ValueError(f"Unknown method at entry {index}: {name!r}; see 'segmenteer list-methods'.")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Method entry {index} params must be a mapping.")
        params = _normalise_params(params)
        if name == "fesi-basic":
            if "improved" in params and params["improved"] is not False:
                raise ValueError("fesi-basic always uses improved=false.")
            params["improved"] = False
        if name == "atlaspatch-sam2":
            _validate_atlas_params(params)
        result.append((name, params))
    return result


def _parser(command_name: str = "multi-method") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"segmenteer {command_name}",
        description="Run multiple tissue-segmentation configurations without ground-truth annotations.",
    )
    parser.add_argument("--dataset", type=Path, required=True, help="WSI directory or CSV with a path column")
    parser.add_argument("--configs", type=Path, required=True, help="YAML list of methods and params (no annotations required)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--format", "--output-format", dest="output_format", choices=("xml", "geojson"), default="geojson")
    parser.add_argument("--workers", type=_workers_arg, default=1)
    parser.add_argument("--device", help="Default device for models that explicitly support it; YAML params.device wins")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--slide-order", choices=SLIDE_ORDERS, default="smallest-first")
    parser.add_argument("--wsi-reader", choices=("auto", "openslide", "tifffile"))
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and configurations without starting models")
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    return parser


def _build_methods(entries: list[tuple[str, dict]], default_device: str | None):
    segmenters = []
    for name, original_params in entries:
        params = dict(original_params)
        target = get_class(name)
        if default_device is not None and "device" not in params:
            if name in TRIDENT_MODEL_IDS or (
                isinstance(target, type) and "device" in inspect.signature(target).parameters
            ):
                params["device"] = default_device
        if name == "atlaspatch-sam2":
            _validate_atlas_params(params)
        _validate_constructor(target, params, name)
        segmenters.append(_make_segmenter(target, params, name))
    # Fail early on duplicate configurations, without modifying existing outputs.
    make_run_ids(segmenters)
    return segmenters


def main(argv: Sequence[str] | None = None, *, command_name: str = "multi-method") -> None:
    parser = _parser(command_name)
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except (ValueError, FileNotFoundError, NotADirectoryError, ImportError, TypeError) as exc:
        parser.error(str(exc))


def _execute(args: argparse.Namespace) -> None:
    from segmenteer.benchmark.workflows import prepare_dataset_output, run_dataset
    from segmenteer.benchmark.console import BenchmarkReporter
    from segmenteer.benchmark.timing import rebuild_timing_csv

    images = load_dataset(args.dataset, path_column=args.path_column, slide_order=args.slide_order)
    entries = load_benchmark_configs(args.configs)
    output_dir = args.output_dir.expanduser().resolve()
    if args.wsi_reader:
        os.environ["WSI_READER"] = args.wsi_reader
    if (args.shard_index is None) != (args.shard_count is None):
        raise ValueError("Internal sharding arguments must be provided together.")
    if args.shard_index is not None and (
        args.shard_count < 2 or args.workers != 1 or args.dry_run
    ):
        raise ValueError("Invalid internal benchmark shard configuration.")
    # Config/YAML and cohort validation before inference; dry-run must not
    # instantiate weights or generate any output artifacts.
    if args.dry_run:
        for index, (name, params) in enumerate(entries, 1):
            target = get_class(name)
            if args.device and "device" not in params and (
                name in TRIDENT_MODEL_IDS or (
                    isinstance(target, type) and "device" in inspect.signature(target).parameters
                )
            ):
                params = {**params, "device": args.device}
            _validate_constructor(target, params, name)
            print(f"{index:02d}. {name}: {yaml.safe_dump(params, default_flow_style=True).strip()}")
        print(f"Dataset: {len(images)} slides | {len(entries)} configurations | output={output_dir}")
        return

    segmenters = _build_methods(entries, args.device)
    if args.shard_index is not None:
        images = _shard_images(images, shard_index=args.shard_index, shard_count=args.shard_count)
        run_dataset(segmenters, images, output_dir=output_dir,
                    output_format=args.output_format, shared_method_workers=True)
        return

    worker_count = min(_resolve_workers(args.workers, segmenters), len(images))
    print(f"Multi-method inference: {len(images)} slides, {len(segmenters)} configurations, {worker_count} worker(s)")
    print(f"Experiment: {output_dir}")
    if worker_count <= 1:
        run_dataset(segmenters, images, output_dir=output_dir, output_format=args.output_format)
        return

    # Existing cooperative runner semantics: prepare one shared experiment,
    # then shard the cohort while retaining the same method IDs and resume data.
    prepare_dataset_output(images=images, output_dir=output_dir,
                           output_format=args.output_format)
    _prepare_parallel_resume(segmenters, output_dir, args.output_format)
    processes = []
    try:
        for index in range(worker_count):
            command = [
                sys.executable, "-m", "segmenteer.cli", "multi-method",
                "--dataset", str(args.dataset.resolve()),
                "--configs", str(args.configs.resolve()),
                "--output-dir", str(output_dir),
                "--format", args.output_format,
                "--workers", "1",
                "--path-column", args.path_column,
                "--slide-order", args.slide_order,
                "--shard-index", str(index), "--shard-count", str(worker_count),
            ]
            if args.device is not None:
                command.extend(("--device", args.device))
            if args.wsi_reader is not None:
                command.extend(("--wsi-reader", args.wsi_reader))
            env = _child_environment(cpu_parallel=not _uses_accelerator(segmenters))
            env["SEGMENTEER_WORKER_LABEL"] = f"[worker {index + 1}/{worker_count}]"
            processes.append(subprocess.Popen(command, env=env))
        codes = [proc.wait() for proc in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"Benchmark workers exited with nonzero codes: {codes}.")
    except BaseException:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            proc.wait()
        raise
    summaries = rebuild_timing_csv(output_dir)
    BenchmarkReporter().print_summary(summaries, expected_slides=len(images))


if __name__ == "__main__":
    main()

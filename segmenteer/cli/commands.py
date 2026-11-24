"""Method-first CLI: `segmenteer atlaspatch-sam2 --dataset slides.csv`.

The legacy CLI and run.py remain independent. All predictions, run IDs,
config.yaml files, timing information and resume semantics are produced by
Segmenteer's existing benchmark implementation.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from segmenteer.cli.datasets import load_dataset
from segmenteer.cli.registry import (
    METHODS, TRIDENT_MODEL_IDS, get_class, name_from_class_path,
)
from segmenteer.io.utils import DEFAULT_EXPERIMENT_NAME
from segmenteer.runner import SLIDE_ORDERS, _resolve_workers, _shard_images, _uses_accelerator

_RUN_CONFIG_KEYS = frozenset({
    "method", "class", "params", "segmenteer_version", "dataset", "path_column",
    "output_root", "output_name", "output_dir", "output_format", "workers",
    "slide_order", "wsi_reader",
})


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Use safe YAML semantics and reject accidental duplicate mapping keys."""


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            exists = key in result
        except TypeError as exc:
            raise ValueError(f"Unhashable YAML key: {key!r}") from exc
        if exists:
            raise ValueError(f"Repeated YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read YAML configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YAML configuration must be a mapping: {path}")
    unexpected = set(payload) - _RUN_CONFIG_KEYS
    if unexpected:
        raise ValueError(f"Unsupported YAML configuration keys: {', '.join(sorted(map(str, unexpected)))}")
    if not isinstance(payload.get("params", {}), dict):
        raise ValueError("YAML 'params' must be a mapping of constructor keyword arguments.")
    return payload


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in values:
        key, sep, raw = item.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or not key or not key.isidentifier() or key.startswith("_"):
            raise ValueError(f"Invalid --set {item!r}; expected --set PARAM=VALUE.")
        if key in params:
            raise ValueError(f"Repeated --set for {key!r}.")
        try:
            params[key] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid --set value for {key!r}: {exc}") from exc
    return params


def _normalise_params(params: dict) -> dict[str, Any]:
    result = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key or key.startswith("_"):
            raise ValueError(f"Invalid YAML parameter key: {key!r}")
        normalised = key.replace("-", "_")
        if not normalised.isidentifier() or normalised in result:
            raise ValueError(f"Invalid or duplicate parameter key: {key!r}")
        result[normalised] = value
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segmenteer",
        description="Run a selected Segmenteer method with a CSV or directory cohort.",
    )
    parser.add_argument(
        "method", choices=[*sorted(METHODS), "run", "list-methods"],
        help="Method to run; 'run' selects the method from --config.",
    )
    parser.add_argument("--dataset", type=Path, help="WSI directory or CSV with a path column")
    parser.add_argument("--config", type=Path, help="Run YAML or previously saved method config.yaml")
    parser.add_argument("--path-column", default=None, help="CSV path column (default: path)")
    parser.add_argument("--output-dir", type=Path, help="Experiment directory (contains method run-ID subfolders)")
    parser.add_argument("--output-root", type=Path, help="Root for experiments (default: outputs)")
    parser.add_argument("--output-name", help=f"Experiment name (default: {DEFAULT_EXPERIMENT_NAME})")
    parser.add_argument("--format", "--output-format", dest="output_format", choices=("geojson", "xml"), default=None)
    parser.add_argument("--workers", type=_workers_arg, default=None, help="Worker processes (default: 1, or 'auto')")
    parser.add_argument("--slide-order", choices=SLIDE_ORDERS, help="Ordering for directory datasets; CSV preserves row order")
    parser.add_argument("--wsi-reader", choices=("auto", "openslide", "tifffile"), help="WSI reading backend")
    parser.add_argument("--device", help="Override model device, e.g. cuda:0 or cpu (also works for TRIDENT)")
    parser.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", help="Override model checkpoint_path")
    parser.add_argument("--mask-threshold", type=float, default=None, help="AtlasPatch-SAM2 probability threshold")
    parser.add_argument("--min-area", type=int, default=None, help="Override min_area, when supported")
    parser.add_argument("--mpp", type=float, default=None, help="Override target micrometres per pixel, when supported")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE", help="Method constructor override; repeatable. E.g. --set min_area=25")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and config; do not initialize model or write outputs")
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    return parser


def _workers_arg(value: str) -> int | str:
    if value.casefold() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer or 'auto'.") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer or 'auto'.")
    return n


def _selected(cli_value: Any, yaml_doc: dict, yaml_key: str, default: Any):
    return cli_value if cli_value is not None else yaml_doc.get(yaml_key, default)


def _resolve_config_method(requested: str, cfg: dict) -> str:
    from_method = cfg.get("method")
    from_class = cfg.get("class")
    if from_method is not None and (not isinstance(from_method, str) or from_method not in METHODS):
        raise ValueError(f"Unknown method in YAML: {from_method!r}.")
    from_class_name = name_from_class_path(from_class, cfg.get("params") or {}) if from_class is not None else None
    if from_method is not None and from_class_name is not None and from_method != from_class_name:
        raise ValueError(f"YAML method {from_method!r} does not match class {from_class!r}.")
    recorded = from_method or from_class_name
    if requested == "run":
        if recorded is None:
            raise ValueError("'segmenteer run' requires YAML 'method' or saved 'class'.")
        return recorded
    if recorded is not None and requested != recorded:
        raise ValueError(f"Requested method {requested!r} differs from YAML method {recorded!r}.")
    return requested


def _validate_constructor(target: object, params: dict[str, Any], method: str | None = None) -> None:
    if method in TRIDENT_MODEL_IDS:
        # TRIDENT exports are configured instances, not constructors. Their
        # `segmenter` member is a lazy factory and must never be replaced by
        # user YAML. Only public TRIDENT settings and a CLI-specific device
        # override are supported.
        allowed = {"model_id", "target_mag", "holes_are_tissue", "batch_size", "num_workers", "device"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"Invalid TRIDENT parameters: {', '.join(sorted(unknown))}. "
                "Use target_mag, holes_are_tissue, batch_size, num_workers or device."
            )
        if "model_id" in params and (
                not isinstance(params["model_id"], str)
                or params["model_id"].casefold() != TRIDENT_MODEL_IDS[method].casefold()):
            raise ValueError(
                f"TRIDENT method {method!r} requires model_id={TRIDENT_MODEL_IDS[method]!r}."
            )
        if "target_mag" in params and params["target_mag"] is not None and (
                isinstance(params["target_mag"], bool)
                or not isinstance(params["target_mag"], int) or params["target_mag"] < 1):
            raise ValueError("TRIDENT target_mag must be a positive integer.")
        if "holes_are_tissue" in params and not isinstance(params["holes_are_tissue"], bool):
            raise ValueError("TRIDENT holes_are_tissue must be a boolean.")
        if "batch_size" in params and (isinstance(params["batch_size"], bool)
                or not isinstance(params["batch_size"], int) or params["batch_size"] < 1):
            raise ValueError("TRIDENT batch_size must be a positive integer.")
        if "num_workers" in params and (isinstance(params["num_workers"], bool)
                or not isinstance(params["num_workers"], int) or params["num_workers"] < 0):
            raise ValueError("TRIDENT num_workers must be a nonnegative integer.")
        if "device" in params and (not isinstance(params["device"], str)
                or not params["device"].strip()):
            raise ValueError("TRIDENT device must be a nonempty string.")
        return
    if not isinstance(target, type):
        raise ValueError(f"Registered method {method!r} is not a class.")
    sig = inspect.signature(target)
    try:
        sig.bind(**params)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {target.__name__}: {exc}") from exc
    # Some Segmenteer constructors accept **kwargs; they validate downstream.


def _make_segmenter(target: object, params: dict[str, Any], method: str):
    if method in TRIDENT_MODEL_IDS:
        # Do not mutate the module's cached singleton; subsequent calls must
        # start with pristine options. The deferred model stays uninitialized.
        segmenter = copy.copy(target)
        for key, value in params.items():
            if key == "model_id":
                value = TRIDENT_MODEL_IDS[method]
            setattr(segmenter, key, value)
        return segmenter
    return target(**params)


def _constructor_defaults(target: object, method: str) -> dict[str, Any]:
    if method in TRIDENT_MODEL_IDS:
        return {
            key: value for key, value in vars(target).items()
            if key != "segmenter" and not key.startswith("_")
            and isinstance(value, (str, int, float, bool, type(None)))
        }
    return {
        key: p.default for key, p in inspect.signature(target).parameters.items()
        if p.default is not inspect.Parameter.empty
        and isinstance(p.default, (str, int, float, bool, type(None)))
    }


def _resolve_params(args: argparse.Namespace, cfg: dict, method: str | None = None) -> dict[str, Any]:
    params = _normalise_params(cfg.get("params") or {})
    flag_params = {
        "device": args.device,
        "checkpoint_path": args.checkpoint_path,
        "mask_threshold": args.mask_threshold,
        "min_area": args.min_area,
        "mpp": args.mpp,
    }
    explicit = {name: value for name, value in flag_params.items() if value is not None}
    extra = _parse_set(args.set)
    duplicate = explicit.keys() & extra.keys()
    if duplicate:
        raise ValueError(f"Conflicting CLI overrides for: {', '.join(sorted(duplicate))}. Use one form per key.")
    result = {**params, **explicit, **extra}
    if method == "fesi-basic":
        if result.get("improved") is not None and result["improved"] is not False:
            raise ValueError("fesi-basic always uses improved=false; choose 'fesi' otherwise.")
        result["improved"] = False
    return result


def _validate_atlas_params(params: dict[str, Any]) -> None:
    if "device" in params and params["device"] is not None and not isinstance(params["device"], str):
        raise ValueError("AtlasPatch-SAM2 device must be a string or null.")
    if "checkpoint_path" in params and params["checkpoint_path"] is not None and not isinstance(params["checkpoint_path"], str):
        raise ValueError("AtlasPatch-SAM2 checkpoint_path must be a path string or null.")
    if "mask_threshold" in params:
        val = params["mask_threshold"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 <= val <= 1:
            raise ValueError("AtlasPatch-SAM2 mask_threshold must be a finite number between 0 and 1.")
    if "min_area" in params:
        val = params["min_area"]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError("AtlasPatch-SAM2 min_area must be a nonnegative integer.")


def _resolve_output(args: argparse.Namespace, cfg: dict, cfg_path: Path | None = None) -> Path:
    directory = _selected(args.output_dir, cfg, "output_dir", None)
    name = _selected(args.output_name, cfg, "output_name", DEFAULT_EXPERIMENT_NAME)
    root = _selected(args.output_root, cfg, "output_root", Path("outputs"))
    if args.output_dir is not None and args.output_name is not None:
        raise ValueError("--output-dir and --output-name cannot be used together.")
    if directory is not None:
        path = Path(directory).expanduser()
        if args.output_dir is None and cfg_path is not None and not path.is_absolute():
            path = cfg_path.parent / path
        return path
    if (not isinstance(name, str) or not name or name.strip() != name
            or name in (".", "..") or "/" in name or "\\" in name):
        raise ValueError("output_name must be a single directory name.")
    root_path = Path(root).expanduser()
    if args.output_root is None and "output_root" in cfg and cfg_path is not None and not root_path.is_absolute():
        root_path = cfg_path.parent / root_path
    return root_path / name


def _spawn_shards(
    *, args: argparse.Namespace, images: list[Path], segmenter: object,
    output_dir: Path, output_format: str, workers: int,
    dataset: Path, config: Path | None,
) -> None:
    from segmenteer.benchmark.workflows import prepare_dataset_output
    from segmenteer.runner import _prepare_parallel_resume, _child_environment
    from segmenteer.benchmark.timing import rebuild_timing_csv
    from segmenteer.benchmark.console import BenchmarkReporter

    prepare_dataset_output(images=images, output_dir=output_dir, output_format=output_format)
    _prepare_parallel_resume([segmenter], output_dir, output_format)
    processes: list[subprocess.Popen] = []
    try:
        for index in range(workers):
            command = [
                sys.executable, "-m", "segmenteer.cli", args.method,
                "--dataset", str(dataset), "--output-dir", str(output_dir),
                "--format", output_format, "--workers", "1",
                "--shard-index", str(index), "--shard-count", str(workers),
            ]
            if config is not None:
                command.extend(("--config", str(config)))
            for option, value in (
                ("--path-column", args.path_column), ("--slide-order", args.slide_order),
                ("--wsi-reader", args.wsi_reader), ("--device", args.device),
                ("--checkpoint-path", args.checkpoint_path),
                ("--mask-threshold", args.mask_threshold), ("--min-area", args.min_area),
                ("--mpp", args.mpp),
            ):
                if value is not None:
                    command.extend((option, str(value)))
            for value in args.set:
                command.extend(("--set", value))
            env = _child_environment(cpu_parallel=not _uses_accelerator([segmenter]))
            env["SEGMENTEER_WORKER_LABEL"] = f"[worker {index + 1}/{workers}]"
            processes.append(subprocess.Popen(command, env=env))
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"Segmenteer workers failed: exit codes {codes}.")
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    summaries = rebuild_timing_csv(output_dir)
    BenchmarkReporter().print_summary(summaries, expected_slides=len(images))


def main(argv: Sequence[str] | None = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.method == "list-methods":
        for name in sorted(METHODS):
            print(name)
        return
    try:
        _execute(args)
    except (ValueError, FileNotFoundError, NotADirectoryError, ImportError, TypeError) as exc:
        parser.error(str(exc))


def _execute(args: argparse.Namespace) -> None:
    cfg_path = args.config.expanduser().resolve() if args.config is not None else None
    cfg = _read_yaml(cfg_path) if cfg_path is not None else {}
    method = _resolve_config_method(args.method, cfg)
    raw_dataset = _selected(args.dataset, cfg, "dataset", None)
    if raw_dataset is None:
        raise ValueError("--dataset is required (or set 'dataset' in the YAML run config).")
    dataset = Path(raw_dataset).expanduser()
    if args.dataset is None and cfg_path is not None and not dataset.is_absolute():
        dataset = cfg_path.parent / dataset
    dataset = dataset.resolve()
    order = _selected(args.slide_order, cfg, "slide_order", "smallest-first")
    if order not in SLIDE_ORDERS:
        raise ValueError(f"Invalid slide_order {order!r}; expected {SLIDE_ORDERS}.")
    path_column = _selected(args.path_column, cfg, "path_column", "path")
    if not isinstance(path_column, str) or not path_column.strip():
        raise ValueError("path_column must be a nonempty string.")
    images = load_dataset(dataset, path_column=path_column, slide_order=order)
    output_dir = _resolve_output(args, cfg, cfg_path).resolve()
    output_format = _selected(args.output_format, cfg, "output_format", "geojson")
    if output_format not in ("geojson", "xml"):
        raise ValueError("output_format must be 'geojson' or 'xml'.")
    wsi_reader = _selected(args.wsi_reader, cfg, "wsi_reader", None)
    if wsi_reader is not None:
        if wsi_reader not in ("auto", "openslide", "tifffile"):
            raise ValueError("wsi_reader must be auto, openslide or tifffile.")
        os.environ["WSI_READER"] = wsi_reader
    params = _resolve_params(args, cfg, method)
    if method == "atlaspatch-sam2":
        _validate_atlas_params(params)
    target = get_class(method)
    _validate_constructor(target, params, method)
    requested_workers = _selected(args.workers, cfg, "workers", 1)
    if isinstance(requested_workers, bool):
        raise ValueError("workers must be a positive integer or 'auto'.")
    try:
        requested_workers = _workers_arg(str(requested_workers))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    # Validate without constructing models or writing output in --dry-run mode.
    # Auto with dry-run cannot know whether the resolved method uses an accelerator;
    # report requested worker count rather than guessing.
    if args.dry_run:
        import segmenteer

        # Defaults shown here are constructor defaults, not runtime-resolved
        # values (e.g. auto device selection happens on initialization).
        defaults = _constructor_defaults(target, method)
        target_class = target if isinstance(target, type) else type(target)
        print(yaml.safe_dump({
            "method": method,
            "class": f"{target_class.__module__}.{target_class.__qualname__}",
            "params": {**defaults, **params},
            "segmenteer_version": segmenteer.__version__,
            "dataset": str(dataset),
            "slides": len(images),
            "output_dir": str(output_dir),
            "output_format": output_format,
            "workers": requested_workers,
        }, sort_keys=False))
        return
    segmenter = _make_segmenter(target, params, method)
    worker_count = min(_resolve_workers(requested_workers, [segmenter]), len(images))
    if (args.shard_index is None) != (args.shard_count is None):
        raise ValueError("Internal shard arguments must be provided together.")
    print(f"Method: {method} ({segmenter.name})")
    print(f"Dataset: {dataset} ({len(images)} slides)")
    print(f"Experiment: {output_dir}")
    if args.shard_index is not None:
        if args.shard_count is None or args.shard_count < 2 or worker_count != 1:
            raise ValueError("Invalid internal shard configuration.")
        images = _shard_images(images, shard_index=args.shard_index, shard_count=args.shard_count)
        from segmenteer.benchmark.workflows import run_dataset
        run_dataset([segmenter], images, output_dir=output_dir,
                    output_format=output_format, shared_method_workers=True)
    elif worker_count > 1:
        _spawn_shards(args=args, images=images, segmenter=segmenter,
                      output_dir=output_dir, output_format=output_format,
                      workers=worker_count, dataset=dataset, config=cfg_path)
    else:
        from segmenteer.benchmark.workflows import run_dataset
        run_dataset([segmenter], images, output_dir=output_dir, output_format=output_format)
    if args.shard_index is None:
        print(f"Predictions written to: {output_dir}")

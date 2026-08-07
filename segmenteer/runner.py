"""Public high-level RUNNER API for directory-based WSI inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SUPPORTED_WSI_SUFFIXES = (
    ".tif",
    ".tiff",
    ".svs",
    ".ndpi",
    ".qptiff",
    ".scn",
    ".mrxs",
)
SLIDE_ORDERS = ("alphabetical", "smallest-first")


def _normalise_slide_order(order: str) -> str:
    selected = str(order).strip().casefold()
    if selected not in SLIDE_ORDERS:
        raise ValueError(f"slide_order must be one of: {', '.join(SLIDE_ORDERS)}")
    return selected


def discover_slides(
    wsi_dir: str | Path,
    *,
    slide_order: str = "smallest-first",
) -> list[Path]:
    """Discover supported top-level WSI files in a deterministic order."""
    root = Path(wsi_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"WSI directory does not exist: {root}")

    paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_WSI_SUFFIXES
    ]
    selected_order = _normalise_slide_order(slide_order)
    if selected_order == "smallest-first":
        return sorted(
            paths,
            key=lambda path: (path.stat().st_size, path.name.casefold(), path.name),
        )
    return sorted(paths, key=lambda path: (path.name.casefold(), path.name))


def _resolve_relative(path: str | Path | None, *, base_dir: Path) -> Path | None:
    if path is None:
        return None
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else base_dir / selected


def _read_manifest(output_dir: Path | None) -> dict | None:
    if output_dir is None:
        return None
    path = output_dir / "dataset_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Existing dataset manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Existing dataset manifest is not a JSON object: {path}")
    return payload


def _manifest_slide_names(manifest: dict, *, path: Path) -> list[str]:
    raw_names = manifest.get("slides")
    if not isinstance(raw_names, list) or not all(
        isinstance(item, str) for item in raw_names
    ):
        raise RuntimeError(
            f"Expected Segmenteer 0.1.0 manifest with a 'slides' list: {path}"
        )

    names = [name.strip() for name in raw_names]
    if not names or any(not name for name in names):
        raise RuntimeError(f"Dataset manifest contains no valid slide names: {path}")
    if any(Path(name).name != name for name in names):
        raise RuntimeError(f"Dataset manifest must contain slide names only: {path}")
    if len(names) != len(set(names)):
        raise RuntimeError(f"Dataset manifest contains duplicate slide names: {path}")

    declared = manifest.get("n_slides")
    if declared is not None and declared != len(names):
        raise RuntimeError(
            f"dataset_manifest.json declares n_slides={declared!r}, "
            f"but contains {len(names)} slide names."
        )
    return names


def _saved_cohort(
    *,
    output_dir: Path,
    manifest: dict,
    wsi_dir: Path,
    slide_order: str,
) -> list[Path]:
    names = _manifest_slide_names(
        manifest, path=output_dir / "dataset_manifest.json"
    )
    discovered = discover_slides(wsi_dir, slide_order=slide_order)
    by_name = {path.name: path for path in discovered}
    missing = [name for name in names if name not in by_name]
    if missing:
        rendered = "\n".join(f"  - {name}" for name in missing[:20])
        suffix = "\n  - ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            "The saved runner cohort cannot be resumed because source WSI files "
            f"are missing from {wsi_dir}:\n{rendered}{suffix}"
        )

    selected = set(names)
    return [path for path in discovered if path.name in selected]


def _uses_accelerator(methods: Sequence[object]) -> bool:
    from segmenteer.core.runtime import resolve_torch_device

    for method in methods:
        device = str(getattr(method, "device", "")).strip().casefold()
        if device == "mps" or device.startswith("cuda"):
            return True
        if method.__class__.__name__ == "TRIDENTSegmenter":
            trident = os.environ.get("SEGMENTEER_TRIDENT_DEVICE", "auto")
            resolved = resolve_torch_device(trident)
            if resolved == "mps" or resolved.startswith("cuda"):
                return True
    return False


def _resolve_workers(requested: int | str, methods: Sequence[object]) -> int:
    value: int | str = requested
    if isinstance(value, str):
        normalised = value.strip().casefold()
        if normalised == "auto":
            if _uses_accelerator(methods):
                return 1
            return max(1, min(12, os.cpu_count() or 1))
        try:
            value = int(normalised)
        except ValueError as exc:
            raise ValueError("workers must be 'auto' or a positive integer.") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("workers must be 'auto' or a positive integer.")
    return value


def _shard_images(
    images: list[Path], *, shard_index: int, shard_count: int
) -> list[Path]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid slide shard index/count.")
    return images[shard_index::shard_count]


def _child_environment(*, cpu_parallel: bool) -> dict[str, str]:
    env = os.environ.copy()
    if cpu_parallel:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env[name] = "1"
    return env


def _launch_shards(
    *,
    script_path: Path,
    output_dir: Path,
    worker_count: int,
    slide_order: str,
    cpu_parallel: bool,
) -> None:
    processes: list[subprocess.Popen] = []
    print(f"Launching {worker_count} cooperative slide worker(s) into {output_dir}")
    for shard_index in range(worker_count):
        command = [
            sys.executable,
            str(script_path),
            "--output-dir",
            str(output_dir),
            "--slide-order",
            slide_order,
            "--workers",
            "1",
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(worker_count),
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=script_path.parent,
                env=_child_environment(cpu_parallel=cpu_parallel),
            )
        )

    exit_codes = [process.wait() for process in processes]
    failed = [str(index) for index, code in enumerate(exit_codes) if code != 0]
    if failed:
        raise RuntimeError(
            "One or more slide workers failed (worker indexes: "
            + ", ".join(failed)
            + ")."
        )


def _prepare_parallel_resume(methods: Sequence[object], output_dir: Path) -> None:
    """Validate configs once under exclusive locks before shared workers start."""
    from segmenteer.benchmark.locking import claim_method_locks
    from segmenteer.benchmark.resume import DatasetResumeStore
    from segmenteer.benchmark.runner import make_run_ids
    from segmenteer.core.base import segmenter_config_dict

    run_ids = make_run_ids(list(methods))
    configs = {
        run_id: segmenter_config_dict(method)
        for method, run_id in zip(methods, run_ids)
    }
    with claim_method_locks(output_dir, run_ids, shared=False):
        DatasetResumeStore(output_dir).prepare(configs)


def run_directory(
    methods: Sequence[object],
    *,
    wsi_dir: str | Path,
    output_root: str | Path = "outputs",
    output_dir: str | Path | None = None,
    slide_order: str = "smallest-first",
    workers: int | str = "auto",
    project_root: str | Path | None = None,
    worker_script: str | Path | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Path:
    """Run selected methods over a WSI directory.

    This is the high-level RUNNER API. It owns slide discovery, saved-cohort
    resume, process sharding, and output setup; the supplied methods own only
    segmentation behavior/configuration.
    """
    if not methods:
        raise ValueError("methods is empty. Select at least one segmentation method.")
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be used together.")

    base = Path(project_root).expanduser() if project_root else Path.cwd()
    source_dir = _resolve_relative(wsi_dir, base_dir=base)
    assert source_dir is not None
    root = _resolve_relative(output_root, base_dir=base)
    assert root is not None
    selected_output = _resolve_relative(output_dir, base_dir=base)
    selected_order = _normalise_slide_order(slide_order)

    existing_manifest = _read_manifest(selected_output)
    if existing_manifest is None:
        images = discover_slides(source_dir, slide_order=selected_order)
        if not images:
            raise FileNotFoundError(f"No supported WSI files found in: {source_dir}")
        resumed = False
    else:
        assert selected_output is not None
        images = _saved_cohort(
            output_dir=selected_output,
            manifest=existing_manifest,
            wsi_dir=source_dir,
            slide_order=selected_order,
        )
        resumed = True

    is_child = shard_index is not None
    if is_child:
        assert shard_index is not None and shard_count is not None
        if shard_count < 2:
            raise ValueError("shard_count must be at least 2 for a cooperative worker.")
        shard = _shard_images(
            images, shard_index=shard_index, shard_count=shard_count
        )
        if selected_output is None:
            raise RuntimeError("A cooperative worker requires a prepared output directory.")
        print(
            f"Slide shard {shard_index + 1}/{shard_count}: "
            f"{len(shard)} of {len(images)} slide(s)"
        )
        from segmenteer.benchmark.workflows import run_dataset

        return run_dataset(
            list(methods),
            shard,
            output_root=root,
            output_dir=selected_output,
            shared_method_workers=True,
        )

    cohort_source = "dataset_manifest.json" if resumed else str(source_dir)
    print(f"Dataset cohort: {cohort_source}")
    print(f"Slides: {len(images)}")
    print(f"Methods: {len(methods)}")
    print(f"Slide order: {selected_order}")

    worker_count = min(_resolve_workers(workers, methods), len(images))
    if worker_count > 1:
        if worker_script is None:
            raise ValueError(
                "workers > 1 requires worker_script. Use run_directory_cli() from a launcher."
            )
        from segmenteer.benchmark.workflows import prepare_dataset_output

        prepared = prepare_dataset_output(
            images=images,
            output_root=root,
            output_dir=selected_output,
        )
        _prepare_parallel_resume(methods, prepared)
        _launch_shards(
            script_path=Path(worker_script).resolve(),
            output_dir=prepared,
            worker_count=worker_count,
            slide_order=selected_order,
            cpu_parallel=not _uses_accelerator(methods),
        )
        return prepared

    from segmenteer.benchmark.workflows import run_dataset

    return run_dataset(
        list(methods),
        images,
        output_root=root,
        output_dir=selected_output,
    )


def run_directory_cli(
    methods: Sequence[object],
    *,
    wsi_dir: str | Path,
    output_root: str | Path = "outputs",
    default_output_dir: str | Path | None = None,
    slide_order: str = "smallest-first",
    workers: int | str = "auto",
    project_root: str | Path | None = None,
    argv: Sequence[str] | None = None,
) -> Path:
    """Small CLI wrapper for :func:`run_directory` used by project ``run.py``."""
    parser = argparse.ArgumentParser(
        description="Run Segmenteer inference over a WSI directory."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="create or join this experiment directory",
    )
    mode.add_argument(
        "--new",
        action="store_true",
        help="ignore DEFAULT_OUTPUT_DIR and create a new timestamped experiment",
    )
    parser.add_argument("--slide-order", choices=SLIDE_ORDERS)
    parser.add_argument("--workers", type=int, metavar="N")
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    selected_output = (
        args.output_dir
        if args.output_dir is not None
        else (None if args.new else default_output_dir)
    )
    selected_order = args.slide_order or slide_order
    selected_workers: int | str = args.workers if args.workers is not None else workers

    output = run_directory(
        methods,
        wsi_dir=wsi_dir,
        output_root=output_root,
        output_dir=selected_output,
        slide_order=selected_order,
        workers=selected_workers,
        project_root=project_root,
        worker_script=Path(sys.argv[0]).resolve(),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    if args.shard_index is None:
        print(f"\nPredictions written to: {output}\n")
    return output

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# Shared model-weight location.  Existing project-local weights are used before
# any downloader/cache is consulted.  Override it before launch when models
# live on another mounted volume, for example:
#   SEGMENTEER_MODELS_DIR=/Volumes/models uv run python run.py
MODELS_DIR = Path(
    os.environ.get("SEGMENTEER_MODELS_DIR", str(PROJECT_ROOT / "models"))
).expanduser()
os.environ["SEGMENTEER_MODELS_DIR"] = str(MODELS_DIR)

# Prefer OpenSlide when it can read the slide and fall back to tifffile only
# when necessary. This avoids tifffile's fragile full-pyramid decode path for
# regular WSI files while retaining support for plain TIFFs.
WSI_READER = "auto"
os.environ["WSI_READER"] = WSI_READER

# Set this only when slide metadata has no reliable level-0 MPP value.
# Typical values are 0.25 for a 40x scan and 0.50 for a 20x scan.
WSI_NATIVE_MPP: float | None = None
if WSI_NATIVE_MPP is None:
    os.environ.pop("WSI_NATIVE_MPP", None)
else:
    os.environ["WSI_NATIVE_MPP"] = str(WSI_NATIVE_MPP)


def _first_mounted_path(*paths: str) -> Path:
    """Choose the first mounted source path, retaining a useful default.

    The project is used both on compute hosts (``/data/...``) and on macOS
    where the same archive is often mounted under ``/Volumes/PA_CPGARCHIVE``.
    """
    candidates = [Path(path).expanduser() for path in paths]
    return next((path for path in candidates if path.is_dir()), candidates[0])


# WSI TIFFs and ASAP XML ground truth are co-located in this archive.
# Pairing is exact by filename stem: <slide>.tif <-> <slide>.xml.
# Keep both paths identical unless the dataset layout is intentionally changed.
WSI_DIR = _first_mounted_path(
    "/Volumes/PA_CPGARCHIVE/projects/tissue-segmentation/data/annotation/slides",
    "/data/pa_cpgarchive/projects/tissue-segmentation/data/annotation/slides",
)
GROUND_TRUTH_DIR = WSI_DIR
ANNOTATION_SUFFIX = ".xml"  # change to ".geojson" for GeoJSON labels
ASAP_GROUPS: set[str] | None = None  # None = include all ASAP annotation groups

# Partial ground truth is normal. Choose exactly one policy:
#   "skip"         run only slides with valid GT (default; comparable benchmark)
#   "unsupervised" run every slide; paired slides retain ground truth for later evaluation
#   "strict"       stop before running unless every slide has valid GT
#
# "unsupervised" does not train a method. It writes prediction GeoJSON for every
# WSI; paired ground truth is retained for the optional post-run evaluator.
UNPAIRED_SLIDE_MODE = "skip"

# Dataset processing sequence. "alphabetical" is deterministic by filename;
# "smallest-first" schedules WSI files from the lowest on-disk byte size to
# the highest. The CLI can override this: --slide-order smallest-first.
SLIDE_ORDER = "smallest-first"

# Output is always created below this project directory. Existing model files,
# environments, Git metadata, and prior output folders are never touched.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Leave this as ``None`` for the normal timestamped-output behaviour.
# Set a path only when every normal launch should create or join one named
# experiment folder. Reuse and multi-process safety are automatic for any
# selected output directory; there is no separate resume or shared-output mode.
# The CLI is usually clearer and never needs an edit here:
#   uv run python run.py --output-dir outputs/my-experiment
DEFAULT_OUTPUT_DIR: Path | None = None

# The benchmark never writes slide-thumbnail images. The live viewer streams
# source WSI tiles on demand, avoiding duplicate image artifacts and I/O.
# Keep benchmark inference lightweight. Per-method prediction heatmaps require
# an extra WSI read, so leave them off unless a run specifically needs them.
SAVE_PREDICTION_HEATMAPS = False

# Throughput profile: use the best available accelerator by default. Set
# SEGMENTEER_DEVICE=cpu, mps, or cuda:0 before launch to override this choice.
# On Apple Silicon, compatible Torch methods use MPS/Metal automatically.
def _best_available_device() -> str:
    requested = os.environ.get("SEGMENTEER_DEVICE", "auto").strip().casefold()
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


BENCHMARK_DEVICE = _best_available_device()
TRIDENT_DEVICE = "mps" #BENCHMARK_DEVICE
os.environ["SEGMENTEER_DEVICE"] = BENCHMARK_DEVICE
os.environ["SEGMENTEER_TRIDENT_DEVICE"] = TRIDENT_DEVICE

# How many independent slide processes to use for a throughput run. ``"auto"``
# uses one process for MPS/CUDA work (multiple competing GPU contexts are usually
# slower) and up to 12 disjoint slide workers for CPU-only methods. Override
# without editing this file: ``uv run python run.py --workers 6``.
SLIDE_WORKERS: int | str = "auto"

# Root results.csv / manifest files are rebuilt from committed artifacts. Batch
# those scans for large datasets; the final index is always refreshed on exit.
DATASET_INDEX_REFRESH_INTERVAL = 16

# Import only after WSI_READER / WSI_NATIVE_MPP are configured.
import segmenteer as seg

# ---------------------------------------------------------------------------
# SEGMENTERS
# ---------------------------------------------------------------------------
# Complete supported benchmark catalog: 11 classical baselines, 6 direct
# deep-learning wrappers, and 4 Trident-backed deep-learning adapters. Enable
# one or more methods as required. Throughput runs use acceleration when the
# selected backend supports it; CPU-only runs can shard a dataset across cores.
# Install all optional backends with:
#
#   uv sync --extra all
#
# Trident adapters honour their model's own target magnification automatically:
# HEST=10x, GrandQC=1x, PathProfiler=4x, and CPG=4x. SAM3 is intentionally
# excluded from this project.
SEGMENTERS = [
    # seg.EntropyMaskerSegmenter(mpp=20),
    # seg.BackgroundSubtractorMOG2Segmenter(mpp=20),
    # seg.FESISegmenter(mpp=20),
    # seg.FESISegmenter(mpp=20, improved=False),
    # seg.FESISegmenter(mpp=10),
    # seg.FESISegmenter(mpp=10, improved=False),
    # seg.HistomicsTKSegmenter(mpp=20),
    # seg.HistomicsTKSegmenter(mpp=10),
    # seg.OtsuSegmenter(mpp=20),
    # seg.BigPictureSegmenter(mpp=8, device=BENCHMARK_DEVICE),
    # seg.BigPictureSegmenter(mpp=8, device=BENCHMARK_DEVICE, select_largest_tissue_objects=True, apply_hole_filling=False),
    # seg.BigPictureSegmenter(mpp=8, device=BENCHMARK_DEVICE),
    # seg.BigPictureSegmenter(
    #     mpp=8,
    #     device=BENCHMARK_DEVICE,
    #     dilation_disk_size=32,
    #     confidence_threshold=0.8,
    #     dilate_mask=True,
    #     apply_hole_filling=True,
    #     select_largest_tissue_objects=True,
    # ),
    # seg.WatershedSegmenter(mpp=20),
    # seg.WatershedSegmenter(mpp=10),
    # seg.WatershedSegmenter(mpp=5),
    # seg.HSVThresholdSegmenter(mpp=20),
    # seg.HSVThresholdSegmenter(mpp=10, min_area=1000),
    # seg.HSVThresholdSegmenter(mpp=5),
    # seg.OtsuSegmenter(mpp=10),
    # seg.OtsuSegmenter(mpp=5),
    # seg.OtsuTissueSegmenter(mpp=20),

    seg.EntropyMaskerSegmenter(mpp=10),
    # seg.EntropyMaskerSegmenter(mpp=5),
    # seg.FESISegmenter(mpp=5),
    # seg.FESISegmenter(mpp=5, improved=False),
    # seg.HistomicsTKSegmenter(mpp=5),

    # seg.RTLucassenSlideSegmenter(mpp=7.04, device=BENCHMARK_DEVICE),
    # seg.TRIDENTCPGSegmenter,
    # seg.TRIDENTPathProfilerSegmenter,
    # seg.TRIDENTGrandQCSegmenter,
    # seg.TRIDENTHESTSegmenter,

    # # Classical methods
    # seg.LiSegmenter(mpp=20),
    # seg.YenSegmenter(mpp=20),
    # seg.MorphologicalSegmenter(mpp=10),
    # seg.HSVThresholdSegmenter(mpp=10),
    # seg.ODGMMSlideSegmenter(mpp=10),

    # # # Direct deep-learning wrappers (all receive the shared device policy)
    # seg.FastSAMSegmenter(mpp=20, device=BENCHMARK_DEVICE),
    # seg.FastSAMSegmenter(mpp=10, device=BENCHMARK_DEVICE),
    # seg.FastSAMSegmenter(mpp=5, device=BENCHMARK_DEVICE),

    # seg.GrandQCSegmenter(mpp=10, device=BENCHMARK_DEVICE),
    # seg.HESTSegmenter(mpp=20, device=BENCHMARK_DEVICE),
    # seg.RTLucassenSlideSegmenter(mpp=7.04, device=BENCHMARK_DEVICE),
    # seg.AtlasPatchSAM2Segmenter(device="cpu"),

    # # # Trident-backed deep-learning methods
    # seg.TRIDENTHESTSegmenter,
    # seg.TRIDENTGrandQCSegmenter,
    # seg.TRIDENTPathProfilerSegmenter,
    # seg.TRIDENTCPGSegmenter,

    # seg.WatershedTissueSegmenter(
    #     mpp=20,
    #     min_distance_um=100,
    # ),
    # seg.LiTissueSegmenter(mpp=10),
    # seg.LiTissueSegmenter(mpp=5),

    # seg.YenTissueSegmenter(mpp=20),
    # seg.YenTissueSegmenter(mpp=10),
    # seg.YenTissueSegmenter(mpp=5),

    # seg.HistomicsTKTissueSegmenter(
    #     mpp=20,
    #     mask_type="simple",
    # ),
    # seg.HistomicsTKTissueSegmenter(
    #     mpp=20,
    #     mask_type="saliency",
    # ),
    # seg.HistomicsTKTissueSegmenter(
    #     mpp=10,
    #     mask_type="saliency",
    # ),
    # seg.OtsuTissueSegmenter(mpp=5),
]



# ---------------------------------------------------------------------------
# ENTRY POINT — do not normally edit below this line
# ---------------------------------------------------------------------------


def _normalise_slide_order(order: str) -> str:
    """Return a validated, canonical directory-processing order."""
    normalised = order.casefold().strip()
    allowed = {"alphabetical", "smallest-first"}
    if normalised not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"SLIDE_ORDER must be one of: {options}.")
    return normalised


def _slide_paths(directory: Path, *, order: str = SLIDE_ORDER) -> list[Path]:
    """Discover supported top-level WSI files in a deterministic order.

    ``smallest-first`` uses each file's on-disk byte size. Files of equal size
    remain ordered case-insensitively by filename, making resumed runs and
    manifests reproducible.
    """
    extensions = ("*.tif", "*.tiff", "*.svs", "*.ndpi", "*.qptiff", "*.scn", "*.mrxs")
    paths = {
        path
        for pattern in extensions
        for path in directory.glob(pattern)
        if path.is_file()
    }
    selected_order = _normalise_slide_order(order)
    if selected_order == "smallest-first":
        return sorted(
            paths,
            key=lambda path: (path.stat().st_size, path.name.casefold(), path.name),
        )
    return sorted(paths, key=lambda path: (path.name.casefold(), path.name))


def _unpaired_mode() -> str:
    mode = UNPAIRED_SLIDE_MODE.casefold().strip()
    allowed = {"skip", "unsupervised", "strict"}
    if mode not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"UNPAIRED_SLIDE_MODE must be one of: {options}.")
    return mode


def _validate_benchmark_paths(
    unpaired_mode: str,
    *,
    validate_live_source: bool = True,
) -> None:
    if not SEGMENTERS:
        raise ValueError("SEGMENTERS is empty. Enable at least one method in run.py.")
    if not validate_live_source:
        return
    if not WSI_DIR.is_dir():
        raise NotADirectoryError(
            f"WSI_DIR does not exist or is not mounted: {WSI_DIR}. "
            "On macOS, verify the archive is mounted under /Volumes/PA_CPGARCHIVE."
        )
    if not GROUND_TRUTH_DIR.is_dir() and unpaired_mode != "unsupervised":
        raise NotADirectoryError(
            f"GROUND_TRUTH_DIR does not exist or is not mounted: {GROUND_TRUTH_DIR}. "
            "Set UNPAIRED_SLIDE_MODE = 'unsupervised' to run without annotations."
        )



def _announce_live_viewer(output_dir: Path) -> None:
    """Print the standalone viewer command as soon as a run folder exists."""
    try:
        shown_output = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_output = output_dir

    command = (
        "python -m app "
        f"--output {shlex.quote(str(shown_output))} "
        f"--data {shlex.quote(str(WSI_DIR))}"
    )
    if GROUND_TRUTH_DIR.is_dir():
        command += f" --annotations {shlex.quote(str(GROUND_TRUTH_DIR))}"
    if ANNOTATION_SUFFIX.casefold() != ".xml":
        command += f" --annotation-suffix {shlex.quote(ANNOTATION_SUFFIX)}"
    for group in sorted(ASAP_GROUPS or ()):
        command += f" --asap-group {shlex.quote(group)}"

    print("\nOutput folder ready. You may launch the viewer now in a second terminal:")
    print(f"  {command}")
    print("The viewer stays available while this benchmark writes results.\n")


def _select_images_for_policy(
    images: list[Path],
    paired_images: set[Path],
    unpaired_mode: str,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (slides_to_run, supervised_slides, prediction_only_or_skipped)."""
    paired = [image for image in images if image in paired_images]
    unpaired = [image for image in images if image not in paired_images]

    if unpaired_mode == "skip":
        if not paired:
            raise FileNotFoundError(
                "No usable ground-truth pairs were found. "
                "Use UNPAIRED_SLIDE_MODE = 'unsupervised' to create prediction-only outputs."
            )
        return paired, paired, unpaired
    if unpaired_mode == "unsupervised":
        return images, paired, unpaired
    # "strict" is validated by the caller before this point.
    return images, paired, []


def _dataset_metadata(
    *,
    pairing,
    discovered_images: list[Path],
    run_images: list[Path],
    supervised_images: list[Path],
    unpaired_images: list[Path],
    unpaired_mode: str,
    slide_order: str,
) -> dict:
    """Build a portable manifest that records evaluation eligibility per WSI."""
    if unpaired_mode == "skip":
        prediction_only_images: list[Path] = []
        skipped_images = unpaired_images
    elif unpaired_mode == "unsupervised":
        prediction_only_images = unpaired_images
        skipped_images = []
    else:
        prediction_only_images = []
        skipped_images = []

    return {
        "schema_version": 2,
        "unpaired_slide_mode": unpaired_mode,
        "slide_order": slide_order,
        "wsi_directory": str(WSI_DIR),
        "annotation_directory": str(GROUND_TRUTH_DIR),
        "runtime_policy": {
            "device": BENCHMARK_DEVICE,
            "process_policy": "adaptive slide sharding",
        },
        "annotation_suffix": ANNOTATION_SUFFIX,
        "asap_groups": sorted(ASAP_GROUPS) if ASAP_GROUPS else None,
        "slides_discovered": [str(image) for image in discovered_images],
        "slides_run": [str(image) for image in run_images],
        "ground_truth_available_slides": [str(image) for image in supervised_images],
        "prediction_only_slides": [str(image) for image in prediction_only_images],
        "skipped_slides": [str(image) for image in skipped_images],
        "ground_truth_pairing": pairing.as_manifest(),
    }


def _normalise_selected_output_dir(output_dir: Path | None) -> Path | None:
    """Resolve a user-selected output path against the project root."""
    if output_dir is None:
        return None
    selected = Path(output_dir).expanduser()
    return selected if selected.is_absolute() else PROJECT_ROOT / selected


def _read_existing_dataset_manifest(output_dir: Path | None) -> dict[str, Any] | None:
    """Return a saved dataset contract when an output folder is being joined."""
    if output_dir is None:
        return None
    path = output_dir / "dataset_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing dataset manifest is unreadable: {path}. "
            "Use a new output directory after repairing or replacing the manifest."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Existing dataset manifest is not a JSON object: {path}")
    return payload


def _manifest_paths(manifest: dict[str, Any], field: str, *, manifest_path: Path) -> list[Path]:
    """Read one ordered source-path list from a dataset manifest."""
    raw_paths = manifest.get(field)
    if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
        raise RuntimeError(
            f"Dataset manifest is missing a valid '{field}' list: {manifest_path}. "
            "Create a new benchmark output directory instead."
        )
    paths = [Path(item) for item in raw_paths]
    if len(set(paths)) != len(paths):
        raise RuntimeError(
            f"Dataset manifest contains duplicate entries in '{field}': {manifest_path}."
        )
    return paths


def _saved_ground_truths(
    output_dir: Path,
    manifest: dict[str, Any],
    run_images: list[Path],
) -> dict[Path, dict]:
    """Load the immutable ground-truth copy saved with a benchmark cohort."""
    manifest_path = output_dir / "dataset_manifest.json"
    supervised_images = _manifest_paths(
        manifest,
        "ground_truth_available_slides",
        manifest_path=manifest_path,
    )
    run_set = set(run_images)
    unexpected = [image for image in supervised_images if image not in run_set]
    if unexpected:
        names = ", ".join(image.name for image in unexpected[:10])
        suffix = " ..." if len(unexpected) > 10 else ""
        raise RuntimeError(
            "Dataset manifest lists supervised slides outside its saved run cohort: "
            f"{names}{suffix}"
        )

    ground_truth_dir = output_dir / "ground_truth"
    missing = [
        ground_truth_dir / f"{image.stem}.geojson"
        for image in supervised_images
        if not (ground_truth_dir / f"{image.stem}.geojson").is_file()
    ]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = "\n  - ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            "The saved benchmark cohort is missing its persisted ground-truth GeoJSON. "
            "Do not rebuild this output from a changing live annotation directory. "
            f"Missing files:\n{rendered}{suffix}"
        )

    return {
        image: seg.load_geojson(ground_truth_dir / f"{image.stem}.geojson")
        for image in supervised_images
    }


def _saved_dataset_contract(
    output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[Path], dict[Path, dict]]:
    """Restore the exact slide cohort and GT used by an existing benchmark."""
    manifest_path = output_dir / "dataset_manifest.json"
    run_images = _manifest_paths(manifest, "slides_run", manifest_path=manifest_path)
    if not run_images:
        raise RuntimeError(f"Dataset manifest has an empty saved cohort: {manifest_path}")

    unavailable = [image for image in run_images if not image.is_file()]
    if unavailable:
        rendered = "\n".join(f"  - {image}" for image in unavailable[:20])
        suffix = "\n  - ..." if len(unavailable) > 20 else ""
        raise FileNotFoundError(
            "The saved benchmark cohort cannot be resumed because source slide files are "
            f"not currently readable. Missing files:\n{rendered}{suffix}"
        )

    return run_images, _saved_ground_truths(output_dir, manifest, run_images)


def _manifest_count(manifest: dict[str, Any], field: str) -> int:
    value = manifest.get(field, [])
    return len(value) if isinstance(value, list) else 0


def _print_dataset_summary(
    *,
    metadata: dict[str, Any],
    run_images: list[Path],
    ground_truths: dict[Path, dict],
    resumed_contract: bool,
) -> None:
    """Print a single, honest summary for either a fresh or saved cohort."""
    pairing = metadata.get("ground_truth_pairing")
    pairing = pairing if isinstance(pairing, dict) else {}
    missing = pairing.get("missing_annotations", [])
    empty = pairing.get("empty_annotations", [])
    unpaired_mode = str(metadata.get("unpaired_slide_mode", UNPAIRED_SLIDE_MODE))

    if resumed_contract:
        print("Dataset contract: restored from the existing output manifest")
    print(f"Slides discovered: {_manifest_count(metadata, 'slides_discovered')}")
    print(f"Valid ground-truth pairs: {len(ground_truths)}")
    print(f"Missing annotations: {len(missing) if isinstance(missing, list) else 0}")
    print(f"Empty annotations after filtering: {len(empty) if isinstance(empty, list) else 0}")
    if unpaired_mode == "skip":
        print(f"Policy: skip unpaired slides — running {len(run_images)} supervised slide(s)")
        print(f"Skipped: {_manifest_count(metadata, 'skipped_slides')} slide(s) without usable ground truth")
    elif unpaired_mode == "unsupervised":
        print(
            "Policy: prediction-only for unpaired slides — "
            f"running {len(run_images)} slide(s): {len(ground_truths)} supervised, "
            f"{_manifest_count(metadata, 'prediction_only_slides')} prediction-only"
        )
    else:
        print(f"Policy: strict pairing — running {len(run_images)} supervised slide(s)")

    groups = metadata.get("asap_groups")
    if str(metadata.get("annotation_suffix", ANNOTATION_SUFFIX)).casefold() == ".xml":
        print("ASAP groups: " + ", ".join(groups) if isinstance(groups, list) and groups else "ASAP groups: all")
    else:
        print("Ground truth: GeoJSON")
    print(f"WSI reader: {WSI_READER}")
    if str(metadata.get("slide_order", SLIDE_ORDER)).casefold() == "smallest-first":
        print("Slide order: smallest file to largest file (on-disk bytes)")
    else:
        print("Slide order: alphabetical (case-insensitive filename)")
    print(f"Execution device: {BENCHMARK_DEVICE}")
    print("Process policy: adaptive slide sharding (one GPU process; bounded CPU workers)")
    print(f"Prediction heatmaps: {'enabled' if SAVE_PREDICTION_HEATMAPS else 'disabled'}")
    print("Metrics: deferred to evaluate_outputs.py (vector metrics by default; pixel metrics opt-in)")


def _fresh_dataset_contract(
    *,
    unpaired_mode: str,
    selected_slide_order: str,
) -> tuple[list[Path], dict[Path, dict], dict[str, Any]]:
    """Discover a new live cohort and serialise its immutable contract."""
    _validate_benchmark_paths(unpaired_mode)
    images = _slide_paths(WSI_DIR, order=selected_slide_order)
    if not images:
        raise FileNotFoundError(f"No supported WSI files found in {WSI_DIR}")

    groups = ASAP_GROUPS if ANNOTATION_SUFFIX.casefold() == ".xml" else None
    pairing = seg.inspect_ground_truths(
        images,
        annotation_dir=GROUND_TRUTH_DIR,
        suffix=ANNOTATION_SUFFIX,
        groups=groups,
    )
    if unpaired_mode == "strict" and pairing.has_unpaired_images:
        raise FileNotFoundError(seg.format_ground_truth_pairing_error(pairing))

    run_images, supervised_images, unpaired_images = _select_images_for_policy(
        images,
        set(pairing.ground_truths),
        unpaired_mode,
    )
    metadata = _dataset_metadata(
        pairing=pairing,
        discovered_images=images,
        run_images=run_images,
        supervised_images=supervised_images,
        unpaired_images=unpaired_images,
        unpaired_mode=unpaired_mode,
        slide_order=selected_slide_order,
    )
    return run_images, pairing.ground_truths, metadata


def _uses_accelerator(segmenters: list[object]) -> bool:
    """Return whether any selected method will actively use CUDA or Metal."""
    if BENCHMARK_DEVICE == "cpu":
        return False
    for segmenter in segmenters:
        device = str(getattr(segmenter, "device", "")).casefold()
        if device == "mps" or device.startswith("cuda"):
            return True
        # Trident resolves its device lazily from SEGMENTEER_TRIDENT_DEVICE.
        if segmenter.__class__.__name__ == "TRIDENTSegmenter":
            return True
    return False


def _resolve_slide_workers(requested: int | str | None, segmenters: list[object]) -> int:
    """Resolve an explicit or adaptive number of independent slide workers."""
    value = SLIDE_WORKERS if requested is None else requested
    if isinstance(value, str):
        normalised = value.strip().casefold()
        if normalised != "auto":
            try:
                value = int(normalised)
            except ValueError as exc:
                raise ValueError("SLIDE_WORKERS must be a positive integer or 'auto'.") from exc
        else:
            # One active MPS/CUDA inference stream avoids competing GPU contexts.
            # CPU-only masks are independent and benefit from process isolation.
            if _uses_accelerator(segmenters):
                return 1
            return max(1, min(12, os.cpu_count() or 1))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("--workers must be a positive integer.")
    return value


def _shard_images(images: list[Path], *, shard_index: int, shard_count: int) -> list[Path]:
    """Return a deterministic, disjoint slice for one zero-based worker index."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid slide-shard index/count.")
    return images[shard_index::shard_count]


def _child_environment(*, cpu_parallel: bool) -> dict[str, str]:
    """Build a child environment that prevents native-library oversubscription."""
    env = os.environ.copy()
    if cpu_parallel:
        # Each shard gets one native compute thread.  N Python processes then
        # occupy the machine without N×BLAS/OpenMP worker explosions.
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


def _launch_slide_shards(
    *,
    output_dir: Path,
    worker_count: int,
    cpu_parallel: bool,
) -> None:
    """Run disjoint slide shards in child interpreters sharing one output folder."""
    script = Path(__file__).resolve()
    processes: list[subprocess.Popen[object]] = []
    print(f"Launching {worker_count} cooperative slide worker(s) into {output_dir}")
    for shard_index in range(worker_count):
        command = [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
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
                cwd=PROJECT_ROOT,
                env=_child_environment(cpu_parallel=cpu_parallel),
            )
        )

    exit_codes = [process.wait() for process in processes]
    failed = [str(index) for index, code in enumerate(exit_codes) if code != 0]
    if failed:
        raise RuntimeError(
            "One or more slide workers failed (worker indexes: " + ", ".join(failed) + ")."
        )


def _run_benchmark(
    output_dir: Path | None = None,
    *,
    slide_order: str | None = None,
    workers: int | str | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Path:
    """Run a complete cohort, or one cooperative shard of a prepared cohort."""
    if (shard_index is None) != (shard_count is None):
        raise ValueError("--shard-index and --shard-count must be used together.")
    is_child_shard = shard_index is not None
    if is_child_shard and (shard_count is None or shard_count < 2):
        raise ValueError("--shard-count must be at least 2 for a cooperative worker.")

    selected_output_dir = _normalise_selected_output_dir(output_dir)
    existing_manifest = _read_existing_dataset_manifest(selected_output_dir)

    if existing_manifest is not None:
        _validate_benchmark_paths(_unpaired_mode(), validate_live_source=False)
        run_images, ground_truths = _saved_dataset_contract(
            selected_output_dir,
            existing_manifest,
        )
        metadata = existing_manifest
        resumed_contract = True
    else:
        unpaired_mode = _unpaired_mode()
        selected_slide_order = _normalise_slide_order(slide_order or SLIDE_ORDER)
        run_images, ground_truths, metadata = _fresh_dataset_contract(
            unpaired_mode=unpaired_mode,
            selected_slide_order=selected_slide_order,
        )
        resumed_contract = False

    if not is_child_shard:
        _print_dataset_summary(
            metadata=metadata,
            run_images=run_images,
            ground_truths=ground_truths,
            resumed_contract=resumed_contract,
        )

        worker_count = min(
            _resolve_slide_workers(workers, SEGMENTERS),
            len(run_images),
        )
        if worker_count > 1:
            # Write the full contract once before children run.  Each child then
            # restores it and receives a deterministic, non-overlapping slice.
            prepared_output_dir = seg.prepare_dataset_output(
                output_root=OUTPUT_ROOT,
                output_dir=selected_output_dir,
                ground_truths=ground_truths,
                dataset_metadata=metadata,
            )
            _announce_live_viewer(prepared_output_dir)
            _launch_slide_shards(
                output_dir=prepared_output_dir,
                worker_count=worker_count,
                cpu_parallel=not _uses_accelerator(SEGMENTERS),
            )
            return prepared_output_dir

        if selected_output_dir is not None:
            print(f"Output directory: creating or joining {selected_output_dir}")
        else:
            print("Output directory: a new timestamped folder will be created.")
        return seg.run_dataset(
            SEGMENTERS,
            run_images,
            ground_truths=ground_truths,
            save_heatmaps=SAVE_PREDICTION_HEATMAPS,
            output_root=OUTPUT_ROOT,
            output_dir=selected_output_dir,
            dataset_metadata=metadata,
            output_ready_callback=_announce_live_viewer,
            dataset_index_refresh_interval=DATASET_INDEX_REFRESH_INTERVAL,
        )

    assert shard_index is not None and shard_count is not None
    shard_images = _shard_images(
        run_images,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    shard_ground_truths = {
        image: ground_truths[image]
        for image in shard_images
        if image in ground_truths
    }
    print(
        f"Slide shard {shard_index + 1}/{shard_count}: "
        f"{len(shard_images)} of {len(run_images)} slide(s)"
    )
    if selected_output_dir is None:
        raise RuntimeError("A cooperative slide worker requires a prepared output directory.")

    return seg.run_dataset(
        SEGMENTERS,
        shard_images,
        ground_truths=shard_ground_truths,
        save_heatmaps=SAVE_PREDICTION_HEATMAPS,
        output_root=OUTPUT_ROOT,
        output_dir=selected_output_dir,
        dataset_metadata=metadata,
        dataset_index_refresh_interval=DATASET_INDEX_REFRESH_INTERVAL,
        shared_method_workers=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a segmenteer dataset benchmark.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--output-dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help=(
            "create or join this output directory; a saved folder restores its exact "
            "slide/ground-truth cohort and only the selected method changes "
            "(relative paths are resolved from the project root)"
        ),
    )
    mode.add_argument(
        "--new",
        action="store_true",
        help="ignore DEFAULT_OUTPUT_DIR and create a new timestamped output directory",
    )
    parser.add_argument(
        "--slide-order",
        choices=("alphabetical", "smallest-first"),
        metavar="ORDER",
        help=(
            "override SLIDE_ORDER: alphabetical, or smallest-first by on-disk byte size"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help=(
            "independent slide processes for this run; omit for adaptive auto mode "
            "(one MPS/CUDA process, up to 12 CPU-only processes)"
        ),
    )
    # Internal cooperative-shard arguments used by the parent launcher. They
    # deliberately remain available for externally managed job schedulers.
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    return parser


def _select_output_dir(args: argparse.Namespace) -> Path | None:
    """Return the selected output directory, or ``None`` for a new timestamp."""
    if args.output_dir is not None:
        return args.output_dir
    if args.new:
        return None
    return DEFAULT_OUTPUT_DIR


def main() -> None:
    args = _build_parser().parse_args()
    selected_output_dir = _select_output_dir(args)
    output_dir = _run_benchmark(
        selected_output_dir,
        slide_order=args.slide_order,
        workers=args.workers,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(f"\nResults written to: {output_dir}\n")
    print("To calculate optional post-run metrics from saved GeoJSON outputs:")
    print(f"  python evaluate_outputs.py {shlex.quote(str(output_dir))}")
    print("Add --pixel-metrics only when full-resolution raster metrics are required.")
    print("To review this folder after completion, or while a later run is active:")
    print(
        "  python -m app "
        f"--output {shlex.quote(str(output_dir))} "
        f"--data {shlex.quote(str(WSI_DIR))}"
    )


if __name__ == "__main__":
    main()

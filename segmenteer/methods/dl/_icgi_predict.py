"""Run the official ICGI predictor with an explicit checkpoint directory."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path


def _icgi_numpy_safe_globals():
    """Allowlist NumPy types used by ICGI checkpoints during restricted loading.

    Supports NumPy 1.x and 2.x serialization without enabling arbitrary pickle
    deserialization.
    """
    import numpy as np

    try:
        from numpy._core import multiarray
    except ImportError:  # NumPy 1.x
        from numpy.core import multiarray

    safe = [
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        (multiarray.scalar, "numpy._core.multiarray.scalar"),
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
        np.dtype,
        np.ndarray,
    ]
    # Include both NumPy 1.x and 2.x dtype classes.
    for dtype_name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        safe.append(type(np.dtype(dtype_name)))
    return safe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run official ICGI predictor on a PNG folder"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args(argv)
    try:
        import torch
        from nnunetv2.inference.predict_tissue import TissueNNUnetPredictor
    except ImportError as exc:
        raise SystemExit(
            "ICGI needs the authors' customized nnU-Net fork, not the ordinary PyPI nnunetv2. "
            'Install with: pip install "git+https://github.com/icgi/'
            'Reliable-and-Efficient-Tissue-Segmentation-in-Whole-Slide-Images.git"'
        ) from exc
    choice = args.device
    if choice == "auto":
        choice = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    device = torch.device(choice)
    if choice == "cpu":
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
    elif choice == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    predictor = TissueNNUnetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    # Scope the NumPy allowlist to ICGI initialization under PyTorch 2.6+.
    safe_globals = getattr(getattr(torch, "serialization", None), "safe_globals", None)
    compatibility = (
        safe_globals(_icgi_numpy_safe_globals())
        if safe_globals is not None
        else nullcontext()
    )
    with compatibility:
        predictor.initialize_from_trained_tissue_model_folder(
            str(args.model_dir),
            use_folds="all",
            checkpoint_name=args.checkpoint,
        )
    predictor.predict_tissue_from_files(
        str(args.input),
        str(args.output),
        suffix=None,
        extension=None,
        exclude=None,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        binary_01=False,
        pp_cfg=None,
    )


if __name__ == "__main__":
    main()

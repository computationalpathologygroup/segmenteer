from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from segmenteer.model_cache import get_method_model_dir

try:
    from trident.segmentation_models.load import (
        SegmentationModel as _TridentSegmentationModel,
    )

    _TRIDENT_AVAILABLE = True
except Exception as exc:
    _TridentSegmentationModel = object
    _TRIDENT_AVAILABLE = False
    _TRIDENT_IMPORT_ERROR = exc
try:
    import torch
    from torch import nn
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
try:
    import onnxruntime

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def get_model_cache_dir() -> Path:
    return get_method_model_dir("cpg")


def _onnx_intra_op_threads() -> int:
    """Limit ONNX threads to the Slurm allocation or allowed CPU affinity.

    Explicitly setting ORT intra_op_num_threads also suppresses its default
    CPU affinity assignments, which fail in some Slurm/Pyxis cpusets.
    """
    for variable in (
        "SEGMENTEER_ONNX_THREADS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_PER_GPU"
    ):
        value = os.environ.get(variable)
        if value is None or not value.strip():
            continue
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f"{variable} must be a positive integer, got {value!r}") from exc
        if count <= 0:
            raise ValueError(f"{variable} must be a positive integer, got {value!r}")
        return count
    if hasattr(os, "sched_getaffinity"):
        allowed = len(os.sched_getaffinity(0))
        if allowed:
            return allowed
    return max(1, os.cpu_count() or 1)


class LIBTRIDENTCPGSegmenter(_TridentSegmentationModel):
    def __init__(self, **build_kwargs: dict[str, Any]):
        if not _TORCH_AVAILABLE or not _ONNX_AVAILABLE:
            raise ImportError(
                "torch, torchvision, trident, and onnxruntime are required for CPG tissue segmenter.\nInstall with: pip install 'segmenteer[cpg]'"
            )
        super().__init__(**build_kwargs)

    def _build(self) -> tuple[nn.Module, transforms.Compose]:
        model_ckpt_name = "cpg.onnx"
        weights_path = get_model_cache_dir() / Path(model_ckpt_name)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"CPG model weights not found at '{weights_path}'. Expected ONNX weights file 'models/cpg/cpg.onnx'. Ensure the model assets are present and installed correctly before initializing LIBTRIDENTCPGSegmenter."
            )
        available_providers = onnxruntime.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = _onnx_intra_op_threads()
        self.ort_session = onnxruntime.InferenceSession(
            str(weights_path), sess_options=options, providers=providers
        )
        self.input_size = 224
        self.precision = torch.float32
        self.target_mag = 4
        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.softmax_fn = torch.nn.LogSoftmax(dim=1)
        return (nn.Module(), eval_transforms)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) == 4, (
            f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        )
        assert image.shape[1] == 3, (
            f"Input must have 3 channels (C), got {image.shape[1]} instead"
        )
        assert (
            image.shape[2] == self.input_size and image.shape[3] == self.input_size
        ), (
            f"Input must be of shape (batch_size, 3, {self.input_size}, {self.input_size}), got {image.shape} instead"
        )
        onnx_inputs = [image.numpy(force=True)]
        onnxruntime_input = {
            input_arg.name: input_value
            for input_arg, input_value in zip(
                self.ort_session.get_inputs(), onnx_inputs
            )
        }
        onnxruntime_outputs = self.ort_session.run(None, onnxruntime_input)[0]
        out = torch.nn.functional.log_softmax(
            torch.as_tensor(onnxruntime_outputs, dtype=torch.float), dim=1
        )
        return torch.argmax(out, dim=1).to(torch.uint8)

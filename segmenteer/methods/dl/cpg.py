from pathlib import Path
from typing import Any

from segmenteer.core.base import TRIDENTSegmentationModel

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

from segmenteer.core.base import TRIDENTSegmentationModel


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "cpg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

class LIBTRIDENTCPGSegmenter(TRIDENTSegmentationModel):
    def __init__(self, **build_kwargs: dict[str, Any]):
        if not _TORCH_AVAILABLE or not _ONNX_AVAILABLE:
            raise ImportError(
                "torch, torchvision, trident, and onnxruntime are required for CPG tissue segmenter.\n"
                "Install with: pip install 'segmenteer[cpg]'"
            )
        super().__init__(**build_kwargs)

    def _build(self) -> tuple[nn.Module, transforms.Compose]:
        """
        Build and load CPGSegmenter model.

        Returns
        -------
        Tuple[nn.Module, transforms.Compose]
            Model and preprocessing transforms.
        """

        model_ckpt_name = "cpg.onnx"
        weights_path = get_model_cache_dir() / Path(model_ckpt_name)

        self.ort_session = onnxruntime.InferenceSession(
            weights_path, providers=["CPUExecutionProvider"]
        )

        # Store configuration
        self.input_size = 224  # This cannot be changed because of onnx export constraints
        self.precision = torch.float32
        self.target_mag = 4

        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.softmax_fn = torch.nn.LogSoftmax(dim=1)

        return nn.Module(), eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # input should be of shape (batch_size, C, H, W)
        assert (
            len(image.shape) == 4
        ), f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        assert (
            image.shape[1] == 3
        ), f"Input must have 3 channels (C), got {image.shape[1]} instead"
        assert (
            image.shape[2] == self.input_size and image.shape[3] == 224
        ), f"Input must be of shape (batch_size, 3, {self.input_size}, {self.input_size}), got {image.shape} instead"
        onnx_inputs = [image.numpy(force=True)]
        onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(self.ort_session.get_inputs(), onnx_inputs)}
        onnxruntime_outputs = self.ort_session.run(None, onnxruntime_input)[0]
        out = torch.nn.functional.log_softmax(
            torch.as_tensor(onnxruntime_outputs, dtype=torch.float),  # The output is (b, 2, 224, 224), note the 2. The foreground and background class need argmax to select the most likely class.
            dim=1,
        )
        return torch.argmax(out, axis=1).to(torch.uint8)

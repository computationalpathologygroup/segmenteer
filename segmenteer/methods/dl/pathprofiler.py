"""PathProfiler adapter compatible with the pinned Trident API.

The module remains import-safe when optional dependencies are absent.  The
actual Trident base class is imported only when this adapter is selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from segmenteer.model_cache import (
    find_local_model,
    get_method_model_dir,
    promote_to_method_cache,
)

_TRIDENT_IMPORT_ERROR: Exception | None = None
try:
    from trident.segmentation_models.load import SegmentationModel as _TridentSegmentationModel

    _TRIDENT_AVAILABLE = True
except Exception as exc:  # optional dependency; do not fail top-level imports
    _TridentSegmentationModel = object  # type: ignore[assignment,misc]
    _TRIDENT_AVAILABLE = False
    _TRIDENT_IMPORT_ERROR = exc

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    from torch import nn
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except Exception as exc:  # torchvision can raise ABI/runtime errors too
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


def get_model_cache_dir() -> Path:
    """Return the shared local PathProfiler weights directory."""
    return get_method_model_dir("pathprofiler")


class LIBTRIDENTPathProfilerSegmenter(_TridentSegmentationModel):
    """Trident model wrapper for PathProfiler's binary tissue checkpoint."""

    def __init__(self, **build_kwargs: Any):
        if not _TRIDENT_AVAILABLE or not _TORCH_AVAILABLE:
            cause = _TRIDENT_IMPORT_ERROR or _TORCH_IMPORT_ERROR
            raise ImportError(
                "TRIDENTPathProfilerSegmenter requires Torch, torchvision, and Trident. "
                "Run: uv sync --extra pathprofiler"
            ) from cause
        super().__init__(**build_kwargs)

    def _build(self) -> tuple[nn.Module, transforms.Compose]:
        model_ckpt_name = "PathProfiler/pathprofiler_tissue_seg_jit.pt"
        checkpoint_file = Path(model_ckpt_name).name
        weights_path = find_local_model("pathprofiler", checkpoint_file)

        if weights_path is None:
            if not getattr(_TridentSegmentationModel, "_has_internet", False):
                raise FileNotFoundError(
                    "PathProfiler weights are not available locally. Download "
                    f"{model_ckpt_name} from RendeiroLab/LazySlide-models-gpl and "
                    f"place it below {get_model_cache_dir()}."
                )
            from huggingface_hub import snapshot_download

            checkpoint_dir = Path(
                snapshot_download(
                    repo_id="RendeiroLab/LazySlide-models-gpl",
                    repo_type="model",
                    local_dir=get_model_cache_dir(),
                    cache_dir=get_model_cache_dir(),
                    allow_patterns=[model_ckpt_name],
                )
            )
            weights_path = promote_to_method_cache(
                checkpoint_dir / model_ckpt_name, "pathprofiler", checkpoint_file
            )

        model = torch.jit.load(str(weights_path), map_location="cpu")
        model.eval()

        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 4
        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Input must be 4D (batch, channels, height, width), got {image.shape}")
        logits = self.model(image)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 4 or logits.shape[1] < 2:
            raise ValueError(
                "PathProfiler checkpoint must return class logits shaped "
                "(batch, >=2, height, width)."
            )
        probabilities = torch.softmax(logits, dim=1)
        return (probabilities[:, 1] > self.confidence_thresh).to(torch.uint8)

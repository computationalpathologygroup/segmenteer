from pathlib import Path
from typing import Any

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
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
    """Return the shared local SlideSegmenter weights directory."""
    return get_method_model_dir("rtlucassen")


class RTLucassenSlideSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 7.04,  # 7.04 mpp corresponds to 1.25x magnification described in paper.
        device: str | None = None,
        *args,
        **kwargs,
    ):
        try:
            from slidesegmenter import SlideSegmenter
        except ImportError:
            raise ImportError(
                "slidesegmenter is required for RTLucassenSlideSegmenter.\n"
                "Install with: uv sync --extra rtlucassen"
            ) from None
        # SlideSegmenter does not support MPS, so retain CUDA-or-CPU selection.
        device = resolve_torch_device(device, allow_mps=False)
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.segmenter = SlideSegmenter(
            device=device,
            alternative_directory=get_model_cache_dir(),
            pen_marking_segmentation=False,
        )

    @property
    def name(self) -> str:
        return "rtlucassen_slidesegmenter"

    def _segment_numpy(self, image):
        normalized = image / 255
        segmentation = self.segmenter.segment(normalized)["tissue"].astype(bool)[..., 0]
        return segmentation

class LIBTRIDENTRTLucassenSlideSegmenterSegmenter(_TridentSegmentationModel):
    """Trident model wrapper for RTLucassenSlideSegmenter's binary tissue checkpoint."""

    def __init__(self, **build_kwargs: Any):
        if not _TRIDENT_AVAILABLE or not _TORCH_AVAILABLE:
            cause = _TRIDENT_IMPORT_ERROR or _TORCH_IMPORT_ERROR
            raise ImportError(
                "TRIDENTPathProfilerSegmenter requires Torch, torchvision, and Trident. "
                "Run: uv sync --extra pathprofiler"
            ) from cause
        super().__init__(**build_kwargs)

    def _build(self) -> tuple[nn.Module, transforms.Compose]:
        model_ckpt_name = "2025-10-18/model_state_dict.pth"
        checkpoint_file = Path(model_ckpt_name).name
        weights_path = find_local_model("rtlucassen", checkpoint_file)

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
                    repo_id="rtlucassen/slidesegmenter",
                    repo_type="model",
                    local_dir=get_model_cache_dir(),
                    cache_dir=get_model_cache_dir(),
                    allow_patterns=[model_ckpt_name],
                )
            )
            weights_path = promote_to_method_cache(
                checkpoint_dir / model_ckpt_name, "rtlucassen", checkpoint_file
            )

        from slidesegmenter import SlideSegmenter
        self.segmenter = SlideSegmenter(
            device="cuda",
            alternative_directory=get_model_cache_dir(),
            pen_marking_segmentation=False,
            channels_last=False,
        )
        model = self.segmenter.model

        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 1.25
        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return model, eval_transforms

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Input must be 4D (batch, channels, height, width), got {image.shape}")
        out = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
        for i, item in enumerate(image):
            # TODO: SlideSegmenter expects the item to be on the cpu first, but this copies the input back and forth.
            out[i, ...] = torch.as_tensor(data=self.segmenter.segment(item.cpu())["tissue"][0, ...], dtype=torch.bool)
        return out.to(torch.uint8)
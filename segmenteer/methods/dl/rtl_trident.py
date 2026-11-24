from pathlib import Path
from typing import Any
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import (
    find_local_model,
    get_method_model_dir,
    promote_to_method_cache,
)
from segmenteer.external_assets import require_network

_TRIDENT_IMPORT_ERROR: Exception | None = None
try:
    from trident.segmentation_models.load import (
        SegmentationModel as _TridentSegmentationModel,
    )

    _TRIDENT_AVAILABLE = True
except Exception as exc:
    _TridentSegmentationModel = object
    _TRIDENT_AVAILABLE = False
    _TRIDENT_IMPORT_ERROR = exc
_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    from torch import nn
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except Exception as exc:
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


def get_model_cache_dir() -> Path:
    return get_method_model_dir("rtlucassen")


class LIBTRIDENTRTLucassenSlideSegmenterSegmenter(_TridentSegmentationModel):
    def __init__(self, **build_kwargs: Any):
        if not _TRIDENT_AVAILABLE or not _TORCH_AVAILABLE:
            cause = _TRIDENT_IMPORT_ERROR or _TORCH_IMPORT_ERROR
            raise ImportError(
                "TRIDENTPathProfilerSegmenter requires Torch, torchvision, and Trident. Run: uv sync --extra pathprofiler"
            ) from cause
        super().__init__(**build_kwargs)

    def _build(self) -> tuple[nn.Module, transforms.Compose]:
        model_ckpt_name = "2025-10-18/model_state_dict.pth"
        checkpoint_file = Path(model_ckpt_name).name
        weights_path = find_local_model("rtlucassen", checkpoint_file)
        if weights_path is None:
            if not getattr(_TridentSegmentationModel, "_has_internet", False):
                raise FileNotFoundError(
                    f"PathProfiler weights are not available locally. Download {model_ckpt_name} from RendeiroLab/LazySlide-models-gpl and place it below {get_model_cache_dir()}."
                )
            require_network(
                "TRIDENT/SlideSegmenter unpinned Hugging Face snapshot", pinned=False
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
            device=resolve_torch_device(None, allow_mps=False),
            alternative_directory=get_model_cache_dir(),
            pen_marking_segmentation=False,
            channels_last=False,
        )
        model = self.segmenter.model
        self.input_size = 512
        self.precision = torch.float32
        self.target_mag = 1.25
        eval_transforms = transforms.Compose([transforms.ToTensor()])
        return (model, eval_transforms)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(
                f"Input must be 4D (batch, channels, height, width), got {image.shape}"
            )
        out = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
        for i, item in enumerate(image):
            out[i, ...] = torch.as_tensor(
                data=self.segmenter.segment(item.cpu())["tissue"][0, ...],
                dtype=torch.bool,
            )
        return out.to(torch.uint8)

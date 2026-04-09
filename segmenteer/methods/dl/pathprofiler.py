from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "pathprofiler"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class LIBTRIDENTPathProfilerSegmenter:
    """PathProfiler segmenter using TRIDENT framework. Lazily loads TRIDENT on first use."""

    def __new__(cls, **build_kwargs: dict[str, Any]):
        """Lazily import and construct TRIDENT segmenter."""
        from segmenteer.core.base import _require_trident

        TRIDENTSegmentationModel, _ = _require_trident()

        # Dynamically create the actual class with TRIDENT base
        class _PathProfilerImpl(TRIDENTSegmentationModel):
            def __init__(self, **kwargs: dict[str, Any]):
                if not _TORCH_AVAILABLE:
                    raise ImportError(
                        "torch, torchvision, and trident are required for PathProfiler.\n"
                        "Install with: pip install 'segmenteer[pathprofiler]'"
                    )
                super().__init__(**kwargs)

            def _build(self) -> tuple[nn.Module, transforms.Compose]:
                """
                Build and load HESTSegmenter model.

                Returns
                -------
                Tuple[nn.Module, transforms.Compose]
                    Model and preprocessing transforms.
                """

                model_ckpt_name = "PathProfiler/pathprofiler_tissue_seg_jit.pt"
                weights_path = get_model_cache_dir() / Path(model_ckpt_name)

                if not weights_path.exists():
                    if not TRIDENTSegmentationModel._has_internet:
                        raise FileNotFoundError(
                            f"Internet connection not available and checkpoint not found locally at {get_model_cache_dir()}.\n\n"
                            f"To proceed, please manually download {model_ckpt_name} from:\n"
                            f"https://huggingface.co/RendeiroLab/LazySlide-models-gpl/\n"
                            f"and place it at:\n{weights_path}"
                        )

                    # If internet is available, download from HuggingFace
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

                    weights_path = checkpoint_dir / model_ckpt_name

                # Load and clean checkpoint
                model = torch.jit.load(weights_path)
                model.eval()

                # Store configuration
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
                # input should be of shape (batch_size, C, H, W)
                assert len(image.shape) == 4, (
                    f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
                )
                softmax_output = self.model(image).softmax(1)
                predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(
                    torch.uint8
                )  # Shape: [bs, 512, 512]
                return predictions

        # Instantiate the dynamic class
        return _PathProfilerImpl(**build_kwargs)

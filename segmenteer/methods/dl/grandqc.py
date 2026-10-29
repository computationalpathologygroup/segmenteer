from segmenteer.benchmark.profiling import measure_phase
from pathlib import Path
import numpy as np
from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import find_local_model, get_method_model_dir
from segmenteer.external_assets import atomic_https_download
from segmenteer.methods.dl._checkpoint_validation import load_complete_state_dict

try:
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_model_cache_dir() -> Path:
    return get_method_model_dir("grandqc")


class GrandQCSegmenter(NumpySegmenter):
    MODEL_FILE = "Tissue_Detection_MPP10.pth"
    ZENODO_RECORD_ID = "14507273"
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float | None = 10,
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        *args,
        **kwargs,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchvision are required for GrandQCSegmenter.\nInstall with: uv sync --extra grandqc"
            )
        super().__init__(*args, **kwargs)
        if mpp is None:
            mpp = 10
        self.mpp = mpp
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(resolve_torch_device(device))
        print(
            f"Initializing GrandQC Tissue Detection model (MPP {self.mpp}, 1x magnification)"
        )
        self._model = None
        self._transform = None

    def _download_from_zenodo(self, filename: str, cache_dir: Path) -> Path:
        # Require an explicit opt-in or SEGMENTEER_GRANDQC_SHA256 for downloads.
        import os

        digest = os.getenv("SEGMENTEER_GRANDQC_SHA256") or None
        url = f"https://zenodo.org/records/{self.ZENODO_RECORD_ID}/files/{filename}"
        return atomic_https_download(
            url,
            cache_dir / filename,
            asset="GrandQC Zenodo checkpoint",
            expected_sha256=digest,
            pinned=False,
        )

    def _load_model(self):
        # Cache the network only after checkpoint validation succeeds.
        self._model = None
        self._transform = None
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required for GrandQC. Run: uv sync --extra grandqc"
            )
        model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None,
        )
        cache_dir = get_model_cache_dir()
        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path)
        else:
            checkpoint_file = find_local_model("grandqc", self.MODEL_FILE)
            if checkpoint_file is None:
                try:
                    checkpoint_file = self._download_from_zenodo(
                        self.MODEL_FILE, cache_dir
                    )
                except Exception as e:
                    print(
                        f"Error downloading from Zenodo: {e}\nPlease download {self.MODEL_FILE} manually from:\nhttps://zenodo.org/records/{self.ZENODO_RECORD_ID}\nand place it in {cache_dir}"
                    )
                    raise
        try:
            state_dict = torch.load(
                checkpoint_file, map_location=self.device, weights_only=True
            )
            load_complete_state_dict(
                model,
                state_dict,
                model_name="GrandQC",
                checkpoint_path=checkpoint_file,
            )
            print(f"Loaded GrandQC Tissue Detection weights from {checkpoint_file}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise
        model = model.to(self.device)
        model.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._model = model
        self._transform = transform

    @property
    def name(self) -> str:
        return f"grandqc_tissue_detection_mpp{self.mpp}"

    def _pad_to_divisible(self, image: np.ndarray, divisor: int = 32):
        h, w = image.shape[:2]
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        if pad_h == 0 and pad_w == 0:
            return (image, 0, 0)
        if image.ndim == 3:
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
        return (padded, pad_h, pad_w)

    def _preprocess_image(self, image: np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        padded_image, pad_h, pad_w = self._pad_to_divisible(image)
        pil_image = PILImage.fromarray(padded_image)
        tensor = self._transform(pil_image)
        return (tensor.unsqueeze(0).to(self.device), pad_h, pad_w)

    def _postprocess_output(
        self, outputs: torch.Tensor, original_shape: tuple, pad_h: int, pad_w: int
    ) -> np.ndarray:
        probs = torch.softmax(outputs, dim=1)
        tissue_probs = probs[:, 0, :, :]
        if pad_h > 0 or pad_w > 0:
            h, w = original_shape[:2]
            tissue_probs = tissue_probs[:, :h, :w]
        mask = (tissue_probs > self.confidence_threshold).squeeze().cpu().numpy()
        return mask.astype(bool)

    def _segment_numpy(self, image):
        if self._model is None:
            with measure_phase("model_initialization"):
                self._load_model()
        input_tensor, pad_h, pad_w = self._preprocess_image(image)
        with torch.no_grad():
            outputs = self._model(input_tensor)
        return self._postprocess_output(outputs, image.shape, pad_h, pad_w)

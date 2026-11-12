from segmenteer.benchmark.profiling import measure_phase
from pathlib import Path
import numpy as np
from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import (
    find_local_model,
    get_method_model_dir,
    promote_to_method_cache,
)
from segmenteer.external_assets import require_network
from segmenteer.methods.dl._checkpoint_validation import load_complete_state_dict

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_resnet50

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
_MODEL_NAMESPACE = "hest"


def get_model_cache_dir() -> Path:
    return get_method_model_dir(_MODEL_NAMESPACE)


class HESTSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        model_repo: str = "MahmoodLab/hest-tissue-seg",
        model_file: str = "deeplabv3_seg_v4.ckpt",
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        mpp: float | None = 8,
        *args,
        **kwargs,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchvision are required for HESTSegmenter.\nInstall with: uv sync --extra hest"
            )
        super().__init__(*args, **kwargs)
        self.model_repo = model_repo
        self.model_file = model_file
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        if mpp is None:
            mpp = 8
        self.mpp = mpp
        self.device = torch.device(resolve_torch_device(device))
        print(f"Initializing HEST model at {self.mpp} MPP")
        self._model = None
        self._transform = None

    def _checkpoint_file(self) -> Path:
        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path).expanduser()
            if not checkpoint_file.is_file():
                raise FileNotFoundError(
                    f"HEST checkpoint_path does not exist: {checkpoint_file}"
                )
            return checkpoint_file
        local_file = find_local_model(_MODEL_NAMESPACE, self.model_file)
        if local_file is not None:
            return local_file
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface-hub is required to download missing HEST weights. Run: uv sync --extra hest"
            ) from exc
        try:
            cached_file = Path(
                hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                    local_files_only=True,
                )
            )
        except Exception:
            cached_file = None
        if cached_file is not None and cached_file.is_file():
            return promote_to_method_cache(
                cached_file, _MODEL_NAMESPACE, self.model_file
            )
        require_network("HEST Hugging Face checkpoint (unversioned)", pinned=False)
        print(f"Downloading HEST model from Hugging Face: {self.model_repo}...")
        downloaded_file = Path(
            hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                cache_dir=get_model_cache_dir(),
            )
        )
        return promote_to_method_cache(
            downloaded_file, _MODEL_NAMESPACE, self.model_file
        )

    def _load_model(self):
        # Expose the model only after complete checkpoint loading.
        self._model = None
        self._transform = None
        model = deeplabv3_resnet50(weights=None, num_classes=2)
        try:
            checkpoint_file = self._checkpoint_file()
            import os

            # Unsafe pickle loading requires explicit operator approval.
            trust_pickle = os.getenv(
                "SEGMENTEER_TRUST_PICKLE_CHECKPOINTS", ""
            ).lower() in ("1", "true", "yes", "on")
            try:
                checkpoint = torch.load(
                    checkpoint_file,
                    map_location=self.device,
                    weights_only=not trust_pickle,
                )
            except Exception as exc:
                if not trust_pickle:
                    raise RuntimeError(
                        "HEST checkpoint cannot be loaded with safe weights-only deserialization. "
                        "Only for a checkpoint whose origin you have independently verified, "
                        "set SEGMENTEER_TRUST_PICKLE_CHECKPOINTS=1 to permit legacy pickle loading."
                    ) from exc
                raise
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            load_complete_state_dict(
                model,
                state_dict,
                model_name="HEST",
                checkpoint_path=checkpoint_file,
                lightning_model_prefix=True,
                drop_hest_aux_classifier=True,
            )
            print(f"Loaded HEST weights from {checkpoint_file}")
        except Exception as e:
            print(f"Error loading HEST model weights from {checkpoint_file}: {e}")
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
        mode = "fast" if self.mpp == 2.0 else "default"
        return f"hest_deeplabv3_{mode}_mpp{self.mpp}"

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess_output(self, outputs: dict, original_shape: tuple) -> np.ndarray:
        logits = outputs["out"]
        upsampled_logits = F.interpolate(
            logits, size=original_shape[:2], mode="bilinear", align_corners=False
        )
        probs = torch.softmax(upsampled_logits, dim=1)
        tissue_probs = probs[:, 1, :, :]
        mask = (tissue_probs > self.confidence_threshold).squeeze().cpu().numpy()
        return mask.astype(bool)

    def _segment_numpy(self, image):
        if self._model is None:
            with measure_phase("model_initialization"):
                self._load_model()
        original_shape = image.shape
        input_tensor = self._preprocess_image(image)
        with torch.no_grad():
            outputs = self._model(input_tensor)
        return self._postprocess_output(outputs, original_shape)

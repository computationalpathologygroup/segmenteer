from pathlib import Path

import numpy as np

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import (
    find_local_model,
    get_method_model_dir,
    promote_to_method_cache,
)

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
    """Return the shared local HEST weights directory."""
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
        mpp: float = 8,
        *args,
        **kwargs,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchvision are required for HESTSegmenter.\n"
                "Install with: uv sync --extra hest"
            )
        super().__init__(*args, **kwargs)
        self.model_repo = model_repo
        self.model_file = model_file
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.mpp = mpp

        self.device = torch.device(resolve_torch_device(device))

        print(f"Initializing HEST model at {self.mpp} MPP")

        self._model = None
        self._transform = None
        self._load_model()

    def _checkpoint_file(self) -> Path:
        """Resolve a local HEST weight before consulting Hugging Face.

        Earlier releases passed ``models/hest`` as Hugging Face's cache path.
        Hugging Face therefore nested the checkpoint below ``models--...`` and
        this wrapper never saw it on later runs.  ``find_local_model`` accepts
        that legacy layout, direct project-local files, and the canonical path.
        """
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
                "huggingface-hub is required to download missing HEST weights. "
                "Run: uv sync --extra hest"
            ) from exc

        # Check Hugging Face's existing global cache before attempting network
        # access.  This helps projects that previously downloaded the model
        # through a different cache location.
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
        self._model = deeplabv3_resnet50(weights=None, num_classes=2)
        checkpoint_file = self._checkpoint_file()

        try:
            checkpoint = torch.load(
                checkpoint_file, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint.get("state_dict", checkpoint)

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "") if k.startswith("model.") else k
                new_state_dict[new_key] = v

            self._model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded HEST weights from {checkpoint_file}")
        except Exception as e:
            print(f"Error loading HEST model weights from {checkpoint_file}: {e}")
            raise

        self._model = self._model.to(self.device)
        self._model.eval()

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
            logits,
            size=original_shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(upsampled_logits, dim=1)
        tissue_probs = probs[:, 1, :, :]
        mask = (tissue_probs > self.confidence_threshold).squeeze().cpu().numpy()

        return mask.astype(bool)

    def _segment_numpy(self, image):
        original_shape = image.shape
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            outputs = self._model(input_tensor)

        return self._postprocess_output(outputs, original_shape)

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import Image
import geojson


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "hest"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class HESTSegmenter:
    def __init__(
        self,
        level: int = 1,
        model_repo: str = "MahmoodLab/hest-tissue-seg",
        model_file: str = "deeplabv3_seg_v4.ckpt",
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        min_area: int = 10,
        mpp: float = 1.0,
    ):
        self.level = level  # TODO: level and mpp should be consistent.
        self.model_repo = model_repo
        self.model_file = model_file
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.mpp = mpp

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing HEST model at {self.mpp} MPP")

        self._model = None
        self._transform = None
        self._load_model()

    def _load_model(self):
        self._model = deeplabv3_resnet50(weights=None, num_classes=2)

        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path)
        else:
            checkpoint_file = get_model_cache_dir() / self.model_file

        if checkpoint_file.exists():
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
                print(
                    f"Warning: Could not load checkpoint from {checkpoint_file}. Error: {e}"
                )
                raise
        else:
            try:
                from huggingface_hub import hf_hub_download

                print(f"Downloading HEST model from HuggingFace: {self.model_repo}...")
                downloaded_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                    cache_dir=get_model_cache_dir(),
                )
                checkpoint = torch.load(
                    downloaded_path, map_location=self.device, weights_only=False
                )
                state_dict = checkpoint.get("state_dict", checkpoint)

                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("model.", "") if k.startswith("model.") else k
                    new_state_dict[new_key] = v

                self._model.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded HEST model from HuggingFace")
            except Exception as e:
                print(
                    f"Error downloading/loading model from HuggingFace: {e}\n"
                    f"Please download {self.model_file} manually from:\n"
                    f"https://huggingface.co/{self.model_repo}/tree/main\n"
                    f"and place it in {get_model_cache_dir()}"
                )
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

    def segment(self, image: Image) -> geojson.FeatureCollection:
        image = image.get_numpy_image(level=self.level)
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        original_shape = image.shape

        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            outputs = self._model(input_tensor)

        mask = self._postprocess_output(outputs, original_shape)

        return mask_to_geojson(mask, self.min_area)

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from segmenteer.core.utils import mask_to_geojson


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "grandqc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class GrandQCSegmenter:
    def __init__(
        self,
        model_repo: str = "MahmoodLab/hest-tissue-seg",
        model_file: str = "GrandQC_MPP1_state_dict.pth",
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        min_area: int = 10,
        input_size: int = 512,
    ):
        self.model_repo = model_repo
        self.model_file = model_file
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.input_size = input_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._transform = None
        self._load_model()

    def _load_model(self):
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required for GrandQC. "
                "Install with: pip install segmentation-models-pytorch"
            )

        self._model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path)
        else:
            checkpoint_file = get_model_cache_dir() / self.model_file

        if checkpoint_file.exists():
            try:
                state_dict = torch.load(
                    checkpoint_file, map_location=self.device, weights_only=True
                )
                self._model.load_state_dict(state_dict)
                print(f"Loaded GrandQC weights from {checkpoint_file}")
            except Exception as e:
                print(
                    f"Warning: Could not load checkpoint from {checkpoint_file}. Error: {e}"
                )
        else:
            try:
                from huggingface_hub import hf_hub_download
                
                print(f"Downloading GrandQC model from {self.model_repo}/{self.model_file}...")
                downloaded_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                    cache_dir=get_model_cache_dir()
                )
                state_dict = torch.load(downloaded_path, map_location=self.device, weights_only=True)
                self._model.load_state_dict(state_dict)
                print(f"Loaded GrandQC model from HuggingFace")
            except Exception as e:
                print(
                    f"Warning: Could not download model from HuggingFace. "
                    f"Error: {e}"
                )
                raise

        self._model = self._model.to(self.device)
        self._model.eval()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "grandqc_unetplusplus"

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        tensor = self._transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def _postprocess_output(
        self, outputs: torch.Tensor, original_shape: tuple
    ) -> np.ndarray:
        upsampled_logits = F.interpolate(
            outputs,
            size=original_shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.sigmoid(upsampled_logits)
        mask = (probs > self.confidence_threshold).cpu().numpy()
        
        while mask.ndim > 2:
            mask = mask.squeeze(0)
        
        return mask.astype(bool)

    def segment(self, image: np.ndarray) -> dict:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        original_shape = image.shape

        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            outputs = self._model(input_tensor)

        mask = self._postprocess_output(outputs, original_shape)

        return mask_to_geojson(mask, self.min_area)
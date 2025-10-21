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
    MODEL_FILE = "Tissue_Detection_MPP10.pth"
    ZENODO_RECORD_ID = "14507273"
    MPP = 10.0

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        min_area: int = 10,
        input_size: int = 512,
    ):
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area
        self.input_size = input_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(
            f"Initializing GrandQC Tissue Detection model (MPP {self.MPP}, 1x magnification)"
        )

        self._model = None
        self._transform = None
        self._load_model()

    def _download_from_zenodo(self, filename: str, cache_dir: Path) -> Path:
        import requests
        from tqdm import tqdm

        url = f"https://zenodo.org/records/{self.ZENODO_RECORD_ID}/files/{filename}"
        output_path = cache_dir / filename

        if output_path.exists():
            return output_path

        print(f"Downloading {filename} from Zenodo...")
        print(f"URL: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"Downloaded to {output_path}")
        return output_path

    def _load_model(self):
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required for GrandQC. "
                "Install with: pip install segmentation-models-pytorch"
            )

        self._model = smp.UnetPlusPlus(
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
            checkpoint_file = cache_dir / self.MODEL_FILE

            if not checkpoint_file.exists():
                try:
                    checkpoint_file = self._download_from_zenodo(
                        self.MODEL_FILE, cache_dir
                    )
                except Exception as e:
                    print(
                        f"Error downloading from Zenodo: {e}\n"
                        f"Please download {self.MODEL_FILE} manually from:\n"
                        f"https://zenodo.org/records/{self.ZENODO_RECORD_ID}\n"
                        f"and place it in {cache_dir}"
                    )
                    raise

        try:
            state_dict = torch.load(
                checkpoint_file, map_location=self.device, weights_only=True
            )
            self._model.load_state_dict(state_dict, strict=False)
            print(f"Loaded GrandQC Tissue Detection weights from {checkpoint_file}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
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
        return f"grandqc_tissue_detection_mpp{self.MPP}"

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

        probs = torch.softmax(upsampled_logits, dim=1)
        tissue_probs = probs[:, 1, :, :]
        mask = (tissue_probs > self.confidence_threshold).squeeze().cpu().numpy()

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
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from segmenteer.core.utils import mask_to_geojson


class HESTSegmenter:
    def __init__(
        self,
        model_name: str = "MahmoodLab/hest-tissue-seg",
        device: str | None = None,
        confidence_threshold: float = 0.5,
        min_area: int = 10,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.min_area = min_area

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForSemanticSegmentation,
            )
        except ImportError:
            raise ImportError(
                "transformers is required for HEST segmenter. "
                "Install with: pip install transformers"
            )

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()

    @property
    def name(self) -> str:
        return "hest_deeplabv3"

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            image_uint8 = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_uint8)

        inputs = self._processor(images=pil_image, return_tensors="pt")
        return inputs.pixel_values.to(self.device)

    def _postprocess_output(
        self, outputs: torch.Tensor, original_shape: tuple
    ) -> np.ndarray:
        logits = outputs.logits

        upsampled_logits = F.interpolate(
            logits,
            size=original_shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.sigmoid(upsampled_logits)
        mask = (probs > self.confidence_threshold).squeeze().cpu().numpy()

        return mask.astype(bool)

    def segment(self, image: np.ndarray) -> dict:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        original_shape = image.shape

        pixel_values = self._preprocess_image(image)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values)

        mask = self._postprocess_output(outputs, original_shape)

        return mask_to_geojson(mask, self.min_area)

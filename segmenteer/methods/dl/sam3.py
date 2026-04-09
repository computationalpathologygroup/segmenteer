from pathlib import Path

import numpy as np

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.utils import mask_to_geojson


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "sam3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class SAM3Segmenter(NumpySegmenter):
    """Tissue segmenter using SAM 3 (Segment Anything with Concepts).

    Supports both text-prompt concept segmentation (via SAM3SemanticPredictor)
    and visual-prompt / segment-everything mode (via the standard SAM interface).

    Note: SAM 3 weights (``sam3.pt``) are not automatically downloaded.
    You must first request access on https://huggingface.co/facebook/sam3 and
    then download ``sam3.pt`` into ``models/sam3/`` or the working directory.
    """

    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: int = 10,
        model_name: str = "sam3.pt",
        text_prompt: str | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | None = None,
        imgsz: int = 1024,
        *args,
        **kwargs,
    ):
        try:
            from ultralytics import SAM as _SAM  # noqa: F401
        except ImportError:
            raise ImportError(
                "ultralytics>=8.3.237 is required for SAM3Segmenter.\n"
                "Install with: pip install 'segmenteer[sam3]'"
            ) from None
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.model_name = model_name
        self.text_prompt = text_prompt
        self.conf = conf
        self.iou = iou
        self.device = device if device else "cuda" if self._cuda_available() else "cpu"
        self.imgsz = imgsz

        print(f"Initializing SAM 3 model: {model_name}")
        print(f"  Device: {self.device}")
        if text_prompt:
            print(f"  Text prompt: '{text_prompt}'")
        else:
            print("  Mode: visual / segment everything")

        self._model = None
        self._semantic_predictor = None
        self._load_model()

    def _cuda_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        model_path = get_model_cache_dir() / self.model_name

        if not model_path.exists():
            # Try working directory
            local_path = Path(self.model_name)
            if local_path.exists():
                model_path = local_path
            else:
                raise FileNotFoundError(
                    f"SAM 3 weights not found at '{model_path}' or '{local_path}'.\n"
                    "Request access at https://huggingface.co/facebook/sam3 and place "
                    f"'{self.model_name}' in models/sam3/ or the working directory."
                )

        if self.text_prompt:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=self.conf,
                task="segment",
                mode="predict",
                model=str(model_path),
                verbose=False,
            )
            self._semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
            print(f"Loaded SAM3SemanticPredictor from {model_path}")
        else:
            from ultralytics import SAM

            self._model = SAM(str(model_path))
            print(f"Loaded SAM 3 from {model_path}")

    @property
    def name(self) -> str:
        return f"sam3_{self.model_name.replace('.pt', '').lower()}"

    def _segment_numpy(self, image):
        if self.text_prompt:
            return self._segment_with_text(image)
        else:
            return self._segment_visual(image)

    def _segment_with_text(self, image):
        self._semantic_predictor.set_image(image)
        results = self._semantic_predictor(text=[self.text_prompt])

        if not results or len(results) == 0:
            return np.zeros(image.shape[:2], dtype=bool)

        result = results[0]
        if (
            not hasattr(result, "masks")
            or result.masks is None
            or len(result.masks) == 0
        ):
            return np.zeros(image.shape[:2], dtype=bool)

        masks = result.masks.data.cpu().numpy()
        combined_mask = masks.any(axis=0) if len(masks.shape) == 3 else masks
        return self._resize_mask(combined_mask.astype(bool), image.shape[:2])

    def _segment_visual(self, image):
        results = self._model.predict(
            source=image,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        if not results or len(results) == 0:
            return np.zeros(image.shape[:2], dtype=bool)

        result = results[0]
        if (
            not hasattr(result, "masks")
            or result.masks is None
            or len(result.masks) == 0
        ):
            return np.zeros(image.shape[:2], dtype=bool)

        masks = result.masks.data.cpu().numpy()
        combined_mask = masks.any(axis=0) if len(masks.shape) == 3 else masks
        return self._resize_mask(combined_mask.astype(bool), image.shape[:2])

    def _resize_mask(self, mask: np.ndarray, target_shape: tuple) -> np.ndarray:
        if mask.shape != target_shape:
            from skimage.transform import resize

            mask = resize(
                mask.astype(float),
                target_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(bool)
        return mask

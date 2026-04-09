from pathlib import Path

import numpy as np

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.utils import \
    mask_to_geojson  # noqa: F401 (kept for potential subclass use)


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "fastsam"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class FastSAMSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: int = 10,
        model_name: str = "FastSAM-x.pt",
        text_prompt: str | None = None,
        conf: float = 0.4,
        iou: float = 0.9,
        device: str | None = None,
        imgsz: int = 1024,
        *args,
        **kwargs,
    ):
        try:
            from ultralytics import FastSAM as _FastSAM  # noqa: F401
        except ImportError:
            raise ImportError(
                "ultralytics is required for FastSAMSegmenter.\n"
                "Install with: pip install 'segmenteer[fastsam]'"
            ) from None
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.model_name = model_name
        self.text_prompt = text_prompt
        self.conf = conf
        self.iou = iou
        self.device = device if device else "cuda" if self._cuda_available() else "cpu"
        self.imgsz = imgsz

        print(f"Initializing FastSAM model: {model_name}")
        print(f"  Device: {self.device}")
        if text_prompt:
            print(f"  Text prompt: '{text_prompt}'")
        else:
            print("  Mode: segment everything")

        self._model = None
        self._load_model()

    def _cuda_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        from ultralytics import FastSAM

        model_path = get_model_cache_dir() / self.model_name

        if model_path.exists():
            print(f"Loading FastSAM from {model_path}")
            self._model = FastSAM(str(model_path))
        else:
            print(f"Downloading FastSAM model: {self.model_name}")
            self._model = FastSAM(self.model_name)

            try:
                import shutil

                downloaded_path = Path(self.model_name)
                if downloaded_path.exists():
                    shutil.move(str(downloaded_path), str(model_path))
                    print(f"Saved model to {model_path}")
            except Exception as e:
                print(f"Note: Could not move model to cache dir: {e}")

    @property
    def name(self) -> str:
        return f"fastsam_{self.model_name.replace('.pt', '').lower()}"

    def _segment_numpy(self, image):
        if self.text_prompt:
            results = self._model(
                image,
                device=self.device,
                retina_masks=True,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                texts=self.text_prompt,
            )
        else:
            results = self._model(
                image,
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

        if len(masks.shape) == 3:
            combined_mask = masks.any(axis=0)
        else:
            combined_mask = masks

        combined_mask = combined_mask.astype(bool)

        if combined_mask.shape != image.shape[:2]:
            from skimage.transform import resize

            combined_mask = resize(
                combined_mask.astype(float),
                image.shape[:2],
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(bool)
        return combined_mask

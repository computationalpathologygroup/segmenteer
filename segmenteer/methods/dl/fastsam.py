from segmenteer.benchmark.profiling import measure_phase
from pathlib import Path
import numpy as np
from PIL import Image
from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import (
    find_local_model,
    get_method_model_dir,
    promote_to_method_cache,
)
from segmenteer.external_assets import require_network

_MODEL_NAMESPACE = "fastsam"


def get_model_cache_dir() -> Path:
    return get_method_model_dir(_MODEL_NAMESPACE)


class FastSAMSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float | None = 10,
        model_name: str = "FastSAM-x.pt",
        text_prompt: str | None = None,
        conf: float = 0.4,
        iou: float = 0.9,
        device: str | None = None,
        imgsz: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mpp is None:
            mpp = 10
        self.mpp = mpp
        self.model_name = model_name
        self.text_prompt = text_prompt
        self.conf = conf
        self.iou = iou
        self.device = resolve_torch_device(device)
        self.imgsz = imgsz
        self._model = None

    def _downloaded_checkpoint_path(self) -> Path | None:
        raw_path = getattr(self._model, "ckpt_path", None)
        if raw_path:
            candidate = Path(raw_path).expanduser()
            if candidate.is_file():
                return candidate
        candidate = Path(self.model_name)
        return candidate if candidate.is_file() else None

    def _load_model(self):
        try:
            from ultralytics import FastSAM
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for FastSAMSegmenter. Install with: uv sync --extra fastsam"
            ) from exc
        model_path = find_local_model(_MODEL_NAMESPACE, self.model_name)
        if model_path is not None:
            mode = (
                f"text prompt {self.text_prompt!r}"
                if self.text_prompt
                else "segment everything"
            )
            print(
                f"Loading FastSAM weights from {model_path} · device={self.device} · {mode}",
                flush=True,
            )
            self._model = FastSAM(str(model_path))
            return
        mode = (
            f"text prompt {self.text_prompt!r}"
            if self.text_prompt
            else "segment everything"
        )
        require_network(
            "FastSAM Ultralytics-managed download (unversioned)", pinned=False
        )
        print(
            f"Loading FastSAM model {self.model_name} (download if needed) · device={self.device} · {mode}",
            flush=True,
        )
        self._model = FastSAM(self.model_name)
        downloaded_path = self._downloaded_checkpoint_path()
        if downloaded_path is not None:
            model_path = promote_to_method_cache(
                downloaded_path, _MODEL_NAMESPACE, self.model_name
            )
            print(f"Saved FastSAM weights to {model_path}")

    @property
    def name(self) -> str:
        return f"fastsam_{self.model_name.replace('.pt', '').lower()}"

    def _segment_numpy(self, image):
        if self._model is None:
            with measure_phase("model_initialization"):
                self._load_model()
        rgb_image = Image.fromarray(image)
        if self.text_prompt:
            results = self._model(
                rgb_image,
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
                rgb_image,
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

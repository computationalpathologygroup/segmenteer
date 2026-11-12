"""GrandQC tissue detection using the published patchwise inference pipeline.

Uses JPEG quality 80, 512×512 tiles, and the trained 10 µm/px resolution.
Source: https://github.com/cpath-ukk/grandqc/blob/main/01_WSI_inference_OPENSLIDE_QC/wsi_tis_detect.py
"""

from pathlib import Path
import numpy as np
from segmenteer.benchmark.profiling import measure_phase
from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import find_local_model, get_method_model_dir
from segmenteer.external_assets import atomic_https_download

try:
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def get_model_cache_dir() -> Path:
    return get_method_model_dir("grandqc")


class GrandQCPatchwiseSegmenter(NumpySegmenter):
    MODEL_FILE = "Tissue_Detection_MPP10.pth"
    ZENODO_RECORD_ID = "14507273"
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float | None = 10,
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        jpeg_quality: int = 80,
        tile_size: int = 512,
        batch_size: int = 4,
        *args,
        **kwargs,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "GrandQC Patchwise requires torch, torchvision, and Pillow. Install with: uv sync --extra grandqc"
            )
        mpp = 10.0 if mpp is None else float(mpp)
        if abs(mpp - 10.0) > 1e-6:
            raise ValueError(
                f"GrandQC Patchwise checkpoint Tissue_Detection_MPP10.pth is trained at 10 um/px; got mpp={mpp}. "
                "Do not apply a shared --mpp override to this model."
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not (1 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be between 1 and 100.")
        if tile_size != 512:
            raise ValueError("GrandQC was trained and published with 512x512 tiles.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        super().__init__(*args, mpp=mpp, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = float(confidence_threshold)
        self.jpeg_quality = int(jpeg_quality)
        self.tile_size = int(tile_size)
        self.batch_size = int(batch_size)
        self.device = torch.device(resolve_torch_device(device))
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
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError("GrandQC requires segmentation_models_pytorch.") from exc
        self._model = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None,
        )
        if self.checkpoint_path:
            checkpoint_file = Path(self.checkpoint_path).expanduser()
        else:
            checkpoint_file = find_local_model("grandqc", self.MODEL_FILE)
            if checkpoint_file is None:
                checkpoint_file = self._download_from_zenodo(
                    self.MODEL_FILE, get_model_cache_dir()
                )
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

        self._model.load_state_dict(state_dict, strict=True)
        self._model = self._model.to(self.device).eval()
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
        return f"grandqc_patchwise_mpp{self.mpp:g}_jpeg{self.jpeg_quality}_tile{self.tile_size}"

    @staticmethod
    def _jpeg_recompress_rgb(image: np.ndarray, quality: int) -> np.ndarray:
        # Match the reference RGB-to-JPEG-to-BGR preprocessing path.
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise RuntimeError("Could not JPEG-encode GrandQC input")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Could not JPEG-decode GrandQC input")
        return decoded

    def _preprocess_image(self, image: np.ndarray):
        """Return a CHW tensor for one 512x512 JPEG-decoded patch."""
        return self._transform(PILImage.fromarray(image))

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if self._model is None:
            with measure_phase("model_initialization"):
                self._load_model()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected HxWx3 uint8 image, got {image.shape}, {image.dtype}."
            )
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("GrandQC Patchwise requires nonempty images.")
        size = self.tile_size
        image = self._jpeg_recompress_rgb(image, self.jpeg_quality)
        output = np.zeros((height, width), dtype=np.bool_)

        tiles = []
        positions = []

        def flush():
            if not tiles:
                return
            tensor = torch.stack(tiles).to(self.device)
            with torch.inference_mode():
                logits = self._model(tensor)
                # Preserve class-0 tissue selection and argmax tie behavior.
                if self.confidence_threshold == 0.5:
                    tissue = logits.argmax(dim=1) == 0
                else:
                    tissue = (
                        torch.softmax(logits, dim=1)[:, 0] > self.confidence_threshold
                    )
                masks = tissue.cpu().numpy()
            for (y, x, y0, x0, crop_height, crop_width), mask in zip(positions, masks):
                output[y : y + crop_height, x : x + crop_width] = mask[
                    y - y0 : y - y0 + crop_height,
                    x - x0 : x - x0 + crop_width,
                ]
            tiles.clear()
            positions.clear()

        for y in range(0, height, size):
            y0 = max(0, min(y, height - size))
            for x in range(0, width, size):
                x0 = max(0, min(x, width - size))
                tile = image[y0 : y0 + size, x0 : x0 + size]
                # Pad small thumbnails with white background.
                pad_h, pad_w = size - tile.shape[0], size - tile.shape[1]
                if pad_h or pad_w:
                    tile = np.pad(
                        tile,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="constant",
                        constant_values=255,
                    )
                tiles.append(self._preprocess_image(tile))
                positions.append(
                    (y, x, y0, x0, min(size, height - y), min(size, width - x))
                )
                if len(tiles) == self.batch_size:
                    flush()
        flush()
        return output

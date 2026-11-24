"""CxTissueSeg ONNX tissue segmentation at 40 µm/px.

Model: https://huggingface.co/conflux-xyz/cx-tissue-seg (Apache-2.0).
Tiling: https://github.com/conflux-xyz/conflux-segmentation
Uses RGB 512×512 tiles with white padding; dependencies load lazily.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from segmenteer.benchmark.profiling import measure_phase
from segmenteer.core.base import NumpySegmenter
from segmenteer.model_cache import get_method_model_dir
from segmenteer.external_assets import require_network


class CxTissueSegSegmenter(NumpySegmenter):
    """Run the official U-Net/MobileNetV3-Small float32 ONNX model.

    Pinned weights are cached on first inference. ``model_path`` accepts an
    explicit local checkpoint; custom revisions use separate caches.
    """

    APPLY_TO_GRAYSCALE = False
    MODEL_MPP = 40.0
    TILE_SIZE = 512
    MODEL_REPO = "conflux-xyz/cx-tissue-seg"
    MODEL_FILENAME = "model.onnx"
    # Use the published float32 checkpoint at an immutable revision.
    MODEL_REVISION = "ae2713f49cc55333d6f446b9f37931197be8192a"
    MODEL_SHA256 = "3a30565a64ebcee8fbac11eb794a534283965236d59961d08351329ed6061893"

    def __init__(
        self,
        mpp: float | None = 40.0,
        device: str | None = "cpu",
        batch_size: int = 1,
        model_path: str | Path | None = None,
        revision: str | None = None,
        auto_download: bool = True,
        *args,
        **kwargs,
    ) -> None:
        mpp = self.MODEL_MPP if mpp is None else float(mpp)
        if not np.isfinite(mpp) or abs(mpp - self.MODEL_MPP) > 1e-6:
            raise ValueError(
                f"CxTissueSeg pretrained weights require mpp=40; received {mpp!r}"
            )
        if device is None:
            device = "cpu"
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(
                'CxTissueSeg ONNX supports device="cpu", "cuda", or "auto". '
                'Use "cpu" on macOS (including Apple Silicon).'
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("revision must be a nonempty string or None")
        super().__init__(*args, mpp=mpp, **kwargs)
        self.device = device
        self.batch_size = batch_size
        self.model_path = (
            str(Path(model_path).expanduser()) if model_path is not None else None
        )
        self.revision = revision
        self.auto_download = bool(auto_download)
        self._segmenter = None
        self._onnx_session = None

    @property
    def name(self) -> str:

        return "cx_tissue_seg_mpp40"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _cache_dir(self) -> Path:
        root = get_method_model_dir("cx-tissue-seg")
        # Keep custom revisions separate from the default checkpoint.
        if self.revision is not None and self.revision != self.MODEL_REVISION:
            digest = hashlib.sha256(self.revision.encode("utf8")).hexdigest()[:16]
            return root / "revisions" / digest
        return root

    def _resolve_model_path(self) -> Path:
        if self.model_path is not None:
            path = Path(self.model_path).expanduser()
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(
                    f"CxTissueSeg ONNX model not found or empty: {path}"
                )
            return path

        root = self._cache_dir()
        candidate = root / self.MODEL_FILENAME
        pinned = self.revision is None or self.revision == self.MODEL_REVISION
        if candidate.is_file() and candidate.stat().st_size:
            if not pinned or self._sha256(candidate) == self.MODEL_SHA256:
                return candidate
    
            if not self.auto_download:
                raise ValueError(
                    f"CxTissueSeg cached checkpoint SHA256 mismatch: {candidate}"
                )
        elif not self.auto_download:
            raise FileNotFoundError(
                f"CxTissueSeg weights missing: {candidate}. "
                "Enable auto_download or pass model_path."
            )
        if not self.auto_download:
            raise FileNotFoundError(f"CxTissueSeg weights missing: {candidate}")

        import re

        require_network(
            "CxTissueSeg Hugging Face checkpoint",
            pinned=pinned
            or bool(self.revision and re.fullmatch(r"[0-9a-fA-F]{40}", self.revision)),
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "CxTissueSeg automatic download requires huggingface_hub. "
                "Install with: pip install huggingface_hub"
            ) from exc
        root.mkdir(parents=True, exist_ok=True)
        # Evict a failed integrity check before retrying the hub download.
        if (
            candidate.exists()
            and pinned
            and self._sha256(candidate) != self.MODEL_SHA256
        ):
            candidate.unlink()
        downloaded = Path(
            hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILENAME,
                revision=self.revision or self.MODEL_REVISION,
                local_dir=str(root),
            )
        )
        if not downloaded.is_file() or not downloaded.stat().st_size:
            raise RuntimeError(
                f"CxTissueSeg download returned an empty checkpoint: {downloaded}"
            )
        if pinned and self._sha256(downloaded) != self.MODEL_SHA256:
            raise RuntimeError(
                f"CxTissueSeg downloaded model SHA256 mismatch: {downloaded}. "
                "Refusing to use unexpected weights."
            )
        return downloaded

    def _providers(self, ort) -> list[str]:
        available = ort.get_available_providers()
        if self.device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CxTissueSeg requested CUDA, but CUDAExecutionProvider is unavailable. "
                    'Install onnxruntime-gpu with CUDA support, or use device="cpu".'
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "auto" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                "CxTissueSeg requires ONNX Runtime CPUExecutionProvider."
            )
        return ["CPUExecutionProvider"]

    def _load_model(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "CxTissueSeg requires ONNX Runtime. Install: pip install onnxruntime"
            ) from exc
        try:
            from conflux_segmentation import Segmenter
        except ImportError as exc:
            raise ImportError(
                "CxTissueSeg requires official tiled inference. "
                'Install: pip install "conflux-segmentation[onnx]>=0.5.0"'
            ) from exc
        model_path = self._resolve_model_path()
        session = ort.InferenceSession(str(model_path), providers=self._providers(ort))
        # Preserve published preprocessing and tiling parameters.
        self._segmenter = Segmenter.from_onnx(
            session,
            activation="sigmoid",
            tile_size=self.TILE_SIZE,
            overlap=0.125,
            blend_mode="gaussian",
            pad_value=255,
            batch_size=self.batch_size,
        )
        self._onnx_session = session

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"CxTissueSeg requires H×W×3 uint8 RGB, got {image.shape} {image.dtype}"
            )
        if not image.shape[0] or not image.shape[1]:
            raise ValueError("CxTissueSeg image dimensions must be nonzero")
        if self._segmenter is None:
            with measure_phase("model_initialization"):
                self._load_model()
        # Positive logits correspond to tissue.
        mask = np.asarray(self._segmenter(image).to_binary().get_mask())
        if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
            raise ValueError(
                f"CxTissueSeg returned invalid binary mask {mask.shape} {mask.dtype}; "
                f"expected {image.shape[:2]} bool"
            )
        return mask

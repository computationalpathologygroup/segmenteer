"""NKI All-Stains tissue segmentation using published TorchScript weights.

Model: https://huggingface.co/NKI-AI/tissue-bg-all-stains
Uses RGB at 12 µm/px, 1024×1024 tiles with 128-pixel overlap, and RGB/255
normalization. Class 1 denotes tissue. Model weights are not bundled.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from segmenteer.benchmark.profiling import measure_phase
from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import find_local_model, get_method_model_dir
from segmenteer.external_assets import require_network


class NKIAllStainsSegmenter(NumpySegmenter):
    """Run the NKI All-Stains binary tissue model.

    Model assets are loaded on first inference; ``model_path`` accepts a local
    ``.pack`` archive. Pin ``revision`` for reproducible downloads.
    """

    APPLY_TO_GRAYSCALE = False
    MODEL_REPO = "NKI-AI/tissue-bg-all-stains"
    MODEL_FILENAME = "tissue_background_all_stains.pack"
    TILE_SIZE = 1024
    TILE_OVERLAP = 128
    MODEL_MPP = 12.0
    TISSUE_CLASS = 1

    def __init__(
        self,
        mpp: float | None = 12.0,
        model_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 1,
        revision: str | None = None,
        auto_download: bool = True,
        *args,
        **kwargs,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "NKI All-Stains requires PyTorch. Install torch in your Segmenteer environment."
            ) from exc
        mpp = self.MODEL_MPP if mpp is None else float(mpp)
        if not np.isfinite(mpp) or abs(mpp - self.MODEL_MPP) > 1e-6:
            raise ValueError(
                f"NKI All-Stains is trained at {self.MODEL_MPP:g} um/px; "
                f"received mpp={mpp:g}. Do not override its native MPP."
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        super().__init__(*args, mpp=mpp, **kwargs)
        self.model_path = (
            str(Path(model_path).expanduser()) if model_path is not None else None
        )
        self.device = torch.device(resolve_torch_device(device))
        self.batch_size = batch_size
        self.revision = revision
        self.auto_download = bool(auto_download)
        self._model = None

    @property
    def name(self) -> str:
        return "nki_all_stains_mpp12"

    @classmethod
    def _tile_starts(cls, length: int) -> list[int]:
        """Generate tile origins with far-edge alignment."""
        if length <= cls.TILE_SIZE:
            return [0]
        last = length - cls.TILE_SIZE
        starts = list(range(0, last + 1, cls.TILE_SIZE - cls.TILE_OVERLAP))
        if starts[-1] != last:
            starts.append(last)
        return starts

    @classmethod
    def _owned_spans(cls, length: int) -> list[tuple[int, int, int]]:
        """Partition tile overlaps at their midpoints for crop merging."""
        starts = cls._tile_starts(length)
        spans = []
        for index, start in enumerate(starts):
            left = 0 if index == 0 else (starts[index - 1] + cls.TILE_SIZE + start) // 2
            right = (
                length
                if index == len(starts) - 1
                else (start + cls.TILE_SIZE + starts[index + 1]) // 2
            )
            spans.append((start, left, right))
        return spans

    def _resolve_model_path(self) -> Path:
        if self.model_path is not None:
            model_file = Path(self.model_path)
            if not model_file.is_file():
                raise FileNotFoundError(f"NKI All-Stains .pack not found: {model_file}")
            return model_file
        # Do not reuse unversioned weights for an explicit revision.
        if self.revision is None:
            existing = find_local_model("nki-all-stains", self.MODEL_FILENAME)
            if existing is not None:
                return existing
        if not self.auto_download:
            raise FileNotFoundError(
                f"NKI All-Stains model is missing. Put {self.MODEL_FILENAME} in "
                f"{get_method_model_dir('nki-all-stains')} or pass model_path."
            )
        import re

        require_network(
            "NKI All-Stains Hugging Face checkpoint",
            pinned=bool(
                self.revision and re.fullmatch(r"[0-9a-fA-F]{40}", self.revision)
            ),
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Automatic NKI All-Stains model download requires huggingface_hub; "
                "install it or pass model_path to the official .pack."
            ) from exc
        return Path(
            hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILENAME,
                revision=self.revision,
                local_dir=str(get_method_model_dir("nki-all-stains")),
            )
        )

    def _load_model(self) -> None:
        import torch

        model_path = self._resolve_model_path()
        try:
            with zipfile.ZipFile(model_path) as package:
                # Official model packages contain model.pt and model_config.xml.
                relevant = {"model.pt": [], "model_config.xml": []}
                for member in package.infolist():
                    if Path(member.filename).name in relevant and not member.is_dir():
                        relevant[Path(member.filename).name].append(member)
                if any(len(group) != 1 for group in relevant.values()):
                    raise ValueError(
                        "NKI .pack requires one unambiguous model.pt and model_config.xml."
                    )
                # Limit uncompressed size before reading ZIP members into memory.
                limits = {"model.pt": 2 * 1024**3, "model_config.xml": 2 * 1024**2}
                for name, group in relevant.items():
                    info = group[0]
                    if not 0 < info.file_size <= limits[name]:
                        raise ValueError(f"NKI .pack contains invalid-size {name}.")
                    if (
                        info.compress_size < 1
                        or info.file_size > info.compress_size * 1000
                    ):
                        raise ValueError(
                            f"NKI .pack contains suspicious compression ratio: {name}"
                        )
                members = {name: group[0].filename for name, group in relevant.items()}
                if "model.pt" not in members or "model_config.xml" not in members:
                    raise ValueError(
                        "Expected NKI .pack with model.pt and model_config.xml."
                    )
                # Validate package metadata before loading the model.
                import xml.etree.ElementTree as ET

                try:
                    ET.fromstring(package.read(members["model_config.xml"]))
                except ET.ParseError as exc:
                    raise ValueError(
                        "NKI .pack contains invalid model_config.xml."
                    ) from exc
                scripted = package.read(members["model.pt"])
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Not an NKI model .pack ZIP: {model_path}") from exc
        # TorchScript packages must come from a trusted source.
        self._model = torch.jit.load(io.BytesIO(scripted), map_location=self.device)
        self._model.eval()

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        import torch

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"NKI All-Stains expects HxWx3 uint8 RGB, got {image.shape} {image.dtype}."
            )
        height, width = image.shape[:2]
        if height < 1 or width < 1:
            raise ValueError("NKI All-Stains requires a nonempty image.")
        if self._model is None:
            with measure_phase("model_initialization"):
                self._load_model()

        size = self.TILE_SIZE
        output = np.zeros((height, width), dtype=np.bool_)
        batch: list = []
        regions: list[tuple[int, int, int, int, int, int]] = []

        def flush() -> None:
            if not batch:
                return
            inputs = torch.stack(batch).to(self.device)
            with torch.inference_mode():
                logits = self._model(inputs)
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
                    len(batch),
                    2,
                    size,
                    size,
                ):
                    raise ValueError(
                        f"NKI model must return logits [batch, 2, {size}, {size}]; "
                        f"got {getattr(logits, 'shape', type(logits))}."
                    )
                predicted = (logits.argmax(dim=1) == self.TISSUE_CLASS).cpu().numpy()
            for (y_start, x_start, y_left, y_right, x_left, x_right), mask in zip(
                regions, predicted
            ):
                output[y_left:y_right, x_left:x_right] = mask[
                    y_left - y_start : y_right - y_start,
                    x_left - x_start : x_right - x_start,
                ]
            batch.clear()
            regions.clear()

        for y_start, y_left, y_right in self._owned_spans(height):
            for x_start, x_left, x_right in self._owned_spans(width):
                tile = np.full((size, size, 3), 255, dtype=np.uint8)
                subimage = image[y_start : y_start + size, x_start : x_start + size]
                tile[: subimage.shape[0], : subimage.shape[1], :] = subimage
                # Match the published RGB/255 normalization without stain adjustment.
                tensor = (
                    torch.from_numpy(tile.transpose(2, 0, 1).copy()).float().div_(255.0)
                )
                batch.append(tensor)
                regions.append((y_start, x_start, y_left, y_right, x_left, x_right))
                if len(batch) == self.batch_size:
                    flush()
        flush()
        return output

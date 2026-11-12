"""ICGI nnU-Net v2 tissue segmenters at 10 µm/px.

Official weights are downloaded lazily to ``SEGMENTEER_MODELS_DIR/icgi``;
the customized nnU-Net fork must be installed separately.
Source: https://github.com/icgi/Reliable-and-Efficient-Tissue-Segmentation-in-Whole-Slide-Images
Weights: CC BY-NC 4.0 (upstream LICENSE.WEIGHTS).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from segmenteer.core.base import NumpySegmenter
from segmenteer.model_cache import get_method_model_dir
from segmenteer.external_assets import require_network, sha256

ICGI_GDRIVE_FILE_ID = "12jdV2BTT0sZX5SciBNNuwT_ZBQhITKdQ"
ICGI_GDRIVE_URL = f"https://drive.google.com/file/d/{ICGI_GDRIVE_FILE_ID}/view"
_ASSET_MANIFEST = "icgi_asset_hashes.json"
_REQUIRED = {
    "trained_on_10um": ("checkpoint_10um.pth", "plans.json", "dataset.json"),
    "trained_on_10um_ResEnc": (
        "checkpoint_10um_ResEnc.pth",
        "plans.json",
        "dataset.json",
    ),
}


def _missing_model_files(root: Path, variant: str) -> list[str]:
    """List missing or invalid model files."""
    model_dir = root / variant
    missing = [
        name
        for name in _REQUIRED[variant]
        if not (model_dir / name).is_file() or not (model_dir / name).stat().st_size
    ]
    manifest_path = root / _ASSET_MANIFEST
    if not missing and manifest_path.is_file():
        # Validate managed caches while allowing manually populated caches.
        try:
            hashes = json.loads(manifest_path.read_text(encoding="utf-8"))["files"][
                variant
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("ICGI cache provenance manifest is invalid.") from exc
        missing.extend(
            name
            for name in _REQUIRED[variant]
            if hashes.get(name) != sha256(model_dir / name)
        )
    return missing


@contextmanager
def _download_lock(root: Path):
    """Serialize access to the shared model cache."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _find_archive_entry(
    archive: zipfile.ZipFile, variant: str, filename: str
) -> zipfile.ZipInfo:
    matches = []
    for info in archive.infolist():
        # Accept the expected variant/file layout under optional parent folders.
        parts = PurePosixPath(info.filename).parts
        if len(parts) >= 2 and parts[-2:] == (variant, filename) and not info.is_dir():
            if any(part in ("..", ".") for part in parts) or info.filename.startswith(
                "/"
            ):
                continue

            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                continue
            matches.append(info)
    if len(matches) != 1:
        raise ValueError(
            f"ICGI archive must contain exactly one {variant}/{filename}; found {len(matches)}"
        )
    return matches[0]


def _extract_official_models(archive_path: Path, destination: Path) -> None:
    """Extract only the required model assets from the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        members = {
            (variant, filename): _find_archive_entry(archive, variant, filename)
            for variant, names in _REQUIRED.items()
            for filename in names
        }
        if sum(info.file_size for info in members.values()) > 16 * 1024**3:
            raise ValueError("ICGI ZIP exceeds 16 GiB total uncompressed model budget")
        with tempfile.TemporaryDirectory(
            prefix=".icgi-extract-", dir=destination
        ) as directory:
            stage = Path(directory)
            for (variant, filename), info in members.items():
                # Limit extracted file sizes to 8 GiB per asset.
                if not 0 < info.file_size <= 8 * 1024**3:
                    raise ValueError(
                        f"Invalid size for ICGI model asset: {variant}/{filename}"
                    )
                target = stage / variant / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if (
                    info.compress_size < 1
                    or info.file_size > max(1, info.compress_size) * 1000
                ):
                    raise ValueError(
                        f"ICGI ZIP suspicious compression ratio: {variant}/{filename}"
                    )
                copied = 0
                with archive.open(info) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > info.file_size:
                            raise ValueError(
                                "ICGI archive entry exceeded declared size"
                            )
                        output.write(chunk)
                if copied != info.file_size:
                    raise ValueError("ICGI archive entry did not match declared size")

            for variant in _REQUIRED:
                if _missing_model_files(stage, variant):
                    raise ValueError(f"Incomplete downloaded ICGI archive: {variant}")
            manifest = {
                "source": ICGI_GDRIVE_URL,
                "archive_sha256": sha256(archive_path),
                "files": {
                    variant: {name: sha256(stage / variant / name) for name in names}
                    for variant, names in _REQUIRED.items()
                },
            }
            for variant, files in _REQUIRED.items():
                target_dir = destination / variant
                target_dir.mkdir(parents=True, exist_ok=True)
                for name in files:
                    os.replace(stage / variant / name, target_dir / name)
            # Record content hashes for subsequent cache verification.
            temp_manifest = stage / _ASSET_MANIFEST
            temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temp_manifest, destination / _ASSET_MANIFEST)


def get_icgi_model_dir(variant: str, *, auto_download: bool = True) -> Path:
    """Resolve the cached model directory, downloading missing assets on use."""
    if variant not in _REQUIRED:
        raise ValueError(f"Unsupported ICGI model variant: {variant}")
    root = get_method_model_dir("icgi")
    missing = _missing_model_files(root, variant)
    if not missing:
        return root / variant
    if not auto_download:
        raise FileNotFoundError(
            f"ICGI model files missing under {root / variant}: {', '.join(missing)}"
        )

    with _download_lock(root):
        if not _missing_model_files(root, variant):
            return root / variant
        require_network(
            "ICGI Google Drive checkpoint ZIP (no published SHA-256)", pinned=False
        )
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                "ICGI model weights are missing. Install gdown once using "
                "`pip install gdown` to enable automatic downloading, "
                f"or manually extract {ICGI_GDRIVE_URL} into {root}."
            ) from exc
        print(
            f"Downloading official ICGI nnU-Net weights into {root} (first use only)..."
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=".icgi-download-", dir=root
            ) as directory:
                zip_path = Path(directory) / "models.zip"
                result = gdown.download(
                    id=ICGI_GDRIVE_FILE_ID, output=str(zip_path), quiet=False
                )
                if not result or not zip_path.is_file():
                    raise RuntimeError("gdown did not produce models.zip")
                if (
                    not zip_path.stat().st_size
                    or zip_path.stat().st_size > 16 * 1024**3
                ):
                    raise RuntimeError("ICGI downloaded ZIP has an unacceptable size.")
                _extract_official_models(zip_path, root)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download or extract the ICGI models from {ICGI_GDRIVE_URL}. "
                f"You may manually place the official model folders in {root}. "
                f"Details: {exc}"
            ) from exc
        missing = _missing_model_files(root, variant)
        if missing:
            raise FileNotFoundError(
                f"ICGI archive lacks {variant}: {', '.join(missing)}"
            )
    return root / variant


class _ICGIBaseSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False
    TRAINING_MPP = 10.0
    RESENC = False
    MODEL_FOLDER = "trained_on_10um"
    CHECKPOINT = "checkpoint_10um.pth"

    def __init__(
        self,
        mpp: float = 10.0,
        upstream_dir: str | Path | None = None,
        executable: str = "nnUNetv2_predict_tissue",
        device: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        mpp = float(mpp)
        if not np.isfinite(mpp) or abs(mpp - self.TRAINING_MPP) > 1e-6:
            raise ValueError(f"ICGI 10 µm/px weights require mpp=10; received {mpp!r}")
        super().__init__(*args, mpp=mpp, **kwargs)
        # Retain serialized parameters for existing experiment identities.
        self.upstream_dir = (
            str(Path(upstream_dir).expanduser().resolve())
            if upstream_dir is not None
            else None
        )
        self.executable = str(executable)
        if not self.executable.strip():
            raise ValueError("executable must be nonempty")
        if device is not None:
            if device not in ("auto", "cpu", "cuda", "mps"):
                raise ValueError("device must be auto, cpu, cuda, or mps")
            self.device = device

    def _resolve_model_dir(self) -> Path:
        # Support configurations referencing a local upstream checkout.
        root = self.upstream_dir or os.environ.get("SEGMENTEER_ICGI_HOME")
        if root is not None:
            home = Path(root).expanduser().resolve()
            model_dir = home / "models" / self.MODEL_FOLDER
            missing = _missing_model_files(home / "models", self.MODEL_FOLDER)
            if missing:
                raise FileNotFoundError(
                    f"ICGI model assets not found under {model_dir}: {', '.join(missing)}"
                )
            return model_dir
        return get_icgi_model_dir(self.MODEL_FOLDER)

    def _resolve_executable(self) -> str:
        found = shutil.which(self.executable)
        if found is not None:
            return found
        candidate = Path(self.executable).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(
            f"Could not find custom ICGI executable: {self.executable}"
        )

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"ICGI requires an RGB H×W×3 image, received {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"ICGI requires uint8 RGB input, received {image.dtype}")
        model_dir = self._resolve_model_dir()
        with tempfile.TemporaryDirectory(prefix="segmenteer-icgi-") as work:
            input_dir, output_dir = Path(work) / "inputs", Path(work) / "outputs"
            input_dir.mkdir()
            output_dir.mkdir()
            Image.fromarray(image, mode="RGB").save(input_dir / "slide.png")
            # Preserve the configurable CLI; default inference uses local assets.
            if self.executable != "nnUNetv2_predict_tissue":
                cmd = [
                    self._resolve_executable(),
                    "-i",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                ]
                if self.RESENC:
                    cmd.append("--resenc")
                cwd = Path(self.upstream_dir) if self.upstream_dir else model_dir.parent
            else:
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("_icgi_predict.py")),
                    "--model-dir",
                    str(model_dir),
                    "--checkpoint",
                    self.CHECKPOINT,
                    "-i",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--device",
                    getattr(self, "device", "auto"),
                ]
                cwd = None
            completed = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                detail = (completed.stderr or completed.stdout).strip()[-3000:]
                message = (
                    f"ICGI inference failed (exit={completed.returncode}): {detail}"
                )
                if "Weights only load failed" in detail or "WeightsUnpickler" in detail:
                    message += (
                        "\nThe checkpoint contains objects not supported by PyTorch's restricted "
                        "loader. This is a checkpoint compatibility problem, not a missing "
                        "ICGI installation. The checkpoint was not modified. "
                        "Please report the unsupported global/type for a safe compatibility fix."
                    )
                raise RuntimeError(message)
            predictions = sorted(output_dir.rglob("*.png"))
            if len(predictions) != 1:
                raise RuntimeError(
                    f"Expected exactly one PNG tissue mask from ICGI; found {len(predictions)} under {output_dir}."
                )
            with Image.open(predictions[0]) as opened:
                predicted = np.asarray(opened)
            if predicted.ndim == 3:
                if predicted.shape[-1] in (3, 4) and np.all(
                    predicted[..., :3] == predicted[..., :1]
                ):
                    predicted = predicted[..., 0]
                else:
                    raise ValueError(
                        f"Unexpected RGB mask format from ICGI: {predicted.shape}"
                    )
            if predicted.shape != image.shape[:2]:
                raise ValueError(
                    f"ICGI mask shape {predicted.shape} differs from input {image.shape[:2]}"
                )
            labels = set(np.unique(predicted).tolist())
            if not labels.issubset({0, 1, 255}):
                raise ValueError(f"Unexpected ICGI mask values: {sorted(labels)[:20]}")
            return predicted > 0


class ICGINnUNetV2Segmenter(_ICGIBaseSegmenter):
    """Official nnU-Net v2, trained at 10 µm/px (standard encoder)."""

    @property
    def name(self) -> str:
        return "icgi_nnunetv2_mpp10"


class ICGIResEncSegmenter(_ICGIBaseSegmenter):
    """Official nnU-Net v2 Residual Encoder, trained at 10 µm/px."""

    RESENC = True
    MODEL_FOLDER = "trained_on_10um_ResEnc"
    CHECKPOINT = "checkpoint_10um_ResEnc.pth"

    @property
    def name(self) -> str:
        return "icgi_resenc_mpp10"

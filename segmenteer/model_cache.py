from __future__ import annotations
import os
import shutil
import tempfile
from segmenteer.external_assets import _file_lock
from pathlib import Path

MODEL_ROOT_ENV = "SEGMENTEER_MODELS_DIR"


def get_models_root() -> Path:
    configured = os.environ.get(MODEL_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path(__file__).resolve().parents[1] / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_method_model_dir(method: str) -> Path:
    if not method or Path(method).name != method:
        raise ValueError("method must be a simple directory name")
    directory = get_models_root() / method
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_local_model(method: str, filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a simple file name")
    root = get_models_root()
    method_dir = get_method_model_dir(method)
    candidates = [method_dir / filename, root / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    nested = sorted(method_dir.glob(f"models--*/snapshots/*/{filename}"))
    for candidate in reversed(nested):
        if candidate.is_file():
            return candidate
    legacy = sorted(method_dir.rglob(filename))
    for candidate in reversed(legacy):
        if candidate.is_file():
            return candidate
    return None


def promote_to_method_cache(source: str | Path, method: str, filename: str) -> Path:
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {source_path}")
    source_path = source_path.resolve()
    target = get_method_model_dir(method) / filename
    with _file_lock(target.with_name(target.name + ".lock")):
        if target.is_file() and target.stat().st_size > 0:
            return target
        if not source_path.stat().st_size:
            raise ValueError(f"Cannot cache an empty model file: {source_path}")
        fd, name = tempfile.mkstemp(
            prefix="." + target.name + ".", suffix=".part", dir=target.parent
        )
        os.close(fd)
        stage = Path(name)
        try:
            shutil.copyfile(source_path, stage)
            if stage.stat().st_size != source_path.stat().st_size:
                raise RuntimeError("Model cache copy was incomplete.")
            os.replace(stage, target)
        finally:
            stage.unlink(missing_ok=True)
        return target

"""Small, dependency-free helpers for deterministic local model-weight storage.

All built-in segmenters resolve weights beneath one shared directory.  By
default this is ``<project>/models``; set ``SEGMENTEER_MODELS_DIR`` before
importing a segmenter to use a different location.  A downloaded Hugging Face
artifact is promoted into the method's canonical path once, so later runs can
load it directly without a network/cache lookup.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

MODEL_ROOT_ENV = "SEGMENTEER_MODELS_DIR"


def get_models_root() -> Path:
    """Return the configured shared models directory, creating it if needed."""
    configured = os.environ.get(MODEL_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
    else:
        # ``segmenteer/model_cache.py`` -> project root is two parents up.
        root = Path(__file__).resolve().parents[1] / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_method_model_dir(method: str) -> Path:
    """Return a segmenter's canonical subdirectory below the shared model root."""
    if not method or Path(method).name != method:
        raise ValueError("method must be a simple directory name")
    directory = get_models_root() / method
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_local_model(method: str, filename: str) -> Path | None:
    """Find an existing weight file without performing any network access.

    The canonical layout is ``<models>/<method>/<filename>``.  The project also
    recognises ``<models>/<filename>`` for existing local installations and the
    nested Hugging Face cache layout produced by older releases.
    """
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a simple file name")

    root = get_models_root()
    method_dir = get_method_model_dir(method)
    candidates = [
        method_dir / filename,
        root / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Older HEST builds passed the method directory directly to Hugging Face as
    # ``cache_dir``.  The hub client then stored files under
    # ``models--<repo>/snapshots/<revision>/``.  Recognise that cache so users
    # do not have to download weights again merely to migrate releases.
    nested = sorted(method_dir.glob(f"models--*/snapshots/*/{filename}"))
    for candidate in reversed(nested):
        if candidate.is_file():
            return candidate

    # A few historical adapters kept upstream repository subdirectories, e.g.
    # ``models/pathprofiler/PathProfiler/<checkpoint>``.  Search only the
    # selected method directory so this compatibility fallback stays bounded.
    legacy = sorted(method_dir.rglob(filename))
    for candidate in reversed(legacy):
        if candidate.is_file():
            return candidate
    return None


def promote_to_method_cache(source: str | Path, method: str, filename: str) -> Path:
    """Make *source* available at the canonical local path for later runs.

    A metadata-preserving copy is made once after a download.  Keeping the
    canonical file independent from an upstream cache protects it from cache
    cleanup or replacement.  Existing canonical files are never overwritten.
    """
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {source_path}")
    source_path = source_path.resolve()

    target = get_method_model_dir(method) / filename
    if target.is_file():
        return target
    shutil.copy2(source_path, target)
    return target

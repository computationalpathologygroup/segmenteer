"""Network policy and integrity checks for external model assets.

Downloads are deferred until inference. ``SEGMENTEER_OFFLINE=1`` blocks
managed downloads; ``SEGMENTEER_ALLOW_UNPINNED_DOWNLOADS=1`` permits
assets without an immutable revision or verified checksum.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.getenv(name, "").lower().strip() in _TRUE


def offline() -> bool:
    return _flag("SEGMENTEER_OFFLINE")


def require_network(asset: str, *, pinned: bool = True) -> None:
    if offline():
        raise RuntimeError(
            f"{asset}: network access disabled by SEGMENTEER_OFFLINE=1. "
            "Pre-populate the model cache or supply a local checkpoint."
        )
    if not pinned and not _flag("SEGMENTEER_ALLOW_UNPINNED_DOWNLOADS"):
        raise RuntimeError(
            f"{asset}: remote version/checksum is not pinned; refusing automatic download. "
            "Provide a locally verified checkpoint or a fixed model revision, "
            "or explicitly set SEGMENTEER_ALLOW_UNPINNED_DOWNLOADS=1 "
            "to accept non-reproducible external assets."
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _file_lock(path: Path):
    
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def atomic_https_download(
    url: str,
    target: Path,
    *,
    asset: str,
    max_bytes: int = 8 * 1024**3,
    expected_sha256: str | None = None,
    pinned: bool = False,
) -> Path:
    """Download with TLS, size limits, locking, and atomic publication.

    Require a checksum or explicit approval for unpinned assets.
    """
    import requests

    source = urlsplit(url)
    if (
        source.scheme != "https"
        or not source.hostname
        or source.username
        or source.password
    ):
        raise ValueError("External asset downloads require a direct HTTPS URL.")
    target = Path(target)
    if expected_sha256 is not None and not __import__("re").fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(target.with_name(target.name + ".lock")):
        if target.is_file() and 0 < target.stat().st_size <= max_bytes:
            if expected_sha256 is None or sha256(target) == expected_sha256.lower():
                return target
            
        require_network(asset, pinned=pinned or expected_sha256 is not None)
        fd, stage_name = tempfile.mkstemp(
            prefix="." + target.name + ".", suffix=".part", dir=target.parent
        )
        os.close(fd)
        stage = Path(stage_name)
        try:
            with requests.get(
                url, stream=True, timeout=(15, 120), allow_redirects=True
            ) as response:
                response.raise_for_status()
                final_url = urlsplit(response.url)
                if final_url.scheme != "https" or not final_url.hostname:
                    raise RuntimeError("Refusing non-HTTPS model download redirect.")
                length = response.headers.get("content-length")
                if length is not None and int(length) > max_bytes:
                    raise ValueError(f"{asset} exceeds the configured maximum size.")
                size = 0
                with stage.open("wb") as file:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError(
                                f"{asset} exceeds the configured maximum size."
                            )
                        file.write(block)
                    file.flush()
                    os.fsync(file.fileno())
                if not size:
                    raise RuntimeError(f"{asset} download returned an empty response.")
            if expected_sha256 and sha256(stage) != expected_sha256.lower():
                raise ValueError(f"{asset} SHA-256 mismatch; refusing downloaded file.")
            os.replace(stage, target)
            return target
        finally:
            stage.unlink(missing_ok=True)

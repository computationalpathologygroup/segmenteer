"""Small, process-safe locks for shared benchmark output directories.

The benchmark stores expensive artifacts in method-specific directories, so
workers only need long-lived locks for the run IDs they own.  Root-level index
files are derived data and use a separate, short-lived lock.

``fcntl.flock`` is intentionally used here because the supported benchmark
hosts are macOS and Linux.  Locks are advisory, automatically released if a
process exits, and require no third-party dependency.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

LOCKS_DIR = ".segmenteer-locks"


class OutputLockError(RuntimeError):
    """Raised when another process owns an incompatible output lock."""


def _lock_path(output_dir: Path | str, category: str, name: str) -> Path:
    return Path(output_dir) / LOCKS_DIR / category / f"{name}.lock"


@contextmanager
def _exclusive_lock(
    path: Path,
    *,
    blocking: bool,
    description: str,
    shared: bool = False,
) -> Iterator[None]:
    """Hold one advisory filesystem lock until the context exits.

    ``shared=True`` permits cooperative slide shards for the exact same method
    configuration while remaining incompatible with a normal exclusive owner.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    flags = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB)
    try:
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise OutputLockError(
                    f"Another process is already running {description} in {path.parent.parent.parent}. "
                    "Use different methods, or wait for that worker to finish."
                ) from None
            raise

        # Shared holders must not mutate the same lock file concurrently.
        # Exclusive locks retain diagnostic metadata for ordinary workers.
        if not shared:
            handle.seek(0)
            handle.truncate()
            json.dump(
                {
                    "pid": os.getpid(),
                    "description": description,
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def claim_method_locks(
    output_dir: Path | str,
    run_ids: list[str],
    *,
    shared: bool = False,
) -> Iterator[None]:
    """Claim all selected method directories without serialising other methods.

    Locks are acquired in sorted order to eliminate cross-process deadlocks.
    Normal workers claim exclusive ownership.  Cooperative slide shards use a
    shared lock and are safe only when their slide sets are disjoint; this still
    prevents a non-sharded invocation from colliding with the active cohort.
    """
    output_dir = Path(output_dir)
    unique_run_ids = sorted(set(run_ids), key=str.casefold)
    if len(unique_run_ids) != len(run_ids):
        raise ValueError("SEGMENTERS contains duplicate method configurations.")

    with ExitStack() as stack:
        for run_id in unique_run_ids:
            stack.enter_context(
                _exclusive_lock(
                    _lock_path(output_dir, "methods", run_id),
                    blocking=False,
                    description=(f"shared method shard '{run_id}'" if shared else f"method '{run_id}'"),
                    shared=shared,
                )
            )
        yield


@contextmanager
def output_index_lock(output_dir: Path | str) -> Iterator[None]:
    """Serialise brief rebuilds of shared root-level generated files."""
    with _exclusive_lock(
        _lock_path(output_dir, "root", "indexes"),
        blocking=True,
        description="shared benchmark indexes",
    ):
        yield


@contextmanager
def dataset_metadata_lock(output_dir: Path | str) -> Iterator[None]:
    """Serialise shared dataset-contract creation and validation."""
    with _exclusive_lock(
        _lock_path(output_dir, "root", "dataset-metadata"),
        blocking=True,
        description="shared dataset metadata",
    ):
        yield

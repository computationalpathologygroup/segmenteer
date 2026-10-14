"""Atomic filesystem writes used by benchmark output artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace *path* after fully flushing temporary content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path | str, data: Any, *, indent: int = 2) -> None:
    """Serialise JSON and atomically replace *path*."""
    write_text_atomic(path, json.dumps(data, indent=indent), encoding="utf-8")

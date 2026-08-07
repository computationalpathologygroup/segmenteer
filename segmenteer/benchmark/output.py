"""Stable prediction/config output writer for the RUNNER component."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from segmenteer.benchmark.runner import BenchmarkResult
from segmenteer.io.atomic import write_json_atomic

PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"

__all__ = ["PredictionOutputWriter"]


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class PredictionOutputWriter:
    """Persist only the public RUNNER method artifacts.

    Each method/configuration owns one directory containing::

        <run_id>/config.yaml
        <run_id>/predictions/<slide>.geojson

    Dataset-level metadata is written separately by the workflow layer.
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)

    def __call__(self, result: BenchmarkResult) -> None:
        if result.failed:
            return
        if result.image_path is None:
            raise ValueError("Runner result is missing image_path; cannot name prediction output.")

        method_dir = self.output_dir / result.run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        predictions_dir.mkdir(parents=True, exist_ok=True)

        write_json_atomic(
            predictions_dir / f"{result.image_path.stem}.geojson",
            result.geojson,
        )

        if result.segmenter_config is not None:
            _write_text_atomic(
                method_dir / CONFIG_FILE,
                yaml.safe_dump(
                    result.segmenter_config,
                    default_flow_style=False,
                    sort_keys=False,
                ),
            )

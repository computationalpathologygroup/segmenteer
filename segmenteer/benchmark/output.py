from __future__ import annotations
import json
import os
import tempfile
import time
from pathlib import Path
import yaml
from segmenteer.benchmark.runner import BenchmarkResult
from segmenteer.benchmark.timing import persist_timing, timing_path
from segmenteer.io.predictions import (
    normalize_prediction_format,
    prediction_path,
    save_prediction,
)

PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"


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
        temporary.unlink(missing_ok=True)
        raise


class PredictionOutputWriter:
    def __init__(self, output_dir: Path | str, output_format: str = "geojson") -> None:
        self.output_dir = Path(output_dir)
        self.output_format = normalize_prediction_format(output_format)
        self._configured: set[str] = set()

    def __call__(self, result: BenchmarkResult) -> None:
        if result.failed:
            return
        if result.image_path is None:
            raise ValueError("Runner result is missing image_path.")
        method_dir = self.output_dir / result.run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        # Guard against stem collisions when cohort validation is bypassed.
        saved_prediction = prediction_path(
            predictions_dir, result.image_path.stem, self.output_format
        )
        if predictions_dir.is_dir():
            for existing in predictions_dir.iterdir():
                if not existing.is_file() or existing.suffix.casefold() not in (
                    ".xml",
                    ".geojson",
                ):
                    continue
                if existing.stem.casefold() != result.image_path.stem.casefold():
                    continue
                if existing != saved_prediction:
                    raise ValueError(
                        f"Ambiguous slide stem {result.image_path.stem!r}: existing prediction "
                        f"{existing.name!r} conflicts with {result.image_path.name!r}. "
                        "Use unique slide stems or separate experiment directories."
                    )
                timing_file = timing_path(
                    self.output_dir, result.run_id, result.image_path.stem
                )
                try:
                    existing_timing = json.loads(
                        timing_file.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Prediction {existing} already exists without verifiable slide identity. "
                        "Use the resume runner to repair partial outputs or a separate experiment directory."
                    ) from exc
                if (
                    not isinstance(existing_timing, dict)
                    or existing_timing.get("filename") != result.image_path.name
                ):
                    raise ValueError(
                        f"Prediction {existing} belongs to source "
                        f"{existing_timing.get('filename') if isinstance(existing_timing, dict) else 'unknown'!r}, "
                        f"not {result.image_path.name!r}; refusing to overwrite it."
                    )
        if result.run_id not in self._configured:
            if result.segmenter_config is not None:
                _write_text_atomic(
                    method_dir / CONFIG_FILE,
                    yaml.safe_dump(
                        result.segmenter_config,
                        default_flow_style=False,
                        sort_keys=False,
                    ),
                )
            self._configured.add(result.run_id)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        save_prediction(result.geojson, saved_prediction)
        result.serialization_time_s = time.perf_counter() - start
        result.total_processing_time_s = (
            result.segmentation_call_time_s or 0.0
        ) + result.serialization_time_s
        result.seconds_per_pixel = (
            result.total_processing_time_s / result.native_pixels
            if result.native_pixels
            else None
        )
        persist_timing(self.output_dir, result, prediction_file=saved_prediction)

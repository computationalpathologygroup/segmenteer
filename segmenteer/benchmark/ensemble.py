"""Prediction output writer and ensemble helpers for the runner component.

Runner artifacts are intentionally limited to predictions and method
configuration.  Evaluation and visualization are separate consumers.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from segmenteer.benchmark.resume import fingerprint
from segmenteer.benchmark.runner import BenchmarkResult

__all__ = ["EnsembleOutputWriter", "load_ensemble_members", "make_ensemble_run_id"]

PREDICTIONS_DIR = "predictions"
CONFIG_FILE = "config.yaml"
MANIFEST_FILE = "ensemble_manifest.json"


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, data: object) -> None:
    _write_text_atomic(path, json.dumps(_sanitize(data), indent=2))


def make_ensemble_run_id(member_run_ids: list[str], strategy: str = "majority", threshold: float = 0.5) -> str:
    return f"ensemble__members={len(member_run_ids)}_strategy={strategy}_threshold={threshold:g}"


class EnsembleOutputWriter:
    """Persist runner predictions in the stable output contract.

    For each method the writer creates only::

        <run_id>/config.yaml
        <run_id>/predictions/<slide>.geojson

    """

    def __init__(self, output_dir: Path | str, image_path: Path | str | None = None, **_ignored) -> None:
        self.output_dir = Path(output_dir)
        self.image_path = Path(image_path) if image_path else None
        self._members: list[dict] = []
        self._run_members: dict[str, list[dict]] = defaultdict(list)
        self._run_names: dict[str, str] = {}

    def __call__(self, result: BenchmarkResult) -> None:
        if not result.failed:
            self.save_member(result)

    def save_member(self, result: BenchmarkResult) -> Path:
        image_path = result.image_path or self.image_path
        image_stem = image_path.stem if image_path else "image"
        is_dataset = self.image_path is None and result.image_path is not None
        method_dir = self.output_dir / result.run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = predictions_dir / f"{image_stem}.geojson"
        _write_json(prediction_path, result.geojson)

        if result.segmenter_config is not None:
            import yaml

            _write_text_atomic(
                method_dir / CONFIG_FILE,
                yaml.safe_dump(result.segmenter_config, default_flow_style=False, sort_keys=False),
            )

        entry = {
            "name": result.method_name,
            "run_id": result.run_id,
            "path": result.run_id,
            "image_stem": image_stem,
            "image": str(image_path) if image_path else None,
            "prediction": f"{result.run_id}/{PREDICTIONS_DIR}/{image_stem}.geojson",
            "config": f"{result.run_id}/{CONFIG_FILE}",
        }
        if is_dataset:
            self._run_members[result.run_id].append(entry)
            self._run_names[result.run_id] = result.method_name
        else:
            self._members.append(entry)
        return method_dir

    def save_ensemble_result(
        self,
        image_path: Path | str,
        geojson: dict,
        member_run_ids: list[str],
        *,
        strategy: str = "majority",
        threshold: float = 0.5,
    ) -> Path:
        import yaml

        image_path = Path(image_path)
        run_id = make_ensemble_run_id(member_run_ids, strategy, threshold)
        method_dir = self.output_dir / run_id
        predictions_dir = method_dir / PREDICTIONS_DIR
        predictions_dir.mkdir(parents=True, exist_ok=True)
        _write_json(predictions_dir / f"{image_path.stem}.geojson", geojson)
        config = {
            "type": "ensemble",
            "strategy": strategy,
            "threshold": threshold,
            "n_members": len(member_run_ids),
            "member_run_ids": member_run_ids,
            "fingerprint": fingerprint(member_run_ids),
        }
        _write_text_atomic(method_dir / CONFIG_FILE, yaml.safe_dump(config, sort_keys=False))
        entry = {
            "name": "ensemble",
            "run_id": run_id,
            "path": run_id,
            "image_stem": image_path.stem,
            "image": str(image_path),
            "prediction": f"{run_id}/{PREDICTIONS_DIR}/{image_path.stem}.geojson",
            "config": f"{run_id}/{CONFIG_FILE}",
        }
        if self.image_path is None:
            self._run_members[run_id].append(entry)
            self._run_names[run_id] = "ensemble"
        else:
            self._members.append(entry)
        return method_dir

    def finalize(self, results: list[BenchmarkResult] | None = None, image_path: Path | str | None = None) -> Path:
        del results
        manifest = {
            "segmenteer_version": _segmenteer_version(),
            "mode": "single",
            "image": str(image_path) if image_path else None,
            "run_dir": str(self.output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_members": len(self._members),
            "members": self._members,
        }
        path = self.output_dir / MANIFEST_FILE
        _write_json(path, manifest)
        return path

    def refresh_dataset_indexes(self, dataset_metadata: dict | None = None) -> Path:
        del dataset_metadata
        members: list[dict] = []
        for method_dir in sorted(self.output_dir.iterdir(), key=lambda path: path.name.casefold()):
            prediction_dir = method_dir / PREDICTIONS_DIR
            if not method_dir.is_dir() or not prediction_dir.is_dir():
                continue
            images = [
                {
                    "name": method_dir.name,
                    "run_id": method_dir.name,
                    "path": method_dir.name,
                    "image_stem": prediction.stem,
                    "prediction": str(prediction.relative_to(self.output_dir)),
                    "config": f"{method_dir.name}/{CONFIG_FILE}",
                }
                for prediction in sorted(prediction_dir.glob("*.geojson"), key=lambda p: p.name.casefold())
            ]
            members.append({"run_id": method_dir.name, "name": self._run_names.get(method_dir.name, method_dir.name), "n_images": len(images), "images": images})
        manifest = {
            "segmenteer_version": _segmenteer_version(),
            "mode": "dataset",
            "run_dir": str(self.output_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_methods": len(members),
            "members": members,
        }
        path = self.output_dir / MANIFEST_FILE
        _write_json(path, manifest)
        return path

    def finalize_dataset(self, all_results=None, dataset_metadata: dict | None = None) -> Path:
        del all_results
        return self.refresh_dataset_indexes(dataset_metadata)

    @property
    def member_dirs(self) -> list[Path]:
        entries = list(self._members)
        for image_entries in self._run_members.values():
            entries.extend(image_entries)
        return [self.output_dir / entry["path"] for entry in entries]


def load_ensemble_members(manifest_path: Path | str, image_stem: str | None = None) -> list[dict]:
    manifest_path = Path(manifest_path)
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries: list[dict] = []
    if manifest.get("mode") == "dataset":
        for block in manifest.get("members", []):
            for entry in block.get("images", []):
                if image_stem is None or entry.get("image_stem") == image_stem:
                    raw_entries.append(entry)
    else:
        raw_entries = list(manifest.get("members", []))

    members: list[dict] = []
    for entry in raw_entries:
        relative = entry.get("prediction")
        if not relative:
            continue
        prediction = json.loads((run_dir / relative).read_text(encoding="utf-8"))
        members.append({
            "name": entry.get("name", entry.get("run_id", "method")),
            "run_id": entry.get("run_id", entry.get("name", "method")),
            "geojson": prediction,
            "metrics": {},
            "metadata": {"image": entry.get("image"), "image_stem": entry.get("image_stem")},
            "member_dir": run_dir / entry.get("path", entry.get("run_id", "")),
        })
    return members


def _segmenteer_version() -> str:
    try:
        from segmenteer import __version__
        return __version__
    except Exception:
        return "unknown"

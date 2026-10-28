"""Convert prediction formats while retaining inference provenance."""

from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from segmenteer.io.atomic import write_json_atomic
from segmenteer.benchmark.timing import rebuild_timing_csv, timing_path, _read_record
from segmenteer.io.predictions import (
    normalize_prediction_format,
    read_prediction,
    save_prediction,
)


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "suffix": path.suffix.casefold(),
    }


def convert_output_directory(
    input_dir: str | Path, output_dir: str | Path, output_format: str
) -> Path:
    source = Path(input_dir).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    selected = normalize_prediction_format(output_format)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == target or source in target.parents or target in source.parents:
        raise ValueError(
            "Source and destination experiment directories must be independent"
        )
    if target.exists():
        raise FileExistsError(f"Destination already exists: {target}")
    manifest_path = source / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("slides"), list):
        raise ValueError("Source does not contain a valid Segmenteer dataset manifest")
    source_format = normalize_prediction_format(
        manifest.get("prediction_format", "geojson")
    )
    method_dirs = [
        p
        for p in sorted(source.iterdir())
        if p.is_dir() and (p / "predictions").is_dir()
    ]
    if not method_dirs:
        raise ValueError("Source experiment contains no prediction directories")
    converted = []
    for method_dir in method_dirs:
        for path in sorted((method_dir / "predictions").glob(f"*.{source_format}")):
            # Preserve timing only when the source prediction signature matches.
            record_path = timing_path(source, method_dir.name, path.stem)
            record = _read_record(record_path) if record_path.is_file() else None
            if record_path.is_file() and record is None:
                raise ValueError(f"Invalid source timing record: {record_path}")
            if record is not None:
                if (
                    record["run_id"] != method_dir.name
                    or Path(record["filename"]).stem != path.stem
                ):
                    raise ValueError(f"Inconsistent timing identity: {record_path}")
                recorded = record.get("prediction_signature")
                if recorded is not None and recorded != _signature(path):
                    raise ValueError(
                        f"Source prediction no longer matches recorded timing: {path}"
                    )
            converted.append(
                (path, method_dir.name, path.stem, read_prediction(path), record)
            )
    if not converted:
        raise ValueError(f"No source *.{source_format} predictions found")
    target.mkdir(parents=True)
    try:
        updated = dict(manifest)
        updated["prediction_format"] = selected
        write_json_atomic(target / "dataset_manifest.json", updated)
        for method_dir in method_dirs:
            if (method_dir / "config.yaml").is_file():
                (target / method_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    method_dir / "config.yaml", target / method_dir.name / "config.yaml"
                )
        for source_path, run_id, stem, payload, record in converted:
            destination = target / run_id / "predictions" / f"{stem}.{selected}"
            save_prediction(payload, destination)
            if record is not None:
                converted_record = dict(record)
                converted_record["prediction_signature"] = _signature(destination)
                # Retain measured inference durations and label their provenance.
                converted_record["timing_provenance"] = (
                    "inherited_from_source_experiment"
                )
                converted_record["timing_origin_prediction_format"] = (
                    record.get("timing_origin_prediction_format") or source_format
                )
                converted_record["converted_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                write_json_atomic(timing_path(target, run_id, stem), converted_record)
        rebuild_timing_csv(target)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Segmenteer prediction outputs without rerunning inference."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-format", choices=("xml", "geojson"), required=True)
    args = parser.parse_args()
    print(
        f"Converted experiment: {convert_output_directory(args.input_dir, args.output_dir, args.output_format)}"
    )


if __name__ == "__main__":
    main()

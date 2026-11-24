"""Strict and deterministic WSI cohort loading for CLI experiments."""
from __future__ import annotations

import csv
from pathlib import Path

from segmenteer.benchmark.slide_identity import validate_unique_slide_stems
from segmenteer.runner import SUPPORTED_WSI_SUFFIXES, discover_slides


def load_dataset(
    source: str | Path,
    *,
    path_column: str = "path",
    slide_order: str = "smallest-first",
) -> list[Path]:
    """Read a CSV of paths or a flat WSI directory.

    A CSV preserves row order. Relative paths inside CSV files are relative to
    the CSV's directory, *not* the process working directory. Output IDs are
    slide stems, so duplicates or case-insensitive collisions are forbidden.
    """
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        slides = discover_slides(source, slide_order=slide_order)
    elif source.is_file():
        if source.suffix.casefold() != ".csv":
            raise ValueError(f"--dataset must be a CSV manifest or WSI directory: {source}")
        slides = []
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or path_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV {source} must contain a {path_column!r} column "
                    f"(found {reader.fieldnames or []!r})."
                )
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Duplicate CSV column headers in {source}.")
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"Extra CSV fields on row {row_number} in {source}.")
                value = row.get(path_column)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing {path_column!r} on row {row_number} in {source}.")
                slide = Path(value.strip()).expanduser()
                if not slide.is_absolute():
                    slide = source.parent / slide
                slide = slide.resolve()
                if slide.suffix.casefold() not in SUPPORTED_WSI_SUFFIXES:
                    raise ValueError(f"Unsupported WSI file on row {row_number}: {slide}")
                if not slide.is_file():
                    raise FileNotFoundError(
                        f"WSI missing on CSV row {row_number}: {slide}. "
                        "Paths must exist inside the container."
                    )
                slides.append(slide)
    else:
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    if not slides:
        raise ValueError(f"Dataset contains no supported slides: {source}")
    validate_unique_slide_stems(slides, context=f"dataset {source}")
    return slides

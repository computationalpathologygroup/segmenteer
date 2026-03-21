"""Scan a segmenteer output directory and build a typed data index.

The index is represented as an ``IndexData`` dataclass whose members are
converted to plain dicts at the API boundary via ``dataclasses.asdict``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PALETTE: tuple[str, ...] = (
    "#2E86AB",  # cerulean
    "#E05C4B",  # coral
    "#3BB273",  # emerald
    "#9B5DE5",  # amethyst
    "#F4A417",  # amber
    "#C15B78",  # raspberry
    "#3D9A8B",  # teal
    "#E87D3E",  # tangerine
    "#5B7FA6",  # steel blue
)

_WSI_EXTENSIONS: tuple[str, ...] = (
    ".tiff",
    ".tif",
    ".svs",
    ".ndpi",
    ".qptiff",
    ".scn",
    ".mrxs",
)


@dataclass(frozen=True)
class MethodInfo:
    """Metadata for a single segmentation run."""

    run_id: str
    name: str
    mpp: float | None
    params: dict[str, str]
    color: str


@dataclass(frozen=True)
class WSIRecord:
    """A single whole-slide image entry in the index."""

    stem: str
    image_path: str | None
    has_wsi: bool
    thumbnail_url: str | None


@dataclass
class IndexData:
    """Full benchmark index; JSON-serialisable via ``dataclasses.asdict``."""

    output_dir: str
    wsis: list[WSIRecord] = field(default_factory=list)
    methods: dict[str, MethodInfo] = field(default_factory=dict)
    scores: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


def load_index(output_dir: Path, data_dir: Path | None = None) -> IndexData:
    """Return a structured ``IndexData`` for all results in *output_dir*."""
    output_dir = Path(output_dir)

    run_ids: list[str] = sorted(
        d.name
        for d in output_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and d.name not in {"thumbnails"}
        and (d / "predictions").exists()
    )

    methods: dict[str, MethodInfo] = {}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    image_paths: dict[str, Path] = {}

    for i, run_id in enumerate(run_ids):
        scores_dir = output_dir / run_id / "eval" / "scores"
        if not scores_dir.exists():
            continue

        methods[run_id] = MethodInfo(
            run_id=run_id,
            name=run_id.split("__")[0],
            mpp=_parse_mpp(run_id),
            params=_parse_params(run_id),
            color=_PALETTE[i % len(_PALETTE)],
        )

        for score_file in sorted(scores_dir.glob("*.json")):
            stem = score_file.stem
            try:
                data: dict[str, Any] = json.loads(
                    score_file.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue

            scores.setdefault(stem, {})[run_id] = data

            if stem not in image_paths:
                img = _resolve_image_path(data.get("image_path"), stem, data_dir)
                if img:
                    image_paths[stem] = img

    thumbnails_dir = output_dir / "thumbnails"
    wsis: list[WSIRecord] = [
        WSIRecord(
            stem=stem,
            image_path=str(image_paths[stem]) if stem in image_paths else None,
            has_wsi=stem in image_paths,
            thumbnail_url=(
                f"/api/thumbnail/{stem}"
                if (thumbnails_dir / f"{stem}.png").exists()
                else None
            ),
        )
        for stem in sorted(scores)
    ]

    return IndexData(
        output_dir=str(output_dir),
        wsis=wsis,
        methods=methods,
        scores=scores,
    )


# ── helpers ──────────────────────────────────────────────────────────────


def _parse_mpp(run_id: str) -> float | None:
    m = re.search(r"mpp=(\d+(?:\.\d+)?)", run_id)
    return float(m.group(1)) if m else None


def _parse_params(run_id: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in run_id.split("__")[1:]:
        kv = part.split("=", 1)
        if len(kv) == 2:
            params[kv[0]] = kv[1]
    return params


def _resolve_image_path(
    stored: str | None, stem: str, data_dir: Path | None
) -> Path | None:
    if stored:
        p = Path(stored)
        if p.exists():
            return p
    if data_dir:
        for ext in _WSI_EXTENSIONS:
            c = data_dir / f"{stem}{ext}"
            if c.exists():
                return c
    return None

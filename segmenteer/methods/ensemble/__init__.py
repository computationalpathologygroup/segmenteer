"""Soft weighted majority-vote ensemble fusion.

Fuses multiple tissue-segmentation GeoJSONs produced by different methods into
a single consensus mask.  Each method votes proportionally to its unsupervised
quality weight (harmonic mean of coverage and compactness).  A pixel is kept
as tissue when its weighted vote fraction exceeds *threshold*:

* threshold = 0.0 → union   (any method wins)
* threshold = 0.5 → majority (>50 % weighted agreement)
* threshold = 1.0 → intersection (all methods must agree)

Public API
----------
``fuse_members``     — core fusion: list[member_dict] → GeoJSON FeatureCollection
``run_ensemble``     — orchestration: load + fuse + write results to disk
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import geojson
import numpy as np
from shapely.geometry import shape as shapely_shape
from skimage.draw import polygon as sk_polygon

from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import save_geojson
from segmenteer.visualization.heatmaps import \
    save_vote_heatmap as _save_vote_heatmap_img

__all__ = ["fuse_members", "run_ensemble", "quality_weight"]


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------


def _geojson_bbox(members: list[dict]) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) covering all member geometries."""
    minx = miny = math.inf
    maxx = maxy = -math.inf
    for m in members:
        for feat in m["geojson"].get("features", []):
            geom = shapely_shape(feat["geometry"])
            bx, by, Bx, By = geom.bounds
            minx, miny = min(minx, bx), min(miny, by)
            maxx, maxy = max(maxx, Bx), max(maxy, By)
    if not math.isfinite(minx):
        return 0.0, 0.0, 1.0, 1.0
    return minx, miny, maxx, maxy


def _rasterise_member(
    member: dict,
    minx: float,
    miny: float,
    h: int,
    w: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Burn one member's polygons into a binary (uint8) mask of shape (h, w)."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    for feat in member["geojson"].get("features", []):
        geom = shapely_shape(feat["geometry"])
        coords = list(geom.exterior.coords)
        rows = np.array([(y - miny) * scale_y for _, y in coords])
        cols = np.array([(x - minx) * scale_x for x, _ in coords])
        rr, cc = sk_polygon(rows, cols, shape=(h, w))
        canvas[rr, cc] = 1
    return canvas


# ---------------------------------------------------------------------------
# Quality weighting
# ---------------------------------------------------------------------------


def quality_weight(member: dict) -> float:
    """Score in (0, 1] based on unsupervised metrics; used as voting weight.

    Uses the harmonic mean of coverage and compactness so both must be
    reasonable to produce a high score.  Clamped to (0.01, 1.0) so every
    method retains at least a small vote.
    """
    u = (member.get("metrics") or {}).get("unsupervised") or {}
    coverage = float(u.get("coverage_ratio", 0.5) or 0.5)
    compactness = float(u.get("mean_compactness", 0.5) or 0.5)
    if coverage + compactness == 0:
        return 1.0
    score = 2 * coverage * compactness / (coverage + compactness)
    return max(0.01, min(1.0, score))


# ---------------------------------------------------------------------------
# Core fusion
# ---------------------------------------------------------------------------


def fuse_members(
    members: list[dict],
    threshold: float = 0.5,
    max_size: int = 2048,
    min_area: int = 50,
) -> dict:
    """Fuse *members* via soft weighted majority voting.

    Parameters
    ----------
    members:
        List of member dicts as returned by :func:`seg.load_ensemble_members`.
    threshold:
        Fraction of total weight a pixel needs to be tissue.
        0.5 = simple majority; 0.0 = union; 1.0 = intersection.
    max_size:
        Longest edge of the rasterisation canvas in pixels.
    min_area:
        Minimum polygon area (raster pixels²) to keep in the output.

    Returns
    -------
    GeoJSON FeatureCollection in the same coordinate space as the input members.
    """
    if not members:
        return geojson.FeatureCollection([])

    minx, miny, maxx, maxy = _geojson_bbox(members)
    span_x = maxx - minx or 1.0
    span_y = maxy - miny or 1.0

    aspect = span_y / span_x
    if aspect >= 1.0:
        h = max_size
        w = max(1, int(round(max_size / aspect)))
    else:
        w = max_size
        h = max(1, int(round(max_size * aspect)))

    scale_x = w / span_x
    scale_y = h / span_y

    weights = np.array([quality_weight(m) for m in members], dtype=np.float32)
    total_weight = weights.sum()

    print(f"  Members: {len(members)}  |  canvas: {w}×{h}")
    for m, w_val in zip(members, weights):
        print(f"    {m['run_id']:45s}  weight={w_val:.3f}")

    vote_map = np.zeros((h, w), dtype=np.float32)
    for member, weight in zip(members, weights):
        mask = _rasterise_member(member, minx, miny, h, w, scale_x, scale_y)
        vote_map += mask.astype(np.float32) * weight

    binary = (vote_map / total_weight) >= threshold

    raw_geojson = mask_to_geojson(binary, min_area=min_area, scaling_factor=1.0)

    def _rescale_coord(x: float, y: float) -> list:
        return [x / scale_x + minx, y / scale_y + miny]

    out_features = []
    for feat in raw_geojson.get("features", []):
        geom = feat["geometry"]
        if geom["type"] != "Polygon":
            continue
        scaled_rings = [
            [_rescale_coord(c[0], c[1]) for c in ring] for ring in geom["coordinates"]
        ]
        out_features.append(
            geojson.Feature(
                geometry=geojson.Polygon(scaled_rings),
                properties={"ensemble": True, "n_methods": len(members)},
            )
        )

    return geojson.FeatureCollection(out_features)


# ---------------------------------------------------------------------------
# Direct loader (no manifest required)
# ---------------------------------------------------------------------------


def load_members_from_dirs(
    method_dirs: list[Path],
    image_stem: str | None = None,
) -> tuple[list[str | None], list[dict]]:
    """Load members directly from method directories without a manifest.

    Scans ``predictions/`` to discover image stems, loads GeoJSON + metrics
    from ``eval/scores/``.

    Returns ``(all_stems, members)`` where ``all_stems`` is a list of image
    stems found (``[None]`` for a single-image layout).
    """
    stem_set: set[str] = set()
    for d in method_dirs:
        pred_dir = d / "predictions"
        if pred_dir.exists():
            stem_set.update(p.stem for p in pred_dir.glob("*.geojson"))

    if not stem_set:
        all_stems: list[str | None] = [None]
    else:
        all_stems = [image_stem] if image_stem is not None else sorted(stem_set)

    members: list[dict] = []
    for d in method_dirs:
        run_id = d.name
        for stem in all_stems:
            if stem is not None:
                geojson_path = d / "predictions" / f"{stem}.geojson"
                scores_path = d / "eval" / "scores" / f"{stem}.json"
            else:
                cands = (
                    list((d / "predictions").glob("*.geojson"))
                    if (d / "predictions").exists()
                    else list(d.glob("*.geojson"))
                )
                if not cands:
                    continue
                geojson_path = cands[0]
                sc = (
                    list((d / "eval" / "scores").glob("*.json"))
                    if (d / "eval" / "scores").exists()
                    else []
                )
                scores_path = sc[0] if sc else None

            if not geojson_path.exists():
                continue

            with geojson_path.open(encoding="utf-8") as fh:
                geojson_data = json.load(fh)

            metrics: dict = {}
            metadata: dict = {}
            if scores_path and Path(scores_path).exists():
                with Path(scores_path).open(encoding="utf-8") as fh:
                    eval_data = json.load(fh)
                metrics = eval_data.get("metrics", {})
                metadata = {k: v for k, v in eval_data.items() if k != "metrics"}

            members.append(
                {
                    "name": run_id.split("__")[0],
                    "run_id": run_id,
                    "geojson": geojson_data,
                    "metrics": metrics,
                    "metadata": metadata,
                    "member_dir": d,
                    "_stem": stem,
                }
            )

    return all_stems, members


# ---------------------------------------------------------------------------
# Vote-heatmap helper
# ---------------------------------------------------------------------------


def _build_vote_heatmap(
    members: list[dict],
    max_size: int,
    path: Path,
    bg_path: Path | None = None,
    alpha: float = 0.40,
) -> None:
    """Rasterise all members, compute per-pixel vote ratio, save as heatmap."""
    if not members:
        return
    minx, miny, maxx, maxy = _geojson_bbox(members)
    span_x = maxx - minx or 1.0
    span_y = maxy - miny or 1.0
    aspect = span_y / span_x
    if aspect >= 1.0:
        h, w = max_size, max(1, int(round(max_size / aspect)))
    else:
        w, h = max_size, max(1, int(round(max_size * aspect)))
    scale_x, scale_y = w / span_x, h / span_y

    weights = np.array([quality_weight(m) for m in members], dtype=np.float32)
    total = weights.sum()
    vote_map = np.zeros((h, w), dtype=np.float32)
    for member, weight in zip(members, weights):
        mask = _rasterise_member(member, minx, miny, h, w, scale_x, scale_y)
        vote_map += mask.astype(np.float32) * weight

    _save_vote_heatmap_img(vote_map / total, path, bg_path=bg_path, alpha=alpha)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_ensemble(
    manifest_path: Path | None,
    threshold: float = 0.5,
    max_size: int = 2048,
    min_area: int = 50,
    image_stem: str | None = None,
    member_dirs: list[str | Path] | None = None,
    max_members: int | None = None,
) -> Path:
    """Fuse all methods and write results into the output directory.

    Parameters
    ----------
    manifest_path:
        Path to ``ensemble_manifest.json``.  Set to ``None`` when using
        *member_dirs* directly.
    threshold:
        Soft-vote threshold. 0.0 = union, 0.5 = majority, 1.0 = intersection.
    max_size:
        Longest edge of the internal rasterisation canvas.
    min_area:
        Minimum polygon area (raster pixels²) to keep.
    image_stem:
        Restrict to one slide by stem name (dataset runs only).
    member_dirs:
        Explicit list of method output directories to include.  Paths may be
        absolute or relative to the current working directory.
    max_members:
        Keep only the top-N members by quality weight.  ``None`` = no limit.

    Returns the output root directory.
    """
    import yaml

    import segmenteer as seg  # local import to avoid circular at module level

    resolved_dirs: list[Path] | None = (
        [Path(d).resolve() for d in member_dirs] if member_dirs is not None else None
    )

    if manifest_path is None:
        if resolved_dirs is None:
            raise ValueError("Either manifest_path or member_dirs must be provided.")
        run_dir = resolved_dirs[0].parent
        all_stems, _preloaded = load_members_from_dirs(resolved_dirs, image_stem)
        use_manifest = False
    else:
        manifest_path = Path(manifest_path)
        run_dir = manifest_path.parent
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        mode = manifest.get("mode", "single")
        all_stems = (
            [
                s
                for s in manifest.get("images", [])
                if image_stem is None or s == image_stem
            ]
            if mode == "dataset"
            else [None]
        )
        _preloaded = None
        use_manifest = True

    run_id = f"ensemble__threshold={threshold}"
    method_dir = run_dir / run_id
    predictions_dir = method_dir / "predictions"
    scores_dir = method_dir / "eval" / "scores"
    heatmaps_dir = method_dir / "eval" / "heatmaps"
    for d in (predictions_dir, scores_dir, heatmaps_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = {
        "class": "ensemble.weighted_majority_vote",
        "params": {
            "threshold": threshold,
            "max_size": max_size,
            "min_area": min_area,
            "source_manifest": str(manifest_path),
            "member_dirs": [str(d) for d in (resolved_dirs or [])],
        },
    }
    (method_dir / "config.yaml").write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    for stem in all_stems:
        label = stem or "(single image)"
        print(f"\n── Fusing {label} ──")

        if use_manifest:
            members = seg.load_ensemble_members(manifest_path, image_stem=stem)
            if resolved_dirs is not None:
                dir_strs = {str(d) for d in resolved_dirs}
                members = [
                    m
                    for m in members
                    if str(Path(m["member_dir"]).resolve()) in dir_strs
                ]
        else:
            members = [m for m in (_preloaded or []) if m.get("_stem") == stem]

        if not members:
            print(f"  No members found for {label}, skipping.")
            continue

        if max_members is not None and len(members) > max_members:
            members = sorted(members, key=quality_weight, reverse=True)[:max_members]
            print(f"  Keeping top {max_members} members by quality weight.")

        fused = fuse_members(
            members, threshold=threshold, max_size=max_size, min_area=min_area
        )
        n_polys = len(fused.get("features", []))
        print(f"  → {n_polys} polygon(s)")

        file_stem = stem or "image"
        save_geojson(fused, predictions_dir / f"{file_stem}.geojson")

        weights = {m["run_id"]: float(quality_weight(m)) for m in members}
        scores = {
            "run_id": run_id,
            "image_stem": file_stem,
            "threshold": threshold,
            "n_methods": len(members),
            "n_polygons": n_polys,
            "methods": [m["run_id"] for m in members],
            "weights": weights,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(manifest_path),
        }
        (scores_dir / f"{file_stem}.json").write_text(
            json.dumps(scores, indent=2), encoding="utf-8"
        )

        thumb = run_dir / "thumbnails" / f"{file_stem}.png"
        try:
            _build_vote_heatmap(
                members, max_size, heatmaps_dir / f"{file_stem}.png", bg_path=thumb
            )
        except ImportError as exc:
            print(f"  [warn] matplotlib not available, skipping heatmap: {exc}")
        except (OSError, ValueError) as exc:
            print(f"  [warn] could not save heatmap for {file_stem}: {exc}")

    print(f"\nDone.  Ensemble written to: {run_dir / run_id}")
    return run_dir

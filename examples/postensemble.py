"""Post-run ensemble fusion for segmenteer benchmark outputs.

Loads all method outputs saved by :class:`EnsembleOutputWriter` and fuses
them with **soft weighted majority voting** – smarter than intersection (AND)
or union (OR) because:

* Each method casts a vote proportional to its unsupervised quality weight
  (coverage × compactness).  Bad methods contribute less.
* A pixel is kept as tissue only when the total weighted vote at that pixel
  exceeds a configurable fraction of the maximum possible weight.

  - threshold = 0.0  →  union  (any method wins)
  - threshold = 0.5  →  majority  (>50 % weighted agreement)
  - threshold = 1.0  →  intersection  (all methods must agree)

  Default: 0.5 (majority).

Usage
-----
Edit the ``__main__`` block at the bottom and run::

    python examples/postensemble.py
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

import segmenteer as seg
from segmenteer.benchmark.ensemble import MANIFEST_FILE
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import save_geojson
from segmenteer.visualization.heatmaps import save_vote_heatmap

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


def _quality_weight(member: dict) -> float:
    """Score in (0, 1] based on unsupervised metrics; used as voting weight."""
    u = (member.get("metrics") or {}).get("unsupervised") or {}
    coverage = float(u.get("coverage_ratio", 0.5) or 0.5)
    compactness = float(u.get("mean_compactness", 0.5) or 0.5)
    # harmonic mean of the two – both must be reasonable to score high
    if coverage + compactness == 0:
        return 1.0
    score = 2 * coverage * compactness / (coverage + compactness)
    # clamp to (0.01, 1.0) so every method retains at least a small vote
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
        Fraction of total weight a pixel needs to be counted as tissue.
        0.5 = simple majority; 0.0 = union; 1.0 = intersection.
    max_size:
        Longest edge of the rasterisation canvas in pixels.  Larger → finer
        detail, but slower.
    min_area:
        Minimum polygon area (in raster pixels²) to keep in the output.

    Returns
    -------
    GeoJSON FeatureCollection whose coordinates are in the same space as the
    input member GeoJSONs.
    """
    if not members:
        return geojson.FeatureCollection([])

    # 1. Bounding box of all geometries combined
    minx, miny, maxx, maxy = _geojson_bbox(members)
    span_x = maxx - minx or 1.0
    span_y = maxy - miny or 1.0

    # 2. Canvas dimensions (preserve aspect ratio, cap longest edge)
    aspect = span_y / span_x
    if aspect >= 1.0:
        h = max_size
        w = max(1, int(round(max_size / aspect)))
    else:
        w = max_size
        h = max(1, int(round(max_size * aspect)))

    scale_x = w / span_x
    scale_y = h / span_y

    # 3. Per-method quality weights
    weights = np.array([_quality_weight(m) for m in members], dtype=np.float32)
    total_weight = weights.sum()

    print(f"  Members: {len(members)}  |  canvas: {w}×{h}")
    for m, w_val in zip(members, weights):
        print(f"    {m['run_id']:45s}  weight={w_val:.3f}")

    # 4. Accumulate weighted votes into a float canvas
    vote_map = np.zeros((h, w), dtype=np.float32)
    for member, weight in zip(members, weights):
        mask = _rasterise_member(member, minx, miny, h, w, scale_x, scale_y)
        vote_map += mask.astype(np.float32) * weight

    # 5. Soft threshold: pixel is tissue if its weighted fraction ≥ threshold
    binary = (vote_map / total_weight) >= threshold

    # 6. Convert raster → GeoJSON in original coordinate space
    #    mask_to_geojson works in (row, col) → we scale coords back afterwards
    raw_geojson = mask_to_geojson(binary, min_area=min_area, scaling_factor=1.0)

    # Re-scale polygon coords from canvas space back to original pixel space
    def _rescale_coord(x: float, y: float) -> list:
        return [x / scale_x + minx, y / scale_y + miny]

    out_features = []
    for feat in raw_geojson.get("features", []):
        geom = feat["geometry"]
        if geom["type"] != "Polygon":
            continue
        scaled_rings = []
        for ring in geom["coordinates"]:
            # GeoJSON polygon: coords are (x=col, y=row)
            scaled_rings.append([_rescale_coord(c[0], c[1]) for c in ring])
        out_features.append(
            geojson.Feature(
                geometry=geojson.Polygon(scaled_rings),
                properties={"ensemble": True, "n_methods": len(members)},
            )
        )

    return geojson.FeatureCollection(out_features)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_ensemble(
    manifest_path: Path,
    threshold: float = 0.5,
    max_size: int = 2048,
    min_area: int = 50,
    image_stem: str | None = None,
) -> Path:
    """Fuse all methods from *manifest_path* and write results into the same
    output directory as all other methods.

    Output layout::

        outputs/<timestamp>/
            ensemble__threshold=<t>/      ← sits alongside other method dirs
                config.yaml
                predictions/
                    <image_stem>.geojson
                eval/
                    scores/
                        <image_stem>.json
                    heatmaps/
                        <image_stem>.png  ← vote-ratio heatmap

    Returns the run directory.
    """
    manifest_path = Path(manifest_path)
    run_dir = manifest_path.parent

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    mode = manifest.get("mode", "single")

    # Write into the same directory as all other methods
    out_root = run_dir

    # Single run_id that encodes the fusion parameters
    run_id = f"ensemble__threshold={threshold}"

    method_dir = out_root / run_id
    predictions_dir = method_dir / "predictions"
    scores_dir = method_dir / "eval" / "scores"
    heatmaps_dir = method_dir / "eval" / "heatmaps"
    for d in (predictions_dir, scores_dir, heatmaps_dir):
        d.mkdir(parents=True, exist_ok=True)

    # config.yaml — written once for the ensemble method
    import yaml

    config = {
        "class": "ensemble.weighted_majority_vote",
        "params": {
            "threshold": threshold,
            "max_size": max_size,
            "min_area": min_area,
            "source_manifest": str(manifest_path),
        },
    }
    (method_dir / "config.yaml").write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    # Collect image stems to process
    if mode == "dataset":
        all_stems: list[str | None] = manifest.get("images", [])
        if image_stem is not None:
            all_stems = [s for s in all_stems if s == image_stem]
    else:
        all_stems = [None]

    for stem in all_stems:
        label = stem or "(single image)"
        print(f"\n── Fusing {label} ──")

        members = seg.load_ensemble_members(manifest_path, image_stem=stem)
        if not members:
            print(f"  No members found for {label}, skipping.")
            continue

        fused = fuse_members(
            members, threshold=threshold, max_size=max_size, min_area=min_area
        )
        n_polys = len(fused.get("features", []))
        print(f"  → {n_polys} polygon(s)")

        file_stem = stem or "image"

        # predictions/<stem>.geojson
        save_geojson(fused, predictions_dir / f"{file_stem}.geojson")

        # eval/scores/<stem>.json
        weights = {m["run_id"]: float(_quality_weight(m)) for m in members}
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

        # eval/heatmaps/<stem>.png — vote-ratio heatmap (thumbnail as background)
        thumb = run_dir / "thumbnails" / f"{file_stem}.png"
        try:
            _save_vote_heatmap(
                members, max_size, heatmaps_dir / f"{file_stem}.png", bg_path=thumb
            )
        except ImportError as exc:
            print(f"  [warn] matplotlib not available, skipping heatmap: {exc}")
        except (OSError, ValueError) as exc:
            print(f"  [warn] could not save heatmap for {file_stem}: {exc}")

    print(f"\nDone.  Ensemble written to: {out_root / run_id}")
    return out_root


def _save_vote_heatmap(
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

    weights = np.array([_quality_weight(m) for m in members], dtype=np.float32)
    total = weights.sum()
    vote_map = np.zeros((h, w), dtype=np.float32)
    for member, weight in zip(members, weights):
        mask = _rasterise_member(member, minx, miny, h, w, scale_x, scale_y)
        vote_map += mask.astype(np.float32) * weight

    save_vote_heatmap(vote_map / total, path, bg_path=bg_path, alpha=alpha)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ensemble(
        manifest_path=Path("outputs/1903_1622/ensemble_manifest.json"),
        threshold=0.5,  # 0.0=union  0.5=majority  1.0=intersection
        max_size=2048,
        min_area=50,
        image_stem=None,  # None = all images; set to e.g. "006_M2" for one
    )

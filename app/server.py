from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.wsi import WSITileServer
from app_official.loader import DEFAULT_HOLE_MODE, DEFAULT_OUTPUT_PREFIX, IndexData, load_official_index
from app_official.loader import _WSI_EXTENSIONS
from app_official.loader import HOLES_ARE_NOT_TISSUE, HOLES_ARE_TISSUE

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
_STATIC_DIR = _HERE / "static"


def create_app(
    output_dir: str,
    metrics_dir: str,
    data_dir: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
    hole_mode: str = DEFAULT_HOLE_MODE,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    tissue_groups: set[str] | None = None,
    hole_groups: set[str] | None = None,
) -> FastAPI:
    """Create a viewer for the selected official evaluation cohort.

    ``output_dir`` is the original run root containing method prediction folders.
    ``metrics_dir`` is the official-results directory containing CSVs such as
    ``metrics__holes_are_not_tissue__<method>.csv``.  Only those CSVs define
    the methods and slides exposed by the viewer.
    """
    output_path = Path(output_dir).expanduser().resolve()
    metrics_path = Path(metrics_dir).expanduser().resolve()
    data_path = Path(data_dir).expanduser().resolve() if data_dir else None
    gt_path = Path(ground_truth_dir).expanduser().resolve() if ground_truth_dir else output_path / "ground_truth"
    tissue_group_set = {group.casefold() for group in (tissue_groups or {"tissue"})}
    hole_group_set = {group.casefold() for group in (hole_groups or {"hole", "holes"})}

    if not output_path.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_path}")
    if not metrics_path.is_dir():
        raise RuntimeError(f"Official metrics directory does not exist: {metrics_path}")
    if hole_mode not in {HOLES_ARE_TISSUE, HOLES_ARE_NOT_TISSUE}:
        raise RuntimeError(f"Unsupported --hole-mode: {hole_mode}")

    try:
        index_data = load_official_index(
            output_path,
            metrics_path,
            data_dir=data_path,
            ground_truth_dir=gt_path,
            hole_mode=hole_mode,
            output_prefix=output_prefix,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    index_payload = dataclasses.asdict(index_data)
    method_ids = set(index_data.methods)
    official_stems = {wsi.stem for wsi in index_data.wsis}
    image_paths = {
        wsi.stem: Path(wsi.image_path)
        for wsi in index_data.wsis
        if wsi.image_path
    }

    def _current_index() -> IndexData:
        return index_data

    def _current_image_paths() -> dict[str, Path]:
        return image_paths

    def _resolve_wsi_path(stem: str) -> Path | None:
        """Resolve a WSI path lazily without scanning --data at startup.

        The official viewer cohort is defined by the metrics CSVs; WSI files are
        only needed when a slide is actually opened or a visible overview card
        requests tiles.  Try exact root-level filenames first because this is
        constant-time even on large mounted volumes.  Fall back to one
        case-insensitive directory pass only for that specific stem.
        """
        cached = image_paths.get(stem)
        if cached and cached.is_file():
            return cached
        if data_path is None or not data_path.is_dir():
            return None
        for extension in _WSI_EXTENSIONS:
            candidate = data_path / f"{stem}{extension}"
            if candidate.is_file():
                image_paths[stem] = candidate
                return candidate
        target = stem.casefold()
        try:
            for candidate in data_path.iterdir():
                if (
                    candidate.is_file()
                    and candidate.suffix.casefold() in _WSI_EXTENSIONS
                    and candidate.stem.casefold() == target
                ):
                    image_paths[stem] = candidate
                    return candidate
        except OSError:
            return None
        return None

    tile_servers: dict[str, WSITileServer] = {}

    def _server(stem: str) -> WSITileServer:
        if stem not in tile_servers:
            path = _resolve_wsi_path(stem)
            if path is None:
                raise HTTPException(404, f"WSI file not found for '{stem}'")
            try:
                tile_servers[stem] = WSITileServer(path)
            except Exception as exc:
                raise HTTPException(422, f"Could not open WSI '{stem}': {exc}") from exc
        return tile_servers[stem]

    app = FastAPI(title="segmenteer official-results viewer", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/index")
    def get_index() -> JSONResponse:
        return JSONResponse(
            index_payload,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/wsi/{stem}/info")
    def get_wsi_info(stem: str) -> JSONResponse:
        server = _server(stem)
        return JSONResponse(
            {
                "stem": stem,
                "width": server.width,
                "height": server.height,
                "mpp": server.mpp,
                "n_levels": server.n_levels,
                "backend": server.backend,
                "tile_read_warning": server.last_error,
            }
        )

    @app.get("/api/wsi/{stem}/dzi")
    def get_dzi(stem: str) -> Response:
        return Response(content=_server(stem).dzi_descriptor(), media_type="application/xml")

    @app.get("/api/wsi/{stem}/dzi_files/{level}/{tile}.jpeg")
    def get_tile(stem: str, level: int, tile: str) -> Response:
        try:
            x_text, y_text = tile.split("_")
            x, y = int(x_text), int(y_text)
        except ValueError as exc:
            raise HTTPException(400, "Tile coordinate must be '{x}_{y}'") from exc
        return Response(content=_server(stem).get_tile(level, x, y), media_type="image/jpeg")

    def _add_metadata(geojson: dict, *, stem: str, layer: str, run_id: str | None = None) -> JSONResponse:
        scale_to_viewport: float | None = None
        try:
            scale_to_viewport = 1.0 / _server(stem).width
        except HTTPException:
            pass
        geojson["_meta"] = {
            "run_id": run_id,
            "stem": stem,
            "layer": layer,
            "coordinate_space": "level_0_pixels",
            "scale_to_viewport": scale_to_viewport,
            "official_hole_mode": hole_mode,
        }
        return JSONResponse(geojson)

    def _read_geojson(path: Path, label: str) -> dict:
        if not path.is_file():
            raise HTTPException(404, f"{label} file not found")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, f"Could not read {label} GeoJSON") from exc

    @lru_cache(maxsize=4096)
    def _ground_truth_geojson(stem: str) -> dict:
        source = gt_path / f"{stem}.geojson"
        geojson = _read_geojson(source, "ground truth")
        if hole_mode == HOLES_ARE_NOT_TISSUE:
            return _holes_are_not_tissue_geojson(
                geojson,
                source_path=source,
                tissue_groups=tissue_group_set,
                hole_groups=hole_group_set,
            )
        return geojson

    @app.get("/api/overlay/{stem}/{run_id}")
    def get_overlay(stem: str, run_id: str) -> JSONResponse:
        if run_id not in method_ids:
            raise HTTPException(404, "Method not found in selected official metrics")
        entry = index_data.scores.get(stem, {}).get(run_id)
        if not isinstance(entry, dict):
            raise HTTPException(404, "Slide/method pair is not in the selected official metrics cohort")
        if not entry.get("_prediction_available", False):
            raise HTTPException(404, "Prediction GeoJSON is not available for this official slide/method pair")
        geojson = _read_geojson(output_path / run_id / "predictions" / f"{stem}.geojson", "prediction")
        return _add_metadata(geojson, stem=stem, run_id=run_id, layer="prediction")

    @app.get("/api/ground-truth/{stem}")
    def get_ground_truth(stem: str) -> JSONResponse:
        if stem not in official_stems:
            raise HTTPException(404, "Slide is not in the selected official metrics cohort")
        if not (gt_path / f"{stem}.geojson").is_file():
            raise HTTPException(404, "Ground truth GeoJSON file not found")
        # Return a defensive copy because _add_metadata mutates the dictionary.
        geojson = json.loads(json.dumps(_ground_truth_geojson(stem)))
        return _add_metadata(geojson, stem=stem, layer="ground_truth")

    @lru_cache(maxsize=2048)
    def _computed_evaluation_layer(stem: str, run_id: str, layer: str) -> dict:
        """Compute TP/FP/FN polygons lazily from official prediction and GT GeoJSON.

        Official July-8 metrics CSVs contain numeric vector metrics, but they do
        not ship saved TP/FP/FN GeoJSON artifacts.  For the official viewer we
        therefore compute the spatial error layers on demand, only for the
        selected slide/method/layer.  Results are cached in memory for the
        duration of the server process.
        """
        prediction_geojson = _read_geojson(
            output_path / run_id / "predictions" / f"{stem}.geojson",
            "prediction",
        )
        ground_truth_geojson = _ground_truth_geojson(stem)
        return _boolean_error_layer_geojson(
            prediction_geojson,
            ground_truth_geojson,
            layer=layer,
            stem=stem,
            run_id=run_id,
        )

    @app.get("/api/evaluation/{stem}/{run_id}/{layer}")
    def get_evaluation_layer(stem: str, run_id: str, layer: str) -> JSONResponse:
        if run_id not in method_ids:
            raise HTTPException(404, "Method not found in selected official metrics")
        entry = index_data.scores.get(stem, {}).get(run_id)
        if not isinstance(entry, dict):
            raise HTTPException(404, "Slide/method pair is not in the selected official metrics cohort")
        allowed = {"true_positive", "false_positive", "false_negative"}
        if layer not in allowed:
            raise HTTPException(400, "layer must be true_positive, false_positive, or false_negative")
        if not entry.get("_prediction_available", False):
            raise HTTPException(404, "Prediction GeoJSON is not available for this official slide/method pair")
        if not (gt_path / f"{stem}.geojson").is_file():
            raise HTTPException(404, "Ground truth GeoJSON file not found")

        # Prefer any pre-existing saved QA layer if present, but compute lazily
        # when official results only contain numeric metrics.
        saved_path = output_path / run_id / "eval" / "errors" / f"{stem}_{layer}.geojson"
        if saved_path.is_file():
            geojson = _read_geojson(saved_path, layer.replace("_", " "))
        else:
            geojson = json.loads(json.dumps(_computed_evaluation_layer(stem, run_id, layer)))
        return _add_metadata(geojson, stem=stem, run_id=run_id, layer=layer)

    @app.get("/api/error-map/{stem}/{run_id}")
    def get_error_map(stem: str, run_id: str) -> Response:
        if run_id not in method_ids:
            raise HTTPException(404, "Method not found in selected official metrics")
        if stem not in index_data.scores or run_id not in index_data.scores.get(stem, {}):
            raise HTTPException(404, "Slide/method pair is not in the selected official metrics cohort")
        path = output_path / run_id / "eval" / "error_maps" / f"{stem}.png"
        if not path.is_file():
            raise HTTPException(404, "Supervised error map not found")
        return Response(content=path.read_bytes(), media_type="image/png")


    @lru_cache(maxsize=1)
    def _summary_payload_cached() -> dict:
        from app_official.summary import summary_payload

        return summary_payload(index_data)

    @lru_cache(maxsize=1)
    def _summary_csv_cached() -> str:
        from app_official.summary import summary_csv

        return summary_csv(index_data)

    @app.get("/api/summary")
    def get_summary() -> JSONResponse:
        return JSONResponse(_summary_payload_cached(), headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/summary.csv")
    def get_summary_csv() -> Response:
        filename = f"segmenteer_official_{hole_mode}_summary_{output_path.name}.csv"
        return Response(
            content=_summary_csv_cached().encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/report")
    def get_report() -> Response:
        from app.report import generate_report

        content = generate_report(_current_index(), output_path)
        filename = f"segmenteer_official_{hole_mode}_report_{output_path.name}.html"
        return Response(
            content=content.encode("utf-8"),
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    return app



# ── On-demand TP/FP/FN geometry ───────────────────────────────────────────────

def _boolean_error_layer_geojson(
    prediction_geojson: dict,
    ground_truth_geojson: dict,
    *,
    layer: str,
    stem: str,
    run_id: str,
) -> dict:
    """Return one spatial error layer as GeoJSON.

    true_positive  = prediction ∩ ground_truth
    false_positive = prediction - ground_truth
    false_negative = ground_truth - prediction
    """
    try:
        from shapely.geometry import GeometryCollection
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HTTPException(500, "TP/FP/FN geometry requires shapely") from exc

    prediction = _geojson_polygon_union(prediction_geojson)
    ground_truth = _geojson_polygon_union(ground_truth_geojson)

    if layer == "true_positive":
        geometry = prediction.intersection(ground_truth)
    elif layer == "false_positive":
        geometry = prediction.difference(ground_truth)
    elif layer == "false_negative":
        geometry = ground_truth.difference(prediction)
    else:  # guarded by endpoint; defensive for internal callers
        raise HTTPException(400, "layer must be true_positive, false_positive, or false_negative")

    if geometry.is_empty:
        geometry = GeometryCollection()
    elif not geometry.is_valid:
        geometry = geometry.buffer(0)

    return _geometry_to_feature_collection(
        geometry,
        properties={
            "part_of_group": layer,
            "computed_on_demand": True,
            "source_slide": stem,
            "segmentation_method": run_id,
        },
    )


def _geojson_polygon_union(geojson: dict):
    """Union all polygonal content in a GeoJSON FeatureCollection."""
    try:
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HTTPException(500, "GeoJSON boolean operations require shapely") from exc

    geometries = []
    for feature in geojson.get("features", []):
        geometry_dict = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry_dict:
            continue
        try:
            geometry = shape(geometry_dict)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty:
            continue
        if isinstance(geometry, (Polygon, MultiPolygon)):
            geometries.append(geometry)
        elif isinstance(geometry, GeometryCollection):
            pieces = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty]
            geometries.extend(pieces)

    if not geometries:
        return GeometryCollection()
    result = unary_union(geometries)
    if not result.is_valid:
        result = result.buffer(0)
    return result


def _geometry_to_feature_collection(geometry, *, properties: dict) -> dict:
    """Convert polygonal shapely geometry into a GeoJSON FeatureCollection."""
    try:
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HTTPException(500, "GeoJSON conversion requires shapely") from exc

    def iter_polygons(value):
        if value.is_empty:
            return
        if isinstance(value, Polygon):
            yield value
            return
        if isinstance(value, MultiPolygon):
            yield from value.geoms
            return
        if isinstance(value, GeometryCollection):
            for part in value.geoms:
                yield from iter_polygons(part)

    features = []
    for polygon in iter_polygons(geometry) or []:
        if polygon.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": dict(properties),
            }
        )
    return {"type": "FeatureCollection", "features": features}


# ── Ground-truth hole handling ────────────────────────────────────────────────

def _holes_are_not_tissue_geojson(
    geojson: dict,
    *,
    source_path: Path,
    tissue_groups: set[str],
    hole_groups: set[str],
) -> dict:
    """Return positive tissue = union(tissue polygons) - union(hole polygons).

    This mirrors the official evaluator's holes_are_not_tissue interpretation
    without writing temporary files.  Files with no recognised ``part_of_group``
    labels fall back to treating all polygonal features as positive tissue.
    """
    try:
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
        from shapely.ops import unary_union
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise HTTPException(500, "holes_are_not_tissue ground truth requires shapely") from exc

    tissue_geometries = []
    hole_geometries = []
    unclassified_geometries = []
    saw_recognised_group = False

    def clean_geometry(geometry_dict: dict):
        geometry = shape(geometry_dict)
        if geometry.is_empty:
            return GeometryCollection()
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        return geometry

    def has_polygonal_content(geometry) -> bool:
        if geometry.is_empty:
            return False
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return True
        if isinstance(geometry, GeometryCollection):
            return any(has_polygonal_content(part) for part in geometry.geoms)
        return False

    def iter_polygons(geometry):
        if geometry.is_empty:
            return
        if isinstance(geometry, Polygon):
            yield geometry
            return
        if isinstance(geometry, MultiPolygon):
            yield from geometry.geoms
            return
        if isinstance(geometry, GeometryCollection):
            for part in geometry.geoms:
                yield from iter_polygons(part)

    for feature in geojson.get("features", []):
        geometry_dict = feature.get("geometry")
        if not geometry_dict:
            continue
        geometry = clean_geometry(geometry_dict)
        if not has_polygonal_content(geometry):
            continue

        properties = feature.get("properties") or {}
        group = str(properties.get("part_of_group") or "").casefold()
        if group in tissue_groups:
            tissue_geometries.append(geometry)
            saw_recognised_group = True
        elif group in hole_groups:
            hole_geometries.append(geometry)
            saw_recognised_group = True
        else:
            unclassified_geometries.append(geometry)

    if not saw_recognised_group:
        tissue_geometries = unclassified_geometries

    tissue_union = unary_union(tissue_geometries) if tissue_geometries else GeometryCollection()
    hole_union = unary_union(hole_geometries) if hole_geometries else GeometryCollection()
    positive_geometry = tissue_union.difference(hole_union)
    if not positive_geometry.is_valid:
        positive_geometry = positive_geometry.buffer(0)

    features = []
    for polygon in iter_polygons(positive_geometry):
        if polygon.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "part_of_group": "tissue_minus_holes",
                    "source_geojson": str(source_path),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}

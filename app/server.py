from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.loader import IndexData, _WSI_EXTENSIONS, load_index

_HERE = Path(__file__).parent
_STATIC_DIR = _HERE / "static"


def create_app(
    output_dir: str,
    metrics_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
) -> FastAPI:
    """Create a viewer over runner output with optional GT and metrics."""
    output_path = Path(output_dir).expanduser().resolve()
    metrics_path = Path(metrics_file).expanduser().resolve() if metrics_file else None
    data_path = Path(data_dir).expanduser().resolve() if data_dir else None
    gt_path = Path(ground_truth_dir).expanduser().resolve() if ground_truth_dir else None

    try:
        index_data = load_index(
            output_path,
            metrics_file=metrics_path,
            data_dir=data_path,
            ground_truth_dir=gt_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc

    index_payload = dataclasses.asdict(index_data)
    method_ids = set(index_data.methods)
    slide_stems = {wsi.stem for wsi in index_data.wsis}
    image_paths = {wsi.stem: Path(wsi.image_path) for wsi in index_data.wsis if wsi.image_path}

    def _resolve_wsi_path(stem: str) -> Path | None:
        cached = image_paths.get(stem)
        if cached is not None and cached.is_file():
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
                if candidate.is_file() and candidate.suffix.casefold() in _WSI_EXTENSIONS and candidate.stem.casefold() == target:
                    image_paths[stem] = candidate
                    return candidate
        except OSError:
            return None
        return None

    tile_servers: dict[str, object] = {}

    def _server(stem: str):
        if stem not in slide_stems:
            raise HTTPException(404, "Slide not found")
        if stem not in tile_servers:
            path = _resolve_wsi_path(stem)
            if path is None:
                raise HTTPException(404, f"WSI file not found for '{stem}'")
            try:
                from app.wsi import WSITileServer
            except ImportError as exc:
                raise HTTPException(503, "WSI tile backend is not installed in this viewer package") from exc
            try:
                tile_servers[stem] = WSITileServer(path)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(422, f"Could not open WSI '{stem}': {exc}") from exc
        return tile_servers[stem]

    app = FastAPI(title="segmenteer viewer", docs_url=None, redoc_url=None)
    app.state.index_data = index_data

    @app.middleware("http")
    async def _disable_viewer_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/") or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/index")
    def get_index() -> JSONResponse:
        return JSONResponse(index_payload, headers={"Cache-Control": "no-store"})

    @app.get("/api/wsi/{stem}/info")
    def get_wsi_info(stem: str) -> JSONResponse:
        server = _server(stem)
        return JSONResponse({
            "stem": stem,
            "width": server.width,
            "height": server.height,
            "mpp": server.mpp,
            "n_levels": server.n_levels,
            "backend": server.backend,
            "tile_read_warning": server.last_error,
        })

    @app.get("/api/wsi/{stem}/thumbnail.jpeg")
    def get_thumbnail(stem: str, max_size: int = 1024) -> Response:
        if stem not in slide_stems:
            raise HTTPException(404, "Slide not found")
        try:
            content = _server(stem).get_thumbnail(max_size=max_size)
        except Exception as exc:
            raise HTTPException(422, f"Could not render WSI thumbnail: {exc}") from exc
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
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

    def _read_geojson(path: Path, label: str) -> dict:
        if not path.is_file():
            raise HTTPException(404, f"{label} file not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, f"Could not read {label} GeoJSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(500, f"Invalid {label} GeoJSON")
        return payload

    def _add_metadata(geojson: dict, *, stem: str, layer: str, run_id: str | None = None) -> JSONResponse:
        scale_to_viewport: float | None = None
        try:
            scale_to_viewport = 1.0 / _server(stem).width
        except HTTPException:
            pass
        payload = json.loads(json.dumps(geojson))
        payload["_meta"] = {
            "run_id": run_id,
            "stem": stem,
            "layer": layer,
            "coordinate_space": "level_0_pixels",
            "scale_to_viewport": scale_to_viewport,
        }
        return JSONResponse(payload)

    @app.get("/api/overlay/{stem}/{run_id}")
    def get_overlay(stem: str, run_id: str) -> JSONResponse:
        if stem not in slide_stems:
            raise HTTPException(404, "Slide not found")
        if run_id not in method_ids:
            raise HTTPException(404, "Method not found")
        path = output_path / run_id / "predictions" / f"{stem}.geojson"
        return _add_metadata(_read_geojson(path, "prediction"), stem=stem, run_id=run_id, layer="prediction")

    @lru_cache(maxsize=4096)
    def _ground_truth_geojson(stem: str) -> dict:
        if gt_path is None:
            raise HTTPException(404, "No ground-truth directory was provided")
        return _read_geojson(gt_path / f"{stem}.geojson", "ground truth")

    @app.get("/api/ground-truth/{stem}")
    def get_ground_truth(stem: str) -> JSONResponse:
        if stem not in slide_stems:
            raise HTTPException(404, "Slide not found")
        return _add_metadata(_ground_truth_geojson(stem), stem=stem, layer="ground_truth")

    # TP / FP / FN are transient inspection layers. They are computed lazily by
    # VIEWER for only the selected (slide, method) pair, cached in memory, and
    # never written to disk. Numeric benchmark metrics remain EVALUATOR-owned.
    if gt_path is not None and index_data.ground_truth_available:
        def _polygon_union(payload: dict):
            try:
                from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
                from shapely.ops import unary_union
                try:
                    from shapely import make_valid
                except ImportError:  # Shapely < 2
                    make_valid = None
            except ImportError as exc:
                raise HTTPException(503, "Spatial comparison requires shapely") from exc

            polygons = []

            def _parts(geometry):
                if geometry is None or geometry.is_empty:
                    return []
                if isinstance(geometry, Polygon):
                    return [geometry]
                if isinstance(geometry, MultiPolygon):
                    return [part for part in geometry.geoms if not part.is_empty]
                output = []
                for member in getattr(geometry, "geoms", []):
                    output.extend(_parts(member))
                return output

            for feature in payload.get("features", []):
                raw = feature.get("geometry") if isinstance(feature, dict) else None
                if not isinstance(raw, dict):
                    continue
                try:
                    geometry = shape(raw)
                    if not geometry.is_valid:
                        geometry = make_valid(geometry) if make_valid is not None else geometry.buffer(0)
                except Exception:
                    continue
                polygons.extend(_parts(geometry))
            if not polygons:
                return GeometryCollection()
            merged = unary_union(polygons)
            if not merged.is_valid:
                merged = make_valid(merged) if make_valid is not None else merged.buffer(0)
            return merged

        def _geometry_feature_collection(geometry, *, layer: str, stem: str, run_id: str) -> dict:
            from shapely.geometry import MultiPolygon, Polygon, mapping

            def _parts(value):
                if value is None or value.is_empty:
                    return []
                if isinstance(value, Polygon):
                    return [value]
                if isinstance(value, MultiPolygon):
                    return [part for part in value.geoms if not part.is_empty]
                output = []
                for member in getattr(value, "geoms", []):
                    output.extend(_parts(member))
                return output

            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(polygon),
                        "properties": {
                            "part_of_group": layer,
                            "source_slide": stem,
                            "segmentation_method": run_id,
                            "produced_by": "segmenteer.viewer.transient_spatial_comparison",
                        },
                    }
                    for polygon in _parts(geometry)
                ],
            }

        @lru_cache(maxsize=256)
        def _spatial_comparison(stem: str, run_id: str) -> dict[str, dict]:
            if stem not in slide_stems:
                raise HTTPException(404, "Slide not found")
            if run_id not in method_ids:
                raise HTTPException(404, "Method not found")
            gt_file = gt_path / f"{stem}.geojson"
            pred_file = output_path / run_id / "predictions" / f"{stem}.geojson"
            ground_truth = _read_geojson(gt_file, "ground truth")
            prediction = _read_geojson(pred_file, "prediction")
            try:
                gt_union = _polygon_union(ground_truth)
                pred_union = _polygon_union(prediction)
                geometries = {
                    "true_positive": pred_union.intersection(gt_union),
                    "false_positive": pred_union.difference(gt_union),
                    "false_negative": gt_union.difference(pred_union),
                }
            except Exception as exc:
                raise HTTPException(422, f"Could not compute spatial comparison: {exc}") from exc
            return {
                layer: _geometry_feature_collection(geometry, layer=layer, stem=stem, run_id=run_id)
                for layer, geometry in geometries.items()
            }

        @app.get("/api/spatial-comparison/{stem}/{run_id}/{layer}")
        def get_spatial_comparison_layer(stem: str, run_id: str, layer: str) -> JSONResponse:
            if layer not in {"true_positive", "false_positive", "false_negative"}:
                raise HTTPException(400, "layer must be true_positive, false_positive, or false_negative")
            payload = _spatial_comparison(stem, run_id)[layer]
            return _add_metadata(payload, stem=stem, run_id=run_id, layer=layer)

    # Metrics are evaluator artifacts. The viewer never computes evaluation
    # geometry or metrics from predictions + ground truth. Summary endpoints
    # exist only when a metrics file yielded usable metric values.
    if index_data.metrics_available:
        @lru_cache(maxsize=1)
        def _summary_payload_cached() -> dict:
            from app.summary import summary_payload
            return summary_payload(index_data)

        @lru_cache(maxsize=1)
        def _summary_csv_cached() -> str:
            from app.summary import summary_csv
            return summary_csv(index_data)

        @app.get("/api/summary")
        def get_summary() -> JSONResponse:
            return JSONResponse(_summary_payload_cached(), headers={"Cache-Control": "no-store"})

        @app.get("/api/summary.csv")
        def get_summary_csv() -> Response:
            filename = f"segmenteer_metrics_summary_{output_path.name}.csv"
            return Response(
                content=_summary_csv_cached().encode("utf-8"),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"},
            )

    return app


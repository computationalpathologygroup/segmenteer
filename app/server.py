"""FastAPI application — tile serving, overlay API, and static files."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.loader import IndexData, load_index
from app.wsi import WSITileServer

_HERE = Path(__file__).parent


def create_app(output_dir: str, data_dir: Optional[str] = None) -> FastAPI:
    output_path = Path(output_dir).resolve()
    data_path = Path(data_dir).resolve() if data_dir else None

    index: IndexData = load_index(output_path, data_path)
    # Build lookup: stem → WSI path
    _image_paths: dict[str, Path] = {
        w.stem: Path(w.image_path) for w in index.wsis if w.image_path
    }
    _tile_servers: dict[str, WSITileServer] = {}

    def _server(stem: str) -> WSITileServer:
        if stem not in _tile_servers:
            if stem not in _image_paths:
                raise HTTPException(404, f"WSI file not found for '{stem}'")
            _tile_servers[stem] = WSITileServer(_image_paths[stem])
        return _tile_servers[stem]

    # ─────────────────────────────────────────────────────────────────
    app = FastAPI(title="segmenteer viewer", docs_url=None, redoc_url=None)
    app.mount(
        "/static",
        StaticFiles(directory=str(_HERE / "static")),
        name="static",
    )

    # ── UI shell ─────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return HTMLResponse(
            (_HERE / "static" / "index.html").read_text(encoding="utf-8")
        )

    # ── Data index ───────────────────────────────────────────────────

    @app.get("/api/index")
    def get_index() -> JSONResponse:
        return JSONResponse(dataclasses.asdict(index))

    # ── Thumbnail ────────────────────────────────────────────────────

    @app.get("/api/thumbnail/{stem}")
    def get_thumbnail(stem: str) -> Response:
        thumb = output_path / "thumbnails" / f"{stem}.png"
        if not thumb.exists():
            raise HTTPException(404, "Thumbnail not found")
        return Response(content=thumb.read_bytes(), media_type="image/png")

    # ── WSI metadata ─────────────────────────────────────────────────

    @app.get("/api/wsi/{stem}/info")
    def get_wsi_info(stem: str) -> JSONResponse:
        srv = _server(stem)
        return JSONResponse(
            {
                "stem": stem,
                "width": srv.width,
                "height": srv.height,
                "mpp": srv.mpp,
                "n_levels": srv.n_levels,
            }
        )

    # ── DZI descriptor ───────────────────────────────────────────────

    @app.get("/api/wsi/{stem}/dzi")
    def get_dzi(stem: str) -> Response:
        srv = _server(stem)
        return Response(content=srv.dzi_descriptor(), media_type="application/xml")

    # ── DZI tile ─────────────────────────────────────────────────────
    # Path follows the DZI convention: {dzi_url}_files/{level}/{x}_{y}.jpeg
    # OSD constructs this automatically when given the .dzi URL string.

    @app.get("/api/wsi/{stem}/dzi_files/{level}/{tile}.jpeg")
    def get_tile(stem: str, level: int, tile: str) -> Response:
        try:
            x_s, y_s = tile.split("_")
            x, y = int(x_s), int(y_s)
        except ValueError:
            raise HTTPException(400, "Tile coordinate must be '{x}_{y}'")
        srv = _server(stem)
        return Response(content=srv.get_tile(level, x, y), media_type="image/jpeg")

    # ── GeoJSON overlay ──────────────────────────────────────────────

    @app.get("/api/overlay/{stem}/{run_id}")
    def get_overlay(stem: str, run_id: str) -> JSONResponse:
        """Return GeoJSON with ``_meta.scale_to_viewport``.

        GeoJSON coordinates are stored in level-0 pixel space (x ∈ [0, W],
        y ∈ [0, H]).  OSD viewport normalises x to [0, 1] spanning the full
        slide width.  So the only transform needed is: coord / slide_width.
        """
        pred_file = output_path / run_id / "predictions" / f"{stem}.geojson"
        if not pred_file.exists():
            raise HTTPException(404, "Prediction file not found")

        geojson = json.loads(pred_file.read_text(encoding="utf-8"))

        scale_to_viewport: float | None = None
        if stem in _image_paths:
            try:
                scale_to_viewport = 1.0 / _server(stem).width
            except HTTPException:
                pass

        geojson["_meta"] = {
            "run_id": run_id,
            "stem": stem,
            "scale_to_viewport": scale_to_viewport,
        }
        return JSONResponse(geojson)

    # ── HTML report ──────────────────────────────────────────────────

    @app.get("/api/report")
    def get_report() -> Response:
        from app.report import generate_report

        html_content = generate_report(index, output_path)
        filename = f"segmenteer_report_{output_path.name}.html"
        return Response(
            content=html_content.encode("utf-8"),
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    # ─────────────────────────────────────────────────────────────────
    return app

"""Generate a self-contained HTML benchmark report from ``IndexData``.

Layout: rows = slides (images), columns = methods.
Each cell renders the slide thumbnail with the tissue-mask polygon overlay
drawn in green on a <canvas> element via embedded JavaScript.
The resulting HTML is fully self-contained:
  - thumbnails embedded as base64 data URIs
  - GeoJSON polygon rings serialised as inline JSON
  - fonts loaded from Google Fonts CDN (falls back offline)
"""

from __future__ import annotations

import base64
import html
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.loader import IndexData, MethodInfo

# ── Metric catalogue ─────────────────────────────────────────────────────────

_KEYS: list[str] = [
    "coverage_ratio",
    "mean_compactness",
    "mean_solidity",
    "num_objects",
    "mean_area",
    "execution_time_s",
]
_LABELS: dict[str, str] = {
    "coverage_ratio":   "Coverage",
    "mean_compactness": "Compactness",
    "mean_solidity":    "Solidity",
    "num_objects":      "Objects",
    "mean_area":        "Area",
    "execution_time_s": "Time",
}
_ABBR: dict[str, str] = {
    "coverage_ratio":   "COV",
    "mean_compactness": "CMP",
    "mean_solidity":    "SLD",
    "num_objects":      "OBJ",
    "mean_area":        "AREA",
    "execution_time_s": "TIME",
}
_HI: frozenset[str] = frozenset({"coverage_ratio", "mean_compactness", "mean_solidity", "mean_area"})
_LO: frozenset[str] = frozenset({"execution_time_s"})

_CELL_PX: int = 160  # canvas width/height in logical pixels


# ── Metric helpers ────────────────────────────────────────────────────────────


def _get(entry: dict[str, Any], key: str) -> float | None:
    for v in (
        entry.get("metrics", {}).get("unsupervised", {}).get(key),
        entry.get("metrics", {}).get(key),
        entry.get(key),
    ):
        if v is not None:
            return float(v)
    return None


def _fmt(val: float | None, key: str) -> str:
    if val is None:
        return "\u2014"
    if key == "coverage_ratio":
        return f"{val * 100:.1f}%"
    if key == "num_objects":
        return str(int(round(val)))
    if key == "mean_area":
        if val >= 1_000_000:
            return f"{val / 1e6:.1f}\u202fM"
        if val >= 1_000:
            return f"{val / 1_000:.1f}\u202fk"
        return str(int(round(val)))
    if key == "execution_time_s":
        return f"{val:.1f}\u202fs"
    if 0 < val < 0.01:
        return f"{val:.2e}"
    return f"{val:.3f}"


def _winners(vals: list[float | None], key: str) -> set[int]:
    pairs = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(pairs) < 2:
        return set()
    if key in _HI:
        best = max(v for _, v in pairs)
    elif key in _LO:
        best = min(v for _, v in pairs)
    else:
        return set()
    return {i for i, v in pairs if v == best}


# ── Thumbnail helper ──────────────────────────────────────────────────────────


def _thumb(stem: str, output_path: Path) -> tuple[str | None, int, int]:
    """Return (data_uri, width_px, height_px) for the PNG thumbnail."""
    p = output_path / "thumbnails" / f"{stem}.png"
    if not p.exists():
        return None, 0, 0
    raw = p.read_bytes()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    try:
        from PIL import Image as _PIL
        w, h = _PIL.open(p).size
    except Exception:
        w = h = 0
    return uri, w, h


# ── Geometry helpers ──────────────────────────────────────────────────────────


def _all_coords(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    gtype = geometry.get("type", "")
    raw   = geometry.get("coordinates", [])
    depth = {
        "Point": 0, "MultiPoint": 1, "LineString": 1,
        "MultiLineString": 2, "Polygon": 2, "MultiPolygon": 3,
    }.get(gtype, 2)
    out: list[tuple[float, float]] = []

    def _walk(c: Any, d: int) -> None:
        if d == 0:
            out.append((float(c[0]), float(c[1])))
        else:
            for item in c:
                _walk(item, d - 1)

    _walk(raw, depth)
    return out


def _exterior_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    gtype  = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    rings: list[list[list[float]]] = []
    if gtype == "Polygon" and coords:
        rings.append([[c[0], c[1]] for c in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([[c[0], c[1]] for c in poly[0]])
    return rings


def _simplify_rings(
    rings: list[list[list[float]]], tol: float
) -> list[list[list[float]]]:
    if tol <= 0 or not rings:
        return rings
    try:
        from shapely.geometry import Polygon as _Poly  # type: ignore[import]
        out: list[list[list[float]]] = []
        for ring in rings:
            if len(ring) < 4:
                out.append(ring)
                continue
            try:
                p = _Poly(ring).simplify(tol, preserve_topology=True)
                if not p.is_empty:
                    out.append([[c[0], c[1]] for c in p.exterior.coords])
            except Exception:
                out.append(ring)
        return out
    except ImportError:
        return rings


# ── JSON payload ──────────────────────────────────────────────────────────────


def _build_payload(index: IndexData, output_path: Path) -> dict[str, Any]:
    """Build the JSON payload embedded in the HTML as ``const R = ...``."""
    methods = list(index.methods.values())
    run_ids = [m.run_id for m in methods]
    slides_out: list[dict[str, Any]] = []

    for w in index.wsis:
        stem = w.stem
        uri, tw, th = _thumb(stem, output_path)

        # Infer WSI pixel dimensions from GeoJSON coordinate extents
        ww = wh = 0.0
        for rid in run_ids:
            pred = output_path / rid / "predictions" / f"{stem}.geojson"
            if not pred.exists():
                continue
            try:
                gj = json.loads(pred.read_text(encoding="utf-8"))
                for feat in gj.get("features", []):
                    for x, y in _all_coords(feat.get("geometry", {})):
                        if x > ww:
                            ww = x
                        if y > wh:
                            wh = y
            except Exception:
                pass

        if ww > 0:
            ww *= 1.02
        if wh > 0:
            wh *= 1.02
        tol = max(ww, wh) / _CELL_PX if max(ww, wh) > 0 else 0

        stem_scores = index.scores.get(stem, {})
        cells: dict[str, Any] = {}

        for m in methods:
            rid  = m.run_id
            pred = output_path / rid / "predictions" / f"{stem}.geojson"
            rings: list[list[list[float]]] = []
            if pred.exists():
                try:
                    gj  = json.loads(pred.read_text(encoding="utf-8"))
                    raw: list[list[list[float]]] = []
                    for feat in gj.get("features", []):
                        raw.extend(_exterior_rings(feat.get("geometry", {})))
                    rings = _simplify_rings(raw, tol)
                except Exception:
                    pass
            entry  = stem_scores.get(rid, {})
            scores = {k: _get(entry, k) for k in _KEYS}
            cells[rid] = {"rings": rings, "scores": scores}

        slides_out.append({
            "stem": stem,
            "thumb": uri,
            "tw": tw,
            "th": th,
            "ww": ww,
            "wh": wh,
            "cells": cells,
        })

    return {
        "cell_px": _CELL_PX,
        "metric_keys": _KEYS,
        "metric_abbr": _ABBR,
        "methods": [
            {"run_id": m.run_id, "name": m.name,
             "color": m.color, "params": m.params}
            for m in methods
        ],
        "slides": slides_out,
    }


# ── HTML table builders ────────────────────────────────────────────────────────


def _chip_html(idx: int, entry: dict[str, Any], win_sets: dict[str, set[int]]) -> str:
    parts: list[str] = []
    for k in _KEYS:
        val = _get(entry, k)
        wc  = " win" if idx in win_sets.get(k, set()) else ""
        parts.append(
            f'<span class="mc{wc}">'
            f'<span class="mk">{html.escape(_ABBR[k])}</span>'
            f'<span class="mv">{html.escape(_fmt(val, k))}</span>'
            f'</span>'
        )
    return "".join(parts)


def _summary_row(
    methods: list[MethodInfo],
    wsis: list,
    scores: dict[str, dict[str, dict[str, Any]]],
) -> str:
    means: list[dict[str, float | None]] = []
    for m in methods:
        d: dict[str, float | None] = {}
        for k in _KEYS:
            vals = [
                _get(scores[w.stem][m.run_id], k)
                for w in wsis
                if w.stem in scores and m.run_id in scores[w.stem]
            ]
            valid = [v for v in vals if v is not None]
            d[k] = statistics.mean(valid) if valid else None
        means.append(d)

    win_sets = {
        k: _winners([means[i].get(k) for i in range(len(methods))], k)
        for k in _KEYS
    }

    cells: list[str] = []
    for i, m in enumerate(methods):
        parts: list[str] = []
        for k in _KEYS:
            val = means[i].get(k)
            wc  = " win" if i in win_sets.get(k, set()) else ""
            parts.append(
                f'<span class="mc{wc}">'
                f'<span class="mk">{html.escape(_ABBR[k])}</span>'
                f'<span class="mv">{html.escape(_fmt(val, k))}</span>'
                f'</span>'
            )
        cells.append(
            f'<td class="cm">'
            f'<div class="chips">{"".join(parts)}</div>'
            f'</td>'
        )

    first = (
        '<td class="cs sticky sum-first">'
        '<div class="sum-inner">'
        '<span class="sum-tag">avg</span>'
        '<span class="sum-nm">Dataset\u202fmean</span>'
        '</div>'
        '</td>'
    )
    return f'<tr class="sumrow">{first}{"".join(cells)}</tr>'


def _table_html(
    methods: list[MethodInfo],
    wsis: list,
    scores: dict[str, dict[str, dict[str, Any]]],
    thumb_map: dict[str, str | None],
) -> str:
    img_th = '<th class="cs sticky th-img"><span class="th-label">Slide</span></th>'
    mth_ths: list[str] = []
    for m in methods:
        prm = " &middot; ".join(
            f"{html.escape(k)}={html.escape(v)}" for k, v in m.params.items()
        )
        prm_html = f'<div class="mprm">{prm}</div>' if prm else ""
        mth_ths.append(
            f'<th class="cm">'
            f'<div class="mhdr">'
            f'<span class="mnm">{html.escape(m.name)}</span>'
            f'</div>'
            f'{prm_html}'
            f'</th>'
        )
    thead = f'<thead><tr>{img_th}{"".join(mth_ths)}</tr></thead>'

    tbody_rows: list[str] = [_summary_row(methods, wsis, scores)]

    for si, w in enumerate(wsis):
        stem        = w.stem
        stem_scores = scores.get(stem, {})
        uri         = thumb_map.get(stem)

        win_sets = {
            k: _winners(
                [_get(stem_scores.get(m.run_id, {}), k) for m in methods], k
            )
            for k in _KEYS
        }

        thumb_tag = (
            f'<img class="sl-img" src="{uri}" alt="{html.escape(stem)}">'
            if uri
            else '<div class="sl-no-img"></div>'
        )
        first = (
            f'<td class="cs sticky">'
            f'{thumb_tag}'
            f'<span class="sl-nm" title="{html.escape(stem)}">{html.escape(stem)}</span>'
            f'</td>'
        )

        cells: list[str] = []
        for i, m in enumerate(methods):
            entry   = stem_scores.get(m.run_id, {})
            rid_esc = html.escape(m.run_id)
            chips   = _chip_html(i, entry, win_sets)
            cells.append(
                f'<td class="cm">'
                f'<canvas class="cv" data-s="{si}" data-r="{rid_esc}"></canvas>'
                f'<div class="chips">{chips}</div>'
                f'</td>'
            )

        tbody_rows.append(f'<tr class="sr">{first}{"".join(cells)}</tr>')

    return f'{thead}<tbody>{"".join(tbody_rows)}</tbody>'



# ── Public API ────────────────────────────────────────────────────────────────


def generate_report(
    index: IndexData,
    output_path: Path,
) -> str:
    """Return a fully self-contained HTML report string."""
    methods   = list(index.methods.values())
    wsis      = index.wsis
    scores    = index.scores
    label     = Path(index.output_dir).name
    generated = datetime.now(timezone.utc).strftime("%Y\u2011%m\u2011%d\u2002%H:%M\u202fUTC")

    thumb_map: dict[str, str | None] = {
        w.stem: _thumb(w.stem, output_path)[0] for w in wsis
    }
    tbl   = _table_html(methods, wsis, scores, thumb_map)
    rdata = _build_payload(index, output_path)

    # Prevent </script> closing the tag while embedded in JSON
    r_json = json.dumps(rdata, separators=(",", ":"), ensure_ascii=False).replace(
        "</", "<\\/"
    )

    label_e   = html.escape(label)
    n_slides  = len(wsis)
    n_methods = len(methods)

    return (
        f'<!DOCTYPE html>\n'
        f'<html lang="en">\n'
        f'<head>\n'
        f'  <meta charset="utf-8">\n'
        f'  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>segmenteer \u00b7 {label_e}</title>\n'
        f'  <link rel="icon" href="data:image/svg+xml,<svg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 32 32\'><text x=\'1\' y=\'26\' font-family=\'Inter,system-ui,sans-serif\' font-weight=\'700\' font-size=\'22\' fill=\'%2302B0dd\'>sg</text></svg>">\n'
        f'  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
        f'  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900'
        f'&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">\n'
        f'  <style>{_CSS}</style>\n'
        f'</head>\n'
        f'<body>\n'
        f'<header>\n'
        f'  <div class="topbar-left">\n'
        f'    <span class="logo">segmenteer</span>\n'
        f'    <span class="run-label">{label_e}</span>\n'
        f'  </div>\n'
        f'  <div class="report-badge"><span>Report</span></div>\n'
        f'  <div class="topbar-right">\n'
        f'    <span class="stat-chip"><strong>{n_slides}</strong>\u2009slides</span>\n'
        f'    <span class="stat-chip"><strong>{n_methods}</strong>\u2009methods</span>\n'
        f'    <span class="stat-chip">{html.escape(generated)}</span>\n'
        f'  </div>\n'
        f'</header>\n'
        f'<div class="tbl-wrap" style="margin-top:var(--hdr-h)">\n'
        f'  <table class="cmp">{tbl}</table>\n'
        f'</div>\n'
        f'<footer>\n'
        f'  <div class="ftr">\n'
        f'    <span>Generated by <strong>segmenteer</strong></span>\n'
        f'    <span class="dim">{html.escape(generated)}</span>\n'
        f'  </div>\n'
        f'</footer>\n'
        f'<script>const R={r_json};</script>\n'
        f'<script>{_JS}</script>\n'
        f'</body>\n'
        f'</html>'
    )


# ── Embedded CSS ──────────────────────────────────────────────────────────────

_CSS = """\
:root{
  --bg:#f8fafc;
  --srf:#ffffff;
  --srf2:#f1f5f9;
  --b:#e2e8f0;
  --b2:#f1f5f9;
  --t:#0f172a;
  --t2:#475569;
  --t3:#94a3b8;
  --acc:#059669;
  --acc-dim:rgba(5,150,105,.08);
  --acc-glow:rgba(5,150,105,.25);
  --win:#059669;
  --hdr-h:48px;
  --slide-col:188px;
  --mono:'JetBrains Mono','Cascadia Code','Fira Code',monospace;
  --sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --radius:5px;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;font-family:var(--sans);background:var(--bg);color:var(--t);
  font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased}

/* ── Header — matches app topbar exactly ── */
header{
  position:fixed;top:0;left:0;right:0;z-index:100;
  height:var(--hdr-h);
  background:var(--srf);
  border-bottom:1px solid var(--b);
  display:flex;align-items:center;
  padding:0 16px;gap:16px;
  box-shadow:var(--shadow);
}
.topbar-left{display:flex;align-items:center;gap:10px;flex:1}
.topbar-right{display:flex;align-items:center;gap:8px}
.logo{font-weight:600;font-size:14px;letter-spacing:-.3px;color:#02B0dd}
.run-label{
  font-family:var(--mono);font-size:11px;color:var(--t3);
  background:var(--srf2);border:1px solid var(--b);
  border-radius:var(--radius);padding:2px 8px;
}
.report-badge{
  display:flex;gap:2px;background:var(--srf2);border:1px solid var(--b);
  border-radius:var(--radius);padding:2px;
}
.report-badge span{
  font-size:12px;font-weight:500;color:var(--t2);
  padding:4px 14px;border-radius:calc(var(--radius) - 1px);
  background:var(--srf);box-shadow:var(--shadow);
}
.stat-chip{
  font-size:11px;color:var(--t2);background:var(--srf2);
  border:1px solid var(--b);border-radius:12px;padding:2px 10px;
}
/* ── Scrollable table wrapper ── */
.tbl-wrap{
  overflow-x:auto;overflow-y:auto;
  height:calc(100vh - var(--hdr-h) - 38px);
  scrollbar-width:thin;scrollbar-color:rgba(15,23,42,.12) transparent;
}
.tbl-wrap::-webkit-scrollbar{width:6px;height:6px}
.tbl-wrap::-webkit-scrollbar-track{background:transparent}
.tbl-wrap::-webkit-scrollbar-thumb{background:rgba(15,23,42,.12);border-radius:3px}

/* ── Table ── */
.cmp{border-collapse:separate;border-spacing:0;min-width:max-content;font-size:12px}

/* ── Header row ── */
thead tr{background:var(--srf)}
thead th{
  position:sticky;top:0;z-index:100;
  padding:14px 16px 10px;
  text-align:left;
  background:var(--srf);
  border-bottom:1px solid var(--b);
  border-right:1px solid var(--b);
  white-space:nowrap;
}
thead th:last-child{border-right:none}
.th-label{font-size:10px;font-weight:700;text-transform:uppercase;
          letter-spacing:.8px;color:var(--t3)}
thead th.cm{border-top:2px solid var(--b)}
.mhdr{display:flex;align-items:center;margin-bottom:3px}
.mnm{font-size:12px;font-weight:600;color:var(--t);letter-spacing:-.1px}
.mprm{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:2px;
      padding-left:0;max-width:164px;white-space:normal}

/* ── Sticky first column ── */
.sticky{
  position:sticky;left:0;z-index:50;
  background:var(--bg);
  width:var(--slide-col);min-width:var(--slide-col);max-width:var(--slide-col);
}
thead th.sticky{z-index:150;background:var(--srf)}
.sticky::after{
  content:'';position:absolute;top:0;right:-1px;bottom:0;
  width:1px;background:var(--b);pointer-events:none;
}

/* ── Summary row ── */
.sumrow{background:rgba(5,150,105,.04)}
.sumrow .cs{background:rgba(248,249,252,.99)}
.sum-first{padding:0}
.sum-inner{
  padding:16px 16px;
  display:flex;align-items:center;gap:8px;
  height:100%;
}
.sum-tag{
  font-family:var(--mono);font-size:9px;font-weight:600;flex-shrink:0;
  color:var(--acc);background:var(--acc-dim);border:1px solid var(--acc-glow);
  border-radius:4px;padding:2px 6px;
}
.sum-nm{font-size:11px;font-weight:500;color:var(--t2)}
.sumrow td.cm{border-bottom:1px solid rgba(5,150,105,.08)}

/* ── Slide rows ── */
.sr .cs{background:var(--bg)}
.sr:hover td{background:rgba(15,23,42,.018)}
.sr:hover .cs{background:#f2f4f8}

/* cells */
td.cs{
  padding:14px 16px;
  border-bottom:1px solid var(--b);
  vertical-align:top;
}
td.cm{
  padding:14px 16px;
  border-bottom:1px solid var(--b);
  border-right:1px solid var(--b);
  vertical-align:top;
}
td.cm:last-child{border-right:none}
td.cm:hover{background:rgba(15,23,42,.012)}

/* slide thumbnail in sticky col */
.sl-img{
  display:block;width:160px;height:160px;object-fit:contain;
  border-radius:6px;border:1px solid var(--b);margin-bottom:8px;
  background:var(--srf2);
}
.sl-no-img{
  width:160px;height:160px;border-radius:6px;border:1px solid var(--b);
  background:var(--srf2);margin-bottom:8px;
}
.sl-nm{
  display:block;
  font-family:var(--mono);font-size:10px;font-weight:500;color:var(--t2);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  max-width:160px;
}

/* canvas */
.cv{
  display:block;width:160px;height:160px;
  border-radius:6px;background:var(--srf2);border:1px solid var(--b);
  margin-bottom:8px;
}

/* ── Metric chips ── */
.chips{display:flex;flex-wrap:wrap;gap:3px;max-width:160px}
.mc{
  display:inline-flex;align-items:center;gap:3px;
  padding:2px 6px 2px 5px;border-radius:4px;
  background:var(--srf2);border:1px solid var(--b);
  font-size:9px;white-space:nowrap;
}
.mc.win{background:var(--acc-dim);border-color:var(--acc-glow)}
.mk{font-family:var(--mono);font-weight:600;font-size:8px;letter-spacing:.3px;color:var(--t3)}
.mv{color:var(--t2)}
.mc.win .mv{color:var(--win);font-weight:700}
.mc.win .mk{color:rgba(5,150,105,.7)}

/* ── Footer ── */
footer{
  height:38px;background:var(--srf);border-top:1px solid var(--b);
  display:flex;align-items:center;
}
.ftr{
  width:100%;padding:0 28px;display:flex;justify-content:space-between;
  font-size:11px;color:var(--t3);
}
.ftr strong{color:var(--t2)}
.dim{color:var(--t3)}

/* ── Row / column crosshair highlight ── */
.row-hl td{background:rgba(37,99,235,.04) !important}
.row-hl .cs{background:rgba(37,99,235,.06) !important}
.col-hl{background:rgba(37,99,235,.04) !important}
thead th.col-hl{background:rgba(37,99,235,.07) !important}
/* intersection cell */
.row-hl .col-hl{background:rgba(37,99,235,.10) !important}

@media print{
  body{background:#fff;color:#111}
  .tbl-wrap{height:auto;overflow:visible}
  .sticky{position:static}
  thead th{position:static}
  .cmp{min-width:unset}
}
"""

# ── Embedded JavaScript ───────────────────────────────────────────────────────

_JS = """\
(function () {
  var S = R.cell_px;

  function containFit(ww, wh) {
    if (ww <= 0 || wh <= 0) return { scale: 1, ox: 0, oy: 0 };
    var sc = Math.min(S / ww, S / wh);
    return { scale: sc, ox: (S - ww * sc) / 2, oy: (S - wh * sc) / 2 };
  }

  function extentFromRings(rings) {
    var mx = 0, my = 0;
    for (var i = 0; i < rings.length; i++) {
      var r = rings[i];
      for (var j = 0; j < r.length; j++) {
        if (r[j][0] > mx) mx = r[j][0];
        if (r[j][1] > my) my = r[j][1];
      }
    }
    return { ww: mx * 1.02 || S, wh: my * 1.02 || S };
  }

  function drawRings(ctx, rings, sc, ox, oy) {
    if (!rings || !rings.length) return;
    ctx.save();
    ctx.strokeStyle = '#059669';
    ctx.lineWidth   = 1.5;
    ctx.fillStyle   = 'rgba(5,150,105,0.10)';
    ctx.shadowColor = 'rgba(5,150,105,0.35)';
    ctx.shadowBlur  = 4;
    for (var i = 0; i < rings.length; i++) {
      var ring = rings[i];
      if (ring.length < 3) continue;
      ctx.beginPath();
      ctx.moveTo(ring[0][0] * sc + ox, ring[0][1] * sc + oy);
      for (var j = 1; j < ring.length; j++) {
        ctx.lineTo(ring[j][0] * sc + ox, ring[j][1] * sc + oy);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  function renderCanvas(canvas) {
    var si   = parseInt(canvas.dataset.s, 10);
    var rid  = canvas.dataset.r;
    var slide = R.slides[si];
    if (!slide) return;
    var cell = slide.cells[rid];
    if (!cell) return;

    canvas.width  = S;
    canvas.height = S;
    var ctx = canvas.getContext('2d');

    ctx.fillStyle = '#f1f5f9';
    ctx.fillRect(0, 0, S, S);

    var rings = cell.rings || [];
    var ww = slide.ww > 0 ? slide.ww : extentFromRings(rings).ww;
    var wh = slide.wh > 0 ? slide.wh : extentFromRings(rings).wh;
    var fit = containFit(ww, wh);

    function paintOverlay() {
      drawRings(ctx, rings, fit.scale, fit.ox, fit.oy);
    }

    if (slide.thumb) {
      var img = new Image();
      img.onload = function () {
        ctx.drawImage(img, fit.ox, fit.oy, ww * fit.scale, wh * fit.scale);
        paintOverlay();
      };
      img.onerror = paintOverlay;
      img.src = slide.thumb;
    } else {
      paintOverlay();
    }
  }

  var io = new IntersectionObserver(function (entries) {
    for (var i = 0; i < entries.length; i++) {
      if (!entries[i].isIntersecting) continue;
      io.unobserve(entries[i].target);
      renderCanvas(entries[i].target);
    }
  }, { rootMargin: '220px' });

  var canvases = document.querySelectorAll('canvas.cv');
  for (var i = 0; i < canvases.length; i++) {
    io.observe(canvases[i]);
  }

  // ── Row + column crosshair highlight ────────────────────────────────
  var table = document.querySelector('table.cmp');
  if (table) {
    var allRows = table.querySelectorAll('tbody tr');
    var headCells = table.querySelectorAll('thead th');

    function clearHighlight() {
      for (var i = 0; i < allRows.length; i++) allRows[i].classList.remove('row-hl');
      for (var i = 0; i < headCells.length; i++) headCells[i].classList.remove('col-hl');
      var all = table.querySelectorAll('td.col-hl');
      for (var i = 0; i < all.length; i++) all[i].classList.remove('col-hl');
    }

    table.addEventListener('mouseover', function(e) {
      var td = e.target.closest('td');
      if (!td) { clearHighlight(); return; }
      var row = td.closest('tr');
      var colIdx = Array.prototype.indexOf.call(row.children, td);

      clearHighlight();

      // highlight row
      row.classList.add('row-hl');

      // highlight column (header + all body cells at same index)
      if (colIdx >= 0 && headCells[colIdx]) headCells[colIdx].classList.add('col-hl');
      for (var i = 0; i < allRows.length; i++) {
        var cell = allRows[i].children[colIdx];
        if (cell) cell.classList.add('col-hl');
      }
    });

    table.addEventListener('mouseleave', clearHighlight);
  }
})();
"""


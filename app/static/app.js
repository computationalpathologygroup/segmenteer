/**
 * segmenteer viewer
 *
 * Views
 * ──────
 *   overview  ── thumbnail grid; one compact score-row per WSI
 *   inspect   ── OpenSeadragon deep-zoom with per-method polygon overlays
 *
 * Coordinate chain (inspect overlay)
 * ────────────────────────────────────
 *   GeoJSON coord (level-0 pixel space, x ∈ [0, slide_W])
 *     × (1 / slide_W)                   → OSD viewport x ∈ [0, 1]
 *   viewportToViewerElementCoordinates() → canvas CSS pixel
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────────────────────

const S = {
  view:              "overview",
  index:             null,          // /api/index payload
  activeStem:        null,
  activeMethodSet:   new Set(),     // run_ids whose overlays are visible (inspect)
  overviewMethodSet: new Set(),     // run_ids whose overlays are visible (overview)
  overlayCache:      {},            // stem → { run_id → { features, scale } }
  overviewViewers:   {},            // stem → mini OSD Viewer (lazily created)
  viewer:            null,          // OSD Viewer (inspect)
  overlayCanvas:     null,          // <canvas> sibling of OSD canvas (inspect)
  metricKey:         "coverage_ratio",
};

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

async function init() {
  const res = await fetch("/api/index");
  if (!res.ok) {
    document.body.innerHTML = `<p style="padding:2rem;color:#f44">Failed to load index: ${res.statusText}</p>`;
    return;
  }
  S.index = await res.json();

  const { wsis, methods, output_dir } = S.index;
  document.getElementById("run-label").textContent = output_dir.split("/").at(-1);
  document.getElementById("wsi-count").textContent =
    `${wsis.length} slide${wsis.length !== 1 ? "s" : ""}`;
  const n = Object.keys(methods).length;
  document.getElementById("method-count").textContent =
    `${n} method${n !== 1 ? "s" : ""}`;

  buildSidebar();
  buildMetricSelect();
  buildOverviewToggles();
  renderOverview();

  document.querySelectorAll(".tab").forEach((btn) =>
    btn.addEventListener("click", () => switchView(btn.dataset.view))
  );

  document.getElementById("search").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll("#wsi-list li").forEach((li) =>
      li.classList.toggle("hidden", !li.dataset.stem.includes(q))
    );
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────────────────────────────────────

function buildSidebar() {
  const ul = document.getElementById("wsi-list");
  ul.innerHTML = "";
  for (const { stem, has_wsi } of S.index.wsis) {
    const li = document.createElement("li");
    li.dataset.stem = stem;
    li.innerHTML = `
      <span class="wsi-dot ${has_wsi ? "has-wsi" : ""}"></span>
      <span class="wsi-stem">${stem}</span>`;
    li.addEventListener("click", () => selectStem(stem));
    ul.appendChild(li);
  }
}

function updateSidebarActive(stem) {
  document.querySelectorAll("#wsi-list li").forEach((li) => {
    li.classList.toggle("active", li.dataset.stem === stem);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspect view
// ─────────────────────────────────────────────────────────────────────────────

async function selectStem(stem) {
  S.activeStem = stem;
  updateSidebarActive(stem);

  // Update card highlights in overview
  document.querySelectorAll(".wsi-card").forEach((c, i) => {
    c.classList.toggle("active", S.index.wsis[i]?.stem === stem);
  });

  // If in inspect view, open the WSI
  if (S.view === "inspect") {
    await openInspect(stem);
  }
}

async function openInspect(stem) {
  const wsiInfo = S.index.wsis.find((w) => w.stem === stem);
  const empty = document.getElementById("inspect-empty");
  const osdEl = document.getElementById("osd-viewer");

  // Update panel header
  document.getElementById("panel-stem").textContent = stem;
  document.getElementById("panel-overview").style.display = "none";
  document.getElementById("panel-inspect").style.display = "flex";

  // Reset overlay state for new stem
  S.activeMethodSet = new Set();

  buildMethodToggles(stem);
  buildMetricsTable(stem);

  if (!wsiInfo?.has_wsi) {
    empty.style.display = "flex";
    osdEl.style.opacity = "0";
    return;
  }

  empty.style.display = "none";
  osdEl.style.opacity = "1";

  // Destroy previous viewer
  if (S.viewer) {
    S.viewer.destroy();
    S.viewer = null;
    S.overlayCanvas = null;
  }

  // Open OpenSeadragon with DZI tile source
  // OSD fetches the XML descriptor, then auto-constructs tile URLs:
  //   /api/wsi/{stem}/dzi_files/{level}/{x}_{y}.jpeg
  S.viewer = OpenSeadragon({
    element: osdEl,
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
    tileSources: `/api/wsi/${stem}/dzi`,
    showNavigationControl: true,
    showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT",
    navigatorSizeRatio: 0.14,
    animationTime: 0.35,
    blendTime: 0.1,
    constrainDuringPan: false,
    maxZoomPixelRatio: 8,
    minZoomLevel: 0.05,
    zoomPerClick: 2,
    zoomPerScroll: 1.25,
    gestureSettingsMouse: { scrollToZoom: true, clickToZoom: false },
    backgroundColor: "#1a1a2e",
  });

  S.viewer.addHandler("open", () => {
    // Defer one frame so the browser has committed layout dimensions
    requestAnimationFrame(() => {
      _initOverlayCanvas();
      drawOverlays();
    });
  });
  // Use animation-finish for snappy redraws (fires after pan/zoom animation ends)
  S.viewer.addHandler("animation", scheduleRedraw);
  S.viewer.addHandler("animation-finish", drawOverlays);
  S.viewer.addHandler("update-viewport", scheduleRedraw);
  S.viewer.addHandler("resize", () => { _syncCanvas(); drawOverlays(); });
}

// ─────────────────────────────────────────────────────────────────────────────
// Method toggles (right panel)
// ─────────────────────────────────────────────────────────────────────────────

function buildMethodToggles(stem) {
  const container = document.getElementById("method-toggles");
  container.innerHTML = "";
  const { methods, scores } = S.index;

  for (const m of Object.values(methods)) {
    const hasData = !!scores[stem]?.[m.run_id];
    const chip    = document.createElement("div");
    chip.className     = "method-chip";
    chip.dataset.runId = m.run_id;
    chip.title         = m.run_id;

    const paramStr = Object.entries(m.params)
      .map(([k, v]) => `${k}=${v}`)
      .join(" · ");

    chip.innerHTML = `
      <span class="chip-swatch" style="background:${m.color}"></span>
      <span class="chip-name">${m.name}</span>
      <span class="chip-params">${paramStr || "—"}</span>
      <span class="chip-eye">${hasData ? "○" : "–"}</span>`;

    if (hasData) {
      chip.addEventListener("click", () => toggleMethod(stem, m.run_id, chip));
    } else {
      chip.style.opacity = "0.4";
      chip.style.cursor  = "default";
    }
    container.appendChild(chip);
  }
}

// Shared overlay fetch helper — populates overlayCache[stem][runId]
async function _fetchOverlay(stem, runId) {
  if (S.overlayCache[stem]?.[runId]) return true;
  try {
    const res = await fetch(`/api/overlay/${stem}/${encodeURIComponent(runId)}`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    S.overlayCache[stem] ??= {};
    S.overlayCache[stem][runId] = {
      features: data.features ?? [],
      scale:    data._meta?.scale_to_viewport ?? null,
    };
    return true;
  } catch (err) {
    console.warn(`Overlay fetch failed for ${stem}/${runId}:`, err);
    return false;
  }
}

async function toggleMethod(stem, runId, chipEl) {
  if (S.activeMethodSet.has(runId)) {
    S.activeMethodSet.delete(runId);
    chipEl.classList.remove("active");
    chipEl.querySelector(".chip-eye").textContent = "○";
    drawOverlays();
    return;
  }

  chipEl.querySelector(".chip-eye").textContent = "…";
  const ok = await _fetchOverlay(stem, runId);
  if (!ok) { chipEl.querySelector(".chip-eye").textContent = "✕"; return; }

  S.activeMethodSet.add(runId);
  chipEl.classList.add("active");
  chipEl.querySelector(".chip-eye").textContent = "●";
  drawOverlays();
}

// ─────────────────────────────────────────────────────────────────────────────
// Overview overlay toggles
// ─────────────────────────────────────────────────────────────────────────────

function buildOverviewToggles() {
  const bar = document.getElementById("overview-method-bar");
  bar.innerHTML = "";
  for (const m of Object.values(S.index.methods)) {
    const btn = document.createElement("button");
    btn.className = "ov-chip";
    btn.dataset.runId = m.run_id;
    btn.style.color = m.color;
    btn.innerHTML = `<span class="chip-swatch" style="background:${m.color}"></span>${m.name}`;
    btn.addEventListener("click", () => toggleOverviewMethod(m.run_id, btn));
    bar.appendChild(btn);
  }
}

async function toggleOverviewMethod(runId, btn) {
  if (S.overviewMethodSet.has(runId)) {
    S.overviewMethodSet.delete(runId);
    btn.classList.remove("active");
    _updateCardScoreVisibility();
    redrawOverviewCanvases();
    return;
  }

  // Add optimistically so the user sees immediate feedback
  btn.classList.add("active");
  S.overviewMethodSet.add(runId);
  _updateCardScoreVisibility();

  // Fetch overlay data for every stem in parallel
  const stems = S.index.wsis.map((w) => w.stem);
  await Promise.all(stems.map((stem) => _fetchOverlay(stem, runId)));
  redrawOverviewCanvases();
}

function _updateCardScoreVisibility() {
  const anyActive = S.overviewMethodSet.size > 0;
  document.querySelectorAll(".card-scores-wrap").forEach((wrap) => {
    wrap.style.display = anyActive ? "block" : "none";
  });
  document.querySelectorAll(".score-row[data-run-id]").forEach((row) => {
    row.style.display = S.overviewMethodSet.has(row.dataset.runId) ? "flex" : "none";
  });
}

function redrawOverviewCanvases() {
  document.querySelectorAll(".card-canvas").forEach((canvas) => {
    drawOverviewCard(canvas.dataset.stem, canvas);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini OSD viewers (overview cards)
// ─────────────────────────────────────────────────────────────────────────────

function _initCardOSD(stem, el) {
  if (S.overviewViewers[stem]) return;
  const viewer = OpenSeadragon({
    element: el,
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
    tileSources: `/api/wsi/${stem}/dzi`,
    showNavigationControl: false,
    showNavigator: false,
    animationTime: 0.2,
    blendTime: 0.05,
    constrainDuringPan: true,
    maxZoomPixelRatio: 6,
    minZoomLevel: 0.5,
    gestureSettingsMouse: { scrollToZoom: true, clickToZoom: false },
    backgroundColor: "#1a1a2e",
  });
  S.overviewViewers[stem] = viewer;

  viewer.addHandler("open", () => {
    requestAnimationFrame(() => _drawCardOverlays(stem));
  });
  viewer.addHandler("animation-finish", () => _drawCardOverlays(stem));
  viewer.addHandler("update-viewport",  () => _drawCardOverlays(stem));
  viewer.addHandler("resize",           () => _drawCardOverlays(stem));
}

function _getOrCreateCardCanvas(viewer) {
  const osdCanvas =
    viewer.drawer?.canvas ?? viewer.element.querySelector("canvas");
  if (!osdCanvas) return null;
  let cv = viewer.element.querySelector(".card-ov-canvas");
  if (!cv) {
    cv = document.createElement("canvas");
    cv.className = "card-ov-canvas";
    cv.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:10;";
    osdCanvas.parentElement.appendChild(cv);
  }
  return cv;
}

function _drawCardOverlays(stem) {
  const viewer = S.overviewViewers[stem];
  if (!viewer) return;
  const cv = _getOrCreateCardCanvas(viewer);
  if (!cv) return;

  const { clientWidth: w, clientHeight: h } = viewer.element;
  if (!w || !h) return;
  cv.width  = w;  cv.height  = h;
  cv.style.width  = `${w}px`;
  cv.style.height = `${h}px`;

  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  const { methods } = S.index;
  for (const runId of S.overviewMethodSet) {
    const entry = S.overlayCache[stem]?.[runId];
    if (!entry) continue;
    _drawFeatureCollection(ctx, entry.features, methods[runId]?.color ?? "#888", entry.scale, viewer);
  }
}

function redrawOverviewCanvases() {
  for (const stem of Object.keys(S.overviewViewers)) {
    _drawCardOverlays(stem);
  }
}

function _setupCardObserver() {
  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const stem  = entry.target.dataset.stem;
        const osdEl = entry.target.querySelector(".card-osd");
        if (osdEl && stem) _initCardOSD(stem, osdEl);
      }
    },
    { rootMargin: "150px" }
  );
  document.querySelectorAll(".wsi-card[data-stem]").forEach((c) => io.observe(c));
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas overlay rendering
// ─────────────────────────────────────────────────────────────────────────────

let _rafPending = false;

function scheduleRedraw() {
  if (_rafPending) return;
  _rafPending = true;
  requestAnimationFrame(() => { _rafPending = false; drawOverlays(); });
}

function _initOverlayCanvas() {
  if (!S.viewer) return;
  document.getElementById("osd-overlay-canvas")?.remove();

  // Append our canvas as a sibling of OSD's inner canvas —
  // both live in OSD's absolutely-positioned inner container.
  const osdCanvas =
    S.viewer.drawer?.canvas ??
    S.viewer.element.querySelector("canvas");
  if (!osdCanvas) { console.warn("OSD canvas not found"); return; }
  const canvas = document.createElement("canvas");
  canvas.id = "osd-overlay-canvas";
  canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;";
  osdCanvas.parentElement.appendChild(canvas);
  S.overlayCanvas = canvas;
  _syncCanvas();
}

function _syncCanvas() {
  if (!S.overlayCanvas || !S.viewer) return;
  // Use CSS pixel dimensions — viewportToViewerElementCoordinates returns CSS px
  const { clientWidth: w, clientHeight: h } = S.viewer.element;
  if (!w || !h) return;
  S.overlayCanvas.width        = w;
  S.overlayCanvas.height       = h;
  S.overlayCanvas.style.width  = `${w}px`;
  S.overlayCanvas.style.height = `${h}px`;
}

function drawOverlays() {
  if (!S.overlayCanvas || !S.viewer) return;
  const canvas = S.overlayCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const stem = S.activeStem;
  if (!stem || S.activeMethodSet.size === 0) return;

  const { methods } = S.index;
  for (const runId of S.activeMethodSet) {
    const entry = S.overlayCache[S.activeStem]?.[runId];
    if (!entry) continue;
    _drawFeatureCollection(ctx, entry.features, methods[runId]?.color ?? "#2E86AB", entry.scale);
  }
}

/** Batch all polygons for one method — one beginPath/stroke call per method. */
function _drawFeatureCollection(ctx, features, color, scale, viewer = S.viewer) {
  if (!features?.length || scale == null || !viewer) return;
  ctx.beginPath();
  for (const { geometry } of features) {
    if (!geometry) continue;
    if (geometry.type === "Polygon") {
      _tracePoly(ctx, geometry.coordinates[0], scale, viewer);
    } else if (geometry.type === "MultiPolygon") {
      for (const poly of geometry.coordinates) _tracePoly(ctx, poly[0], scale, viewer);
    }
  }
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.8;
  ctx.globalAlpha = 0.9;
  ctx.stroke();
  ctx.fillStyle   = _hexToRgba(color, 0.08);
  ctx.globalAlpha = 1;
  ctx.fill();
}

function _tracePoly(ctx, ring, scale, viewer = S.viewer) {
  if (!ring || ring.length < 3 || !viewer) return;
  let first = true;
  for (const [rx, ry] of ring) {
    const { x, y } = viewer.viewport.viewportToViewerElementCoordinates(
      new OpenSeadragon.Point(rx * scale, ry * scale)
    );
    if (first) { ctx.moveTo(x, y); first = false; }
    else         ctx.lineTo(x, y);
  }
  ctx.closePath();
}

// ─────────────────────────────────────────────────────────────────────────────
// View switching
// ─────────────────────────────────────────────────────────────────────────────

async function switchView(view) {
  S.view = view;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.view === view)
  );
  document.getElementById("view-overview").style.display =
    view === "overview" ? "" : "none";
  document.getElementById("view-inspect").style.display =
    view === "inspect" ? "" : "none";

  if (view === "inspect" && S.activeStem) {
    await openInspect(S.activeStem);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Overview grid
// ─────────────────────────────────────────────────────────────────────────────

const METRIC_LABELS = {
  coverage_ratio:   "Coverage",
  mean_compactness: "Compactness",
  mean_solidity:    "Solidity",
  num_objects:      "# Objects",
  mean_area:        "Mean area",
  execution_time_s: "Time (s)",
};

function buildMetricSelect() {
  const sel = document.getElementById("metric-select");
  for (const [k, v] of Object.entries(METRIC_LABELS)) {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = v;
    if (k === S.metricKey) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", (e) => {
    S.metricKey = e.target.value;
    renderOverview();
  });
}

/**
 * Each card shows: thumbnail · stem · compact score rows.
 * One row per method: ● name  value  (colour-coded by percentile rank).
 * No bars — too noisy with many methods.
 */
function renderOverview() {
  // Destroy any existing card OSD viewers before re-rendering
  for (const viewer of Object.values(S.overviewViewers)) viewer.destroy();
  S.overviewViewers = {};

  const grid = document.getElementById("overview-grid");
  grid.innerHTML = "";
  const { wsis, methods, scores } = S.index;
  const methodList = Object.values(methods);

  for (const { stem, thumbnail_url, has_wsi } of wsis) {
    const card = document.createElement("div");
    card.className = "wsi-card";
    card.dataset.stem = stem;
    if (S.activeStem === stem) card.classList.add("active");

    // Card thumbnail area: mini OSD for slides with tiles, static thumb otherwise
    const thumbInner = has_wsi
      ? `<div class="card-osd"></div>`
      : thumbnail_url
        ? `<img src="${thumbnail_url}" loading="lazy" alt="${stem}">`
        : `<span class="no-thumb">no thumbnail</span>`;

    const scoreRows = methodList.map((m) => {
      const val = _getMetric(scores, stem, m.run_id, S.metricKey);
      if (val === null) return "";
      const label = _fmtVal(val, S.metricKey);
      const visible = S.overviewMethodSet.has(m.run_id);
      return `<div class="score-row" data-run-id="${m.run_id}" style="display:${visible ? 'flex' : 'none'}">
        <span class="score-dot" style="background:${m.color}"></span>
        <span class="score-name" title="${m.run_id}">${m.name}</span>
        <span class="score-val">${label}</span>
      </div>`;
    }).join("");

    const anyVisible = S.overviewMethodSet.size > 0;
    card.innerHTML = `
      <div class="card-thumb">${thumbInner}</div>
      <div class="card-body">
        <div class="card-stem">${stem}</div>
        <div class="card-scores-wrap" style="display:${anyVisible ? 'block' : 'none'}">
          <div class="card-scores-label">unsupervised metrics</div>
          <div class="card-scores">${scoreRows || "<span class='no-scores'>no scores</span>"}</div>
        </div>
      </div>`;

    // Open inspect when clicking the label area; OSD handles its own pan/zoom
    card.querySelector(".card-body").addEventListener("click", (e) => {
      e.stopPropagation();
      selectStem(stem);
      switchView("inspect");
    });
    grid.appendChild(card);
  }

  _setupCardObserver();
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics table (right panel, inspect view)
// ─────────────────────────────────────────────────────────────────────────────

const UNSUPERVISED_COLS = [
  { key: "coverage_ratio",   label: "Coverage", bar: true  },
  { key: "mean_compactness", label: "Compact",  bar: true  },
  { key: "mean_solidity",    label: "Solidity", bar: true  },
  { key: "num_objects",      label: "Objects",  bar: false },
  { key: "mean_area",        label: "Area",     bar: false },
];

function buildMetricsTable(stem) {
  const container  = document.getElementById("metrics-table");
  const { methods, scores } = S.index;
  const methodList = Object.values(methods);

  const colMax = Object.fromEntries(
    UNSUPERVISED_COLS.map(({ key }) => [
      key,
      Math.max(...methodList.map((m) => _getMetric(scores, stem, m.run_id, key) ?? 0), 1e-9),
    ])
  );

  const headerCells = UNSUPERVISED_COLS.map(({ label }) => `<th>${label}</th>`).join("");

  const bodyRows = methodList.map((m) => {
    const cells = UNSUPERVISED_COLS.map(({ key, bar }) => {
      const val = _getMetric(scores, stem, m.run_id, key);
      if (val === null) return `<td class="metric-null">—</td>`;
      const display = _fmtVal(val, key);
      if (!bar) return `<td><span class="metric-val">${display}</span></td>`;
      const pct      = (val / colMax[key]) * 100;
      const barColor = _metricColor(pct);
      return `<td>
        <div class="metric-bar">
          <span class="metric-bar-val">${display}</span>
          <div class="metric-bar-inner">
            <div class="metric-bar-fill" style="width:${pct.toFixed(1)}%;background:${barColor}"></div>
          </div>
        </div></td>`;
    }).join("");

    return `<tr>
      <td><div class="method-cell">
        <span class="method-dot" style="background:${m.color}"></span>
        <span class="method-cell-name" title="${m.run_id}">${m.name}</span>
      </div></td>
      ${cells}
    </tr>`;
  }).join("");

  container.innerHTML = `
    <table class="metrics-tbl">
      <thead><tr><th>Method</th>${headerCells}</tr></thead>
      <tbody>${bodyRows}</tbody>
    </table>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function _getMetric(scores, stem, runId, key) {
  const entry = scores[stem]?.[runId];
  if (!entry) return null;
  const val =
    entry.metrics?.unsupervised?.[key] ??
    entry.metrics?.[key] ??
    entry[key] ??
    null;
  return val !== null && val !== undefined ? Number(val) : null;
}

function _fmtVal(val, key) {
  if (key === "num_objects") return String(Math.round(val));
  if (key === "mean_area")
    return val > 999_999
      ? `${(val / 1e6).toFixed(1)}M`
      : `${(val / 1000).toFixed(1)}k`;
  if (val > 0 && val < 0.01) return val.toExponential(1);
  return val.toFixed(3);
}

function _metricColor(pct) {
  if (pct >= 66) return "var(--good)";
  if (pct >= 33) return "var(--mid)";
  return "var(--bad)";
}

function _hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ─────────────────────────────────────────────────────────────────────────────
init();

/**
 * segmenteer official evaluation viewer.
 *
 * Geometry contract
 * -----------------
 * Predictions, GT, and on-demand TP/FP/FN layers are served as level-0 GeoJSON.
 * The API adds ``scale_to_viewport = 1 / slide_width`` so the browser draws
 * every layer in exactly the same OpenSeadragon coordinate space.
 */

"use strict";

const COLORS = {
  ground_truth: "#2563eb",   // blue
  true_positive: "#16a34a",  // green
  false_positive: "#dc2626", // red
  false_negative: "#f59e0b", // orange
};

const ERROR_LAYERS = [
  { key: "true_positive", label: "True positive", short: "TP" },
  { key: "false_positive", label: "False positive", short: "FP" },
  { key: "false_negative", label: "False negative", short: "FN" },
];

const EVALUATION_COLS = [
  { key: "dice", label: "Dice" },
  { key: "iou", label: "IoU" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "over_segmentation_rate", label: "Over-segmentation" },
  { key: "under_segmentation_rate", label: "Under-segmentation" },
];

const SUPERVISED_COLS = EVALUATION_COLS;
const UNSUPERVISED_COLS = [];

const METRIC_LABELS = Object.fromEntries(
  EVALUATION_COLS.map(({ key, label }) => [key, label])
);

const S = {
  view: "inspect",
  index: null,
  summary: null,
  activeStem: null,
  activeMethodSet: new Set(),
  overviewMethodSet: new Set(),
  overviewGroundTruthVisible: false,
  predictionCache: {},          // stem -> runId -> {features, scale}
  groundTruthCache: {},         // stem -> {features, scale}
  evaluationCache: {},          // stem -> runId -> kind -> {features, scale}
  overviewViewers: {},
  viewer: null,
  overlayCanvas: null,
  metricKey: "coverage_ratio",
  gtVisible: false,
  evaluationVisible: {
    true_positive: false,
    false_positive: false,
    false_negative: false,
  },
  evaluationRunId: null,
};

async function init() {
  try {
    const res = await fetch("/api/index");
    if (!res.ok) throw new Error(res.statusText);
    S.index = await res.json();
  } catch (err) {
    document.body.innerHTML = `<p style="padding:2rem;color:#dc2626">Failed to load viewer index: ${escapeHtml(String(err))}</p>`;
    return;
  }

  const { wsis, methods, output_dir: outputDir } = S.index;
  if (_hasAnySupervisedMetric()) S.metricKey = "dice";

  document.getElementById("run-label").textContent = outputDir.split("/").at(-1);
  // Do not display slide/method counts in the navbar; the official cohort size
  // is defined by the selected metrics CSVs and should not distract from review.

  // Official-results mode: keep first paint cheap. Do not build the overview
  // grid, do not load the summary table, and do not open any WSI/GeoJSON until
  // the user explicitly asks for the overview or a slide/layer.
  S.summary = null;
  buildSidebar();
  buildMetricSelect();
  buildOverviewToggles();
  buildOverviewSummary();
  await switchView("inspect");

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  document.getElementById("search").addEventListener("input", (event) => {
    const query = event.target.value.toLowerCase();
    document.querySelectorAll("#wsi-list li").forEach((item) => {
      item.classList.toggle("hidden", !item.dataset.stem.toLowerCase().includes(query));
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar and selection
// ─────────────────────────────────────────────────────────────────────────────

function buildSidebar() {
  const list = document.getElementById("wsi-list");
  list.innerHTML = "";
  for (const wsi of S.index.wsis) {
    const item = document.createElement("li");
    item.dataset.stem = wsi.stem;
    item.innerHTML = `
      <span class="wsi-dot ${wsi.has_wsi ? "has-wsi" : ""}"></span>
      <span class="wsi-stem" title="${escapeAttr(wsi.stem)}">${escapeHtml(wsi.stem)}</span>
      ${wsi.has_ground_truth ? '<span class="gt-badge" title="Valid ground truth used">GT</span>' : ""}
      ${_runStatusBadge(wsi, true)}`;
    item.addEventListener("click", () => selectStem(wsi.stem));
    list.appendChild(item);
  }
}

function _runStatusBadge(wsi, compact) {
  const info = {
    supervised: { label: compact ? "EVAL" : "evaluated", title: "Legacy result with supervised metrics" },
    ground_truth_available: { label: compact ? "GT" : "GT available", title: "Valid ground truth is saved; run evaluate_outputs.py to calculate metrics" },
    prediction_only: { label: compact ? "PRED" : "prediction-only", title: "Prediction-only: no usable ground truth was paired" },
    skipped_no_ground_truth: { label: compact ? "SKIP" : "skipped", title: "Skipped because no usable ground truth was paired" },
    not_run: { label: compact ? "—" : "not run", title: "No result was generated for this slide" },
  }[wsi.run_status] ?? { label: compact ? "—" : "unknown", title: "Run status unavailable" };
  return `<span class="run-status status-${escapeAttr(wsi.run_status ?? "unknown")}" title="${escapeAttr(info.title)}">${escapeHtml(info.label)}</span>`;
}

function updateSidebarActive(stem) {
  document.querySelectorAll("#wsi-list li").forEach((item) => {
    item.classList.toggle("active", item.dataset.stem === stem);
  });
}

async function selectStem(stem) {
  S.activeStem = stem;
  updateSidebarActive(stem);
  document.querySelectorAll(".wsi-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.stem === stem);
  });
  if (S.view === "inspect") await openInspect(stem);
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspect view
// ─────────────────────────────────────────────────────────────────────────────

async function openInspect(stem) {
  const wsiInfo = _getWsi(stem);
  if (!wsiInfo) return;

  document.getElementById("panel-stem").textContent = stem;
  document.getElementById("panel-overview").style.display = "none";
  document.getElementById("panel-inspect").style.display = "flex";

  // Official-results mode: opening a slide should not fetch polygons yet.
  // Predictions and ground truth are loaded only when their layer buttons are clicked.
  S.activeMethodSet = new Set();
  S.gtVisible = false;
  S.evaluationVisible = {
    true_positive: false,
    false_positive: false,
    false_negative: false,
  };
  S.evaluationRunId = _supervisedRunIds(stem)[0] ?? null;

  buildMethodToggles(stem);
  buildEvaluationControls(stem);
  buildMetricsTable(stem);

  _refreshMethodToggles();
  _refreshEvaluationControls(stem);

  const empty = document.getElementById("inspect-empty");
  const osdElement = document.getElementById("osd-viewer");
  if (!wsiInfo.has_wsi) {
    empty.style.display = "flex";
    osdElement.style.opacity = "0";
    drawOverlays();
    return;
  }

  empty.style.display = "none";
  osdElement.style.opacity = "1";
  if (S.viewer) {
    S.viewer.destroy();
    S.viewer = null;
    S.overlayCanvas = null;
  }

  S.viewer = OpenSeadragon({
    element: osdElement,
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
    tileSources: `/api/wsi/${encodeURIComponent(stem)}/dzi`,
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
    requestAnimationFrame(() => {
      _initOverlayCanvas();
      drawOverlays();
    });
  });
  S.viewer.addHandler("animation", scheduleRedraw);
  S.viewer.addHandler("animation-finish", drawOverlays);
  S.viewer.addHandler("update-viewport", scheduleRedraw);
  S.viewer.addHandler("resize", () => {
    _syncCanvas();
    drawOverlays();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction, GT, and error-layer controls
// ─────────────────────────────────────────────────────────────────────────────

function buildMethodToggles(stem) {
  const container = document.getElementById("method-toggles");
  container.innerHTML = "";
  for (const method of Object.values(S.index.methods)) {
    const hasData = Boolean(S.index.scores[stem]?.[method.run_id]);
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "method-chip";
    chip.dataset.runId = method.run_id;
    chip.title = method.run_id;

    const params = Object.entries(method.params)
      .map(([key, value]) => `${key}=${value}`)
      .join(" · ");
    chip.innerHTML = `
      <span class="chip-swatch" style="background:${method.color}"></span>
      <span class="chip-name">${escapeHtml(method.name)}</span>
      <span class="chip-params">${escapeHtml(params || "—")}</span>
      <span class="chip-eye">${hasData ? "○" : "–"}</span>`;

    if (hasData) {
      chip.addEventListener("click", () => togglePrediction(stem, method.run_id));
    } else {
      chip.disabled = true;
      chip.classList.add("disabled");
    }
    container.appendChild(chip);
  }
  _refreshMethodToggles();
}

function _refreshMethodToggles() {
  document.querySelectorAll("#method-toggles .method-chip").forEach((chip) => {
    const active = S.activeMethodSet.has(chip.dataset.runId);
    chip.classList.toggle("active", active);
    const eye = chip.querySelector(".chip-eye");
    if (eye && !chip.disabled) eye.textContent = active ? "●" : "○";
  });
}

async function togglePrediction(stem, runId) {
  if (S.activeMethodSet.has(runId)) {
    S.activeMethodSet.delete(runId);
    _refreshMethodToggles();
    drawOverlays();
    return;
  }
  const chip = document.querySelector(`#method-toggles .method-chip[data-run-id="${cssEscape(runId)}"]`);
  const eye = chip?.querySelector(".chip-eye");
  if (eye) eye.textContent = "…";
  const loaded = await _fetchPrediction(stem, runId);
  if (loaded) S.activeMethodSet.add(runId);
  if (eye && !loaded) eye.textContent = "✕";
  _refreshMethodToggles();
  drawOverlays();
}

function buildEvaluationControls(stem) {
  const wsi = _getWsi(stem);
  const groundTruthContainer = document.getElementById("ground-truth-toggle");
  const hasGroundTruth = Boolean(wsi?.has_ground_truth);
  groundTruthContainer.innerHTML = "";

  const gtButton = document.createElement("button");
  gtButton.type = "button";
  gtButton.id = "gt-layer-button";
  gtButton.className = "layer-chip";
  gtButton.style.color = COLORS.ground_truth;
  gtButton.innerHTML = `<span class="chip-swatch" style="background:${COLORS.ground_truth}"></span>Ground truth <span class="chip-eye">○</span>`;
  gtButton.disabled = !hasGroundTruth;
  if (!hasGroundTruth) gtButton.title = "This output does not contain saved ground truth. Re-run supervised evaluation to create it.";
  gtButton.addEventListener("click", () => toggleGroundTruth(stem));
  groundTruthContainer.appendChild(gtButton);

  const select = document.getElementById("evaluation-method-select");
  select.innerHTML = "";
  const supervisedRuns = _supervisedRunIds(stem);
  if (!supervisedRuns.length) {
    select.disabled = true;
    select.append(new Option("No supervised result", ""));
  } else {
    select.disabled = false;
    for (const runId of supervisedRuns) {
      const method = S.index.methods[runId];
      select.append(new Option(`${method?.name ?? runId} — ${runId}`, runId));
    }
    if (!S.evaluationRunId || !supervisedRuns.includes(S.evaluationRunId)) {
      S.evaluationRunId = supervisedRuns[0];
    }
    select.value = S.evaluationRunId;
  }
  select.onchange = async () => {
    S.evaluationRunId = select.value || null;
    await _loadVisibleEvaluationLayers(stem);
    _refreshEvaluationControls(stem);
    drawOverlays();
  };

  const errorsContainer = document.getElementById("evaluation-toggles");
  errorsContainer.innerHTML = "";
  for (const descriptor of ERROR_LAYERS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "layer-chip";
    button.dataset.layer = descriptor.key;
    button.style.color = COLORS[descriptor.key];
    button.innerHTML = `<span class="chip-swatch" style="background:${COLORS[descriptor.key]}"></span>${descriptor.label} <span class="chip-eye">○</span>`;
    button.disabled = !supervisedRuns.length;
    button.addEventListener("click", () => toggleEvaluationLayer(stem, descriptor.key));
    errorsContainer.appendChild(button);
  }
  _refreshEvaluationControls(stem);
}

async function toggleGroundTruth(stem) {
  if (!(_getWsi(stem)?.has_ground_truth)) return;
  if (S.gtVisible) {
    S.gtVisible = false;
  } else {
    const loaded = await _fetchGroundTruth(stem);
    if (loaded) S.gtVisible = true;
  }
  _refreshEvaluationControls(stem);
  drawOverlays();
}

async function toggleEvaluationLayer(stem, layer) {
  if (!S.evaluationRunId) return;
  if (S.evaluationVisible[layer]) {
    S.evaluationVisible[layer] = false;
  } else {
    const loaded = await _fetchEvaluationLayer(stem, S.evaluationRunId, layer);
    if (loaded) S.evaluationVisible[layer] = true;
  }
  _refreshEvaluationControls(stem);
  drawOverlays();
}

async function _loadVisibleEvaluationLayers(stem) {
  if (!S.evaluationRunId) return;
  const active = ERROR_LAYERS
    .filter(({ key }) => S.evaluationVisible[key])
    .map(({ key }) => _fetchEvaluationLayer(stem, S.evaluationRunId, key));
  await Promise.all(active);
}

function _refreshEvaluationControls(stem) {
  const gtButton = document.getElementById("gt-layer-button");
  if (gtButton) {
    gtButton.classList.toggle("active", S.gtVisible);
    const eye = gtButton.querySelector(".chip-eye");
    if (eye && !gtButton.disabled) eye.textContent = S.gtVisible ? "●" : "○";
  }

  document.querySelectorAll("#evaluation-toggles .layer-chip").forEach((button) => {
    const active = Boolean(S.evaluationVisible[button.dataset.layer]);
    button.classList.toggle("active", active);
    const eye = button.querySelector(".chip-eye");
    if (eye && !button.disabled) eye.textContent = active ? "●" : "○";
  });

  const status = document.getElementById("evaluation-status");
  const errorMapLink = document.getElementById("error-map-link");
  const score = S.evaluationRunId ? S.index.scores[stem]?.[S.evaluationRunId] : null;
  const wsi = _getWsi(stem);
  if (wsi?.run_status === "prediction_only") {
    status.textContent = "Prediction-only slide: no usable ground truth was paired.";
  } else if (wsi?.run_status === "skipped_no_ground_truth") {
    status.textContent = "Skipped because no usable ground truth was paired.";
  } else if (!score?.metrics?.supervised) {
    status.textContent = "No evaluation metrics are available for this method on this slide.";
  } else {
    status.textContent = "";
  }

  if (S.evaluationRunId && score?.evaluation_layers?.error_map) {
    errorMapLink.href = `/api/error-map/${encodeURIComponent(stem)}/${encodeURIComponent(S.evaluationRunId)}`;
    errorMapLink.classList.remove("hidden");
  } else {
    errorMapLink.classList.add("hidden");
    errorMapLink.removeAttribute("href");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeoJSON fetch/cache helpers
// ─────────────────────────────────────────────────────────────────────────────

function _entryFromGeojson(data) {
  return {
    features: data.features ?? [],
    scale: data._meta?.scale_to_viewport ?? null,
  };
}

async function _fetchGeojson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(response.statusText);
    return _entryFromGeojson(await response.json());
  } catch (error) {
    console.warn(`Overlay fetch failed: ${url}`, error);
    return null;
  }
}

async function _fetchPrediction(stem, runId) {
  if (S.predictionCache[stem]?.[runId]) return true;
  const entry = await _fetchGeojson(`/api/overlay/${encodeURIComponent(stem)}/${encodeURIComponent(runId)}`);
  if (!entry) return false;
  S.predictionCache[stem] ??= {};
  S.predictionCache[stem][runId] = entry;
  return true;
}

async function _fetchGroundTruth(stem) {
  if (S.groundTruthCache[stem]) return true;
  const entry = await _fetchGeojson(`/api/ground-truth/${encodeURIComponent(stem)}`);
  if (!entry) return false;
  S.groundTruthCache[stem] = entry;
  return true;
}

async function _fetchEvaluationLayer(stem, runId, layer) {
  if (S.evaluationCache[stem]?.[runId]?.[layer]) return true;
  const entry = await _fetchGeojson(
    `/api/evaluation/${encodeURIComponent(stem)}/${encodeURIComponent(runId)}/${encodeURIComponent(layer)}`
  );
  if (!entry) return false;
  S.evaluationCache[stem] ??= {};
  S.evaluationCache[stem][runId] ??= {};
  S.evaluationCache[stem][runId][layer] = entry;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Overview
// ─────────────────────────────────────────────────────────────────────────────

function buildOverviewToggles() {
  const groundTruthBar = document.getElementById("overview-ground-truth-bar");
  groundTruthBar.innerHTML = "";
  const hasGroundTruth = S.index.wsis.some((wsi) => wsi.has_ground_truth);
  const groundTruthButton = document.createElement("button");
  groundTruthButton.type = "button";
  groundTruthButton.className = `ov-chip${S.overviewGroundTruthVisible ? " active" : ""}`;
  groundTruthButton.style.color = COLORS.ground_truth;
  groundTruthButton.disabled = !hasGroundTruth;
  groundTruthButton.title = hasGroundTruth
    ? "Show or hide ground-truth contours in overview cards"
    : "No ground-truth annotations are available";
  groundTruthButton.innerHTML = `<span class="chip-swatch" style="background:${COLORS.ground_truth}"></span>Ground truth`;
  groundTruthButton.addEventListener("click", () => toggleOverviewGroundTruth(groundTruthButton));
  groundTruthBar.appendChild(groundTruthButton);

  const bar = document.getElementById("overview-method-bar");
  bar.innerHTML = "";
  for (const method of Object.values(S.index.methods)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ov-chip";
    button.dataset.runId = method.run_id;
    button.style.color = method.color;
    button.innerHTML = `<span class="chip-swatch" style="background:${method.color}"></span>${escapeHtml(method.name)}`;
    button.addEventListener("click", () => toggleOverviewMethod(method.run_id, button));
    bar.appendChild(button);
  }
}

async function toggleOverviewGroundTruth(button) {
  S.overviewGroundTruthVisible = !S.overviewGroundTruthVisible;
  button.classList.toggle("active", S.overviewGroundTruthVisible);
  // Do not bulk-fetch all ground-truth polygons in the overview. This viewer
  // is intended for interactive inspection; polygon GeoJSON is fetched only for
  // visible overview cards, and in the inspect view only after the user clicks.
  await _loadVisibleOverviewOverlays();
  redrawOverviewCanvases();
}

async function toggleOverviewMethod(runId, button) {
  if (S.overviewMethodSet.has(runId)) {
    S.overviewMethodSet.delete(runId);
    button.classList.remove("active");
    _updateCardScoreVisibility();
    redrawOverviewCanvases();
    return;
  }
  S.overviewMethodSet.add(runId);
  button.classList.add("active");
  _updateCardScoreVisibility();
  // Do not fetch prediction GeoJSON for every slide. Load only overlays for
  // currently instantiated overview cards; inspect-view layers are fetched on click.
  await _loadVisibleOverviewOverlays();
  redrawOverviewCanvases();
}

async function _loadOverviewSummary() {
  try {
    const response = await fetch("/api/summary", { cache: "no-store" });
    if (!response.ok) throw new Error(response.statusText);
    return await response.json();
  } catch (error) {
    console.warn("Could not load supervised summary", error);
    return null;
  }
}

function buildOverviewSummary() {
  const tableTarget = document.getElementById("overview-summary-table");
  if (!tableTarget) return;

  const summary = S.summary;
  if (!summary) {
    tableTarget.innerHTML = '<span class="summary-empty">Summary is not available yet.</span>';
    return;
  }

  const rows = Array.isArray(summary.rows) ? summary.rows : [];
  if (!rows.some((row) => row.n > 0)) {
    tableTarget.innerHTML = '<p class="panel-note">No completed evaluation metrics are available yet.</p>';
    return;
  }

  const byGroup = new Map();
  for (const row of rows) {
    const group = row.metric_group ?? "Metrics";
    if (!byGroup.has(group)) byGroup.set(group, []);
    byGroup.get(group).push(row);
  }
  const methods = Object.values(S.index.methods);
  tableTarget.innerHTML = [...byGroup.entries()]
    .map(([group, groupRows]) => _overviewSummaryTableHtml(group, groupRows, methods))
    .join("");
}

function _overviewSummaryTableHtml(group, rows, methods) {
  const metricRows = [...new Map(rows.map((row) => [row.metric_key, row])).values()];
  const byMetricAndMethod = new Map(rows.map((row) => [`${row.metric_key}::${row.run_id}`, row]));
  const header = metricRows.map((metric) =>
    `<th>${escapeHtml(metric.metric_label)}</th>`
  ).join("");
  const body = methods.map((method) => {
    const cells = metricRows.map((metric) => {
      const row = byMetricAndMethod.get(`${metric.metric_key}::${method.run_id}`);
      return `<td>${_overviewSummaryValueHtml(row)}</td>`;
    }).join("");
    return `<tr><th scope="row" title="${escapeAttr(method.run_id)}"><span class="summary-method-dot" style="background:${method.color}"></span>${escapeHtml(method.name)}</th>${cells}</tr>`;
  }).join("");
  return `<div class="overview-summary-group">
    <div class="overview-summary-table-wrap">
      <table class="overview-summary-table">
        <thead><tr><th>Method</th>${header}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  </div>`;
}

function _overviewSummaryValueHtml(row) {
  if (!row || !Number.isFinite(Number(row.mean)) || !row.n) {
    return '<span class="summary-metric-empty">—</span>';
  }
  const mean = _fmtVal(Number(row.mean), row.metric_key);
  const standardDeviation = Number.isFinite(Number(row.standard_deviation))
    ? _fmtVal(Number(row.standard_deviation), row.metric_key)
    : "—";
  return `<span class="summary-metric-mean">${escapeHtml(mean)}</span>
    <span class="summary-metric-detail">SD ${escapeHtml(standardDeviation)} · n=${escapeHtml(row.n)}</span>`;
}

function buildMetricSelect() {
  const select = document.getElementById("metric-select");
  select.innerHTML = "";
  for (const [key, label] of Object.entries(METRIC_LABELS)) {
    const option = new Option(label, key, false, key === S.metricKey);
    select.add(option);
  }
  select.addEventListener("change", (event) => {
    S.metricKey = event.target.value;
    renderOverview();
  });
}

function renderOverview() {
  for (const viewer of Object.values(S.overviewViewers)) viewer.destroy();
  S.overviewViewers = {};

  const grid = document.getElementById("overview-grid");
  grid.innerHTML = "";
  const methods = Object.values(S.index.methods);
  const metricScope = _isSupervisedMetric(S.metricKey) ? "supervised" : "unsupervised";

  for (const wsi of S.index.wsis) {
    const card = document.createElement("div");
    card.className = "wsi-card";
    card.dataset.stem = wsi.stem;
    card.classList.toggle("active", S.activeStem === wsi.stem);

    const overviewVisual = wsi.has_wsi
      ? '<div class="card-osd"></div>'
      : '<span class="no-thumb">source WSI unavailable</span>';

    const scoreRows = methods.map((method) => {
      const value = _getMetric(S.index.scores, wsi.stem, method.run_id, S.metricKey);
      if (value === null) return "";
      const visible = S.overviewMethodSet.has(method.run_id);
      return `<div class="score-row" data-run-id="${escapeAttr(method.run_id)}" style="display:${visible ? "flex" : "none"}">
        <span class="score-dot" style="background:${method.color}"></span>
        <span class="score-name" title="${escapeAttr(method.run_id)}">${escapeHtml(method.name)}</span>
        <span class="score-val">${_fmtVal(value, S.metricKey)}</span>
      </div>`;
    }).join("");

    card.innerHTML = `
      <div class="card-thumb">${overviewVisual}</div>
      <div class="card-body">
        <div class="card-heading">
          <div class="card-stem">${escapeHtml(wsi.stem)}</div>
          ${_runStatusBadge(wsi, false)}
        </div>
        <div class="card-scores-wrap" style="display:${S.overviewMethodSet.size ? "block" : "none"}">
          <div class="card-scores-label">${metricScope} metric</div>
          <div class="card-scores">${scoreRows || "<span class='no-scores'>no score for selected metric</span>"}</div>
        </div>
      </div>`;

    card.querySelector(".card-body").addEventListener("click", async (event) => {
      event.stopPropagation();
      await selectStem(wsi.stem);
      await switchView("inspect");
    });
    grid.appendChild(card);
  }
  _setupCardObserver();
}

function _updateCardScoreVisibility() {
  const anyVisible = S.overviewMethodSet.size > 0;
  document.querySelectorAll(".card-scores-wrap").forEach((wrap) => {
    wrap.style.display = anyVisible ? "block" : "none";
  });
  document.querySelectorAll(".score-row[data-run-id]").forEach((row) => {
    row.style.display = S.overviewMethodSet.has(row.dataset.runId) ? "flex" : "none";
  });
}

function _initCardOSD(stem, element) {
  if (S.overviewViewers[stem]) return;
  const viewer = OpenSeadragon({
    element,
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
    tileSources: `/api/wsi/${encodeURIComponent(stem)}/dzi`,
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
    _loadOverviewOverlaysForStem(stem).then(() => _drawCardOverlays(stem));
  });
  viewer.addHandler("animation-finish", () => _drawCardOverlays(stem));
  viewer.addHandler("update-viewport", () => _drawCardOverlays(stem));
  viewer.addHandler("resize", () => _drawCardOverlays(stem));
}

function _getOrCreateCardCanvas(viewer) {
  const osdCanvas = viewer.drawer?.canvas ?? viewer.element.querySelector("canvas");
  if (!osdCanvas) return null;
  let canvas = viewer.element.querySelector(".card-ov-canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.className = "card-ov-canvas";
    canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:10;";
    osdCanvas.parentElement.appendChild(canvas);
  }
  return canvas;
}

function _drawCardOverlays(stem) {
  const viewer = S.overviewViewers[stem];
  if (!viewer) return;
  const canvas = _getOrCreateCardCanvas(viewer);
  if (!canvas) return;
  const { clientWidth: width, clientHeight: height } = viewer.element;
  if (!width || !height) return;
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  if (S.overviewGroundTruthVisible) {
    const groundTruth = S.groundTruthCache[stem];
    if (groundTruth) {
      _drawFeatureCollection(ctx, groundTruth.features, _groundTruthStyle(), groundTruth.scale, viewer);
    }
  }
  for (const runId of S.overviewMethodSet) {
    const entry = S.predictionCache[stem]?.[runId];
    if (!entry) continue;
    _drawFeatureCollection(ctx, entry.features, _predictionStyle(S.index.methods[runId]?.color ?? "#64748b"), entry.scale, viewer);
  }
}

function redrawOverviewCanvases() {
  for (const stem of Object.keys(S.overviewViewers)) _drawCardOverlays(stem);
}

function _setupCardObserver() {
  // Lazily instantiate only overview cards that enter the viewport. This keeps
  // the overview usable without opening hundreds of WSI tile sources at once.
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      const stem = entry.target.dataset.stem;
      const osdElement = entry.target.querySelector(".card-osd");
      if (stem && osdElement) _initCardOSD(stem, osdElement);
    }
  }, { rootMargin: "120px" });
  document.querySelectorAll(".wsi-card[data-stem]").forEach((card) => observer.observe(card));
}

async function _loadOverviewOverlaysForStem(stem) {
  const tasks = [];
  const wsi = _getWsi(stem);
  if (S.overviewGroundTruthVisible && wsi?.has_ground_truth) {
    tasks.push(_fetchGroundTruth(stem));
  }
  for (const runId of S.overviewMethodSet) {
    // Only the official metrics index defines valid slide/method pairs.
    // This prevents the overview from pulling predictions from methods that
    // were not part of the selected 8 July official CSV cohort.
    if (S.index.scores[stem]?.[runId]) tasks.push(_fetchPrediction(stem, runId));
  }
  await Promise.all(tasks);
}

async function _loadVisibleOverviewOverlays() {
  // Fetch polygons only for cards whose OpenSeadragon thumbnails already exist.
  // Toggling a method never loads predictions for all slides.
  await Promise.all(Object.keys(S.overviewViewers).map((stem) => _loadOverviewOverlaysForStem(stem)));
}


// ─────────────────────────────────────────────────────────────────────────────
// OpenSeadragon canvas overlay rendering
// ─────────────────────────────────────────────────────────────────────────────

let redrawPending = false;

function scheduleRedraw() {
  if (redrawPending) return;
  redrawPending = true;
  requestAnimationFrame(() => {
    redrawPending = false;
    drawOverlays();
  });
}

function _initOverlayCanvas() {
  if (!S.viewer) return;
  document.getElementById("osd-overlay-canvas")?.remove();
  const osdCanvas = S.viewer.drawer?.canvas ?? S.viewer.element.querySelector("canvas");
  if (!osdCanvas) return;
  const canvas = document.createElement("canvas");
  canvas.id = "osd-overlay-canvas";
  canvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;z-index:20;";
  osdCanvas.parentElement.appendChild(canvas);
  S.overlayCanvas = canvas;
  _syncCanvas();
}

function _syncCanvas() {
  if (!S.overlayCanvas || !S.viewer) return;
  const { clientWidth: width, clientHeight: height } = S.viewer.element;
  if (!width || !height) return;
  S.overlayCanvas.width = width;
  S.overlayCanvas.height = height;
  S.overlayCanvas.style.width = `${width}px`;
  S.overlayCanvas.style.height = `${height}px`;
}

function drawOverlays() {
  if (!S.overlayCanvas || !S.viewer || !S.activeStem) return;
  const canvas = S.overlayCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const stem = S.activeStem;

  // 1. GT outline (reference)
  if (S.gtVisible) {
    const gt = S.groundTruthCache[stem];
    if (gt) _drawFeatureCollection(ctx, gt.features, _groundTruthStyle(), gt.scale);
  }

  // 2. Segmentation prediction polygons (one color per selected method)
  for (const runId of S.activeMethodSet) {
    const prediction = S.predictionCache[stem]?.[runId];
    if (!prediction) continue;
    _drawFeatureCollection(
      ctx,
      prediction.features,
      _predictionStyle(S.index.methods[runId]?.color ?? "#64748b"),
      prediction.scale,
    );
  }

  // 3. TP/FP/FN geometry for the selected evaluation method, computed lazily by the official server.
  if (S.evaluationRunId) {
    for (const { key } of ERROR_LAYERS) {
      if (!S.evaluationVisible[key]) continue;
      const layer = S.evaluationCache[stem]?.[S.evaluationRunId]?.[key];
      if (layer) _drawFeatureCollection(ctx, layer.features, _errorStyle(key), layer.scale);
    }
  }
}

function _predictionStyle(color) {
  return { stroke: color, fill: color, fillAlpha: 0.09, lineWidth: 1.8, dash: [] };
}

function _groundTruthStyle() {
  return { stroke: COLORS.ground_truth, fill: COLORS.ground_truth, fillAlpha: 0.035, lineWidth: 2.3, dash: [7, 4] };
}

function _errorStyle(layer) {
  return { stroke: COLORS[layer], fill: COLORS[layer], fillAlpha: 0.32, lineWidth: 1.9, dash: [] };
}

function _drawFeatureCollection(ctx, features, style, scale, viewer = S.viewer) {
  if (!features?.length || scale == null || !viewer) return;
  ctx.save();
  ctx.beginPath();
  for (const feature of features) _traceGeometry(ctx, feature?.geometry, scale, viewer);
  if (style.fillAlpha > 0) {
    ctx.fillStyle = style.fill;
    ctx.globalAlpha = style.fillAlpha;
    ctx.fill("evenodd");
  }
  ctx.globalAlpha = 0.96;
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = style.lineWidth;
  ctx.setLineDash(style.dash ?? []);
  ctx.stroke();
  ctx.restore();
}

function _traceGeometry(ctx, geometry, scale, viewer) {
  if (!geometry) return;
  if (geometry.type === "Polygon") {
    for (const ring of geometry.coordinates ?? []) _traceRing(ctx, ring, scale, viewer);
  } else if (geometry.type === "MultiPolygon") {
    for (const polygon of geometry.coordinates ?? []) {
      for (const ring of polygon) _traceRing(ctx, ring, scale, viewer);
    }
  } else if (geometry.type === "GeometryCollection") {
    for (const member of geometry.geometries ?? []) _traceGeometry(ctx, member, scale, viewer);
  }
}

function _traceRing(ctx, ring, scale, viewer) {
  if (!ring || ring.length < 3) return;
  let first = true;
  for (const point of ring) {
    const transformed = viewer.viewport.viewportToViewerElementCoordinates(
      new OpenSeadragon.Point(point[0] * scale, point[1] * scale)
    );
    if (first) {
      ctx.moveTo(transformed.x, transformed.y);
      first = false;
    } else {
      ctx.lineTo(transformed.x, transformed.y);
    }
  }
  ctx.closePath();
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

function buildMetricsTable(stem) {
  const container = document.getElementById("metrics-table");
  const methods = Object.values(S.index.methods).filter((method) => S.index.scores[stem]?.[method.run_id]);
  if (!methods.length) {
    container.innerHTML = '<p class="panel-note">No score files found for this slide.</p>';
    return;
  }

  const supervisedMethods = methods.filter((method) => Boolean(S.index.scores[stem]?.[method.run_id]?.metrics?.supervised));
  const tables = [];
  if (supervisedMethods.length) {
    tables.push(_metricTableHtml("Evaluation", stem, supervisedMethods, EVALUATION_COLS));
  }
  container.innerHTML = tables.join("");
}

function _metricTableHtml(title, stem, methods, columns) {
  const header = columns.map(({ label }) => `<th>${escapeHtml(label)}</th>`).join("");
  const rows = methods.map((method) => {
    const cells = columns.map(({ key }) => {
      const value = _getMetric(S.index.scores, stem, method.run_id, key);
      return `<td><span class="metric-val">${value === null ? "—" : _fmtVal(value, key)}</span></td>`;
    }).join("");
    return `<tr>
      <td><div class="method-cell">
        <span class="method-dot" style="background:${method.color}"></span>
        <span class="method-cell-name" title="${escapeAttr(method.run_id)}">${escapeHtml(method.name)}</span>
      </div></td>${cells}</tr>`;
  }).join("");
  return `<div class="metrics-group">
    <table class="metrics-tbl"><thead><tr><th>Method</th>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// View switching
// ─────────────────────────────────────────────────────────────────────────────

async function switchView(view) {
  S.view = view;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
  document.getElementById("view-overview").style.display = view === "overview" ? "" : "none";
  document.getElementById("view-inspect").style.display = view === "inspect" ? "" : "none";
  document.getElementById("panel-overview").style.display = view === "overview" ? "flex" : "none";
  document.getElementById("panel-inspect").style.display = view === "inspect" ? "flex" : "none";
  if (view === "overview") {
    if (!S.summary) {
      S.summary = await _loadOverviewSummary();
      buildOverviewSummary();
    }
    renderOverview();
  }
  if (view === "inspect" && S.activeStem) await openInspect(S.activeStem);
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric/data helpers
// ─────────────────────────────────────────────────────────────────────────────

function _getMetric(scores, stem, runId, key) {
  const entry = scores[stem]?.[runId];
  if (!entry) return null;
  const source = _isSupervisedMetric(key) ? entry.metrics?.supervised : entry.metrics?.unsupervised;
  const value = source?.[key] ?? entry.metrics?.[key] ?? entry[key] ?? null;
  return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value) : null;
}

function _isSupervisedMetric(key) {
  return SUPERVISED_COLS.some((column) => column.key === key);
}

function _hasAnySupervisedMetric() {
  return Object.values(S.index?.scores ?? {}).some((perSlide) =>
    Object.values(perSlide).some((entry) => Boolean(entry?.metrics?.supervised))
  );
}

function _availableRunIds(stem) {
  return Object.keys(S.index.scores[stem] ?? {}).filter((runId) => Boolean(S.index.methods[runId]));
}

function _supervisedRunIds(stem) {
  return _availableRunIds(stem).filter((runId) => Boolean(S.index.scores[stem]?.[runId]?.metrics?.supervised));
}

function _getWsi(stem) {
  return S.index.wsis.find((wsi) => wsi.stem === stem);
}

function _fmtVal(value, key) {
  if (key === "num_objects") return String(Math.round(value));
  if (key === "mean_area") {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
    return String(Math.round(value));
  }
  if (["over_segmentation_rate", "under_segmentation_rate", "coverage_ratio"].includes(key)) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (key === "hausdorff") return value >= 1000 ? `${value.toFixed(0)}` : value.toFixed(2);
  if (key === "execution_time_s") return `${value.toFixed(2)}s`;
  if (value > 0 && value < 0.01) return value.toExponential(2);
  return value.toFixed(3);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function cssEscape(value) {
  return window.CSS?.escape ? window.CSS.escape(value) : String(value).replace(/["\\]/g, "\\$&");
}

init();

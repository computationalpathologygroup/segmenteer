/**
 * segmenteer decoupled viewer.
 *
 * Predictions are the primary runner artifact. Ground truth is an optional
 * GeoJSON overlay. Metrics are optional evaluator artifacts. The viewer never
 * never computes benchmark metrics. TP/FP/FN are transient visual comparisons
 * computed on demand for only the currently inspected slide + method.
 */

"use strict";

const COLORS = {
  ground_truth: "#2563eb",   // fixed GT blue
  true_positive: "#16a34a",
  false_positive: "#dc2626",
  false_negative: "#f59e0b",
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
const UNSUPERVISED_COLS = [
  { key: "num_objects", label: "Objects" },
  { key: "total_area", label: "Total area" },
  { key: "mean_area", label: "Mean area" },
  { key: "mean_compactness", label: "Compactness" },
  { key: "mean_solidity", label: "Solidity" },
  { key: "coverage_ratio", label: "Coverage" },
];

const METRIC_LABELS = Object.fromEntries(
  [...EVALUATION_COLS, ...UNSUPERVISED_COLS].map(({ key, label }) => [key, label])
);

const S = {
  view: "overview",
  index: null,
  summary: null,
  activeStem: null,
  activeMethodSet: new Set(),
  overviewMethodSet: new Set(),
  overviewGroundTruthVisible: false,
  predictionCache: {},          // stem -> runId -> {features, scale}
  groundTruthCache: {},         // stem -> {features, scale}
  evaluationCache: {},          // transient stem -> runId -> layer -> {features, scale}
  overviewViewers: {},
  viewer: null,
  overlayCanvas: null,
  metricKey: "num_objects",
  overviewSort: "default",
  gtVisible: false,
  evaluationRunId: null,
  evaluationVisible: {
    true_positive: false,
    false_positive: false,
    false_negative: false,
  },
};

async function init() {
  try {
    const res = await fetch("/api/index");
    if (!res.ok) throw new Error(res.statusText);
    S.index = normaliseIndexPayload(await res.json());
    if (S.index.viewer_build !== "20260807_18") {
      throw new Error(`Viewer frontend/backend build mismatch (frontend 20260807_18, backend ${S.index.viewer_build || "unknown"}). Reload the page.`);
    }
  } catch (err) {
    document.body.innerHTML = `<p style="padding:2rem;color:#dc2626">Failed to load viewer index: ${escapeHtml(String(err))}</p>`;
    return;
  }

  const { wsis, methods, output_dir: outputDir } = S.index;
  if (_hasAnySupervisedMetric()) S.metricKey = "dice";

  document.getElementById("run-label").textContent = outputDir.split("/").at(-1);
  document.getElementById("slide-count").textContent = String(wsis.length);
  // Keep first paint cheap: do not open WSI/GeoJSON until requested.
  S.summary = null;
  configureOptionalUI();
  buildSidebar();
  if (S.index.metrics_available) buildMetricSelect();
  buildOverviewToggles();
  if (S.index.metrics_available) buildDiceSortControl();
  if (S.index.metrics_available) buildOverviewSummary();
  await switchView("overview");

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  document.getElementById("search").addEventListener("input", (event) => {
    const query = event.target.value.toLowerCase();
    document.querySelectorAll("#wsi-list li").forEach((item) => {
      item.classList.toggle("hidden", !item.dataset.stem.toLowerCase().includes(query));
    });
  });
  document.getElementById("download-overview-html").addEventListener("click", downloadOverviewHtml);
}

function normaliseIndexPayload(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const methods = {};
  const entries = Array.isArray(source.methods)
    ? source.methods.map((method, index) => [method?.run_id ?? method?.id ?? String(index), method])
    : Object.entries(source.methods ?? {});

  for (const [key, candidate] of entries) {
    const method = candidate && typeof candidate === "object" ? candidate : {};
    const runId = String(method.run_id ?? method.id ?? key ?? "").trim();
    if (!runId) continue;
    const name = String(
      method.name ?? method.display_name ?? method.formal_name ?? method.label ?? runId
    ).trim() || runId;
    const label = String(
      method.label ?? method.display_name ?? method.formal_name ?? name
    ).trim() || name;
    methods[runId] = {
      ...method,
      run_id: runId,
      name,
      label,
      details: String(method.details ?? "").trim(),
      color: String(method.color ?? "#64748b"),
    };
  }

  return {
    ...source,
    wsis: Array.isArray(source.wsis) ? source.wsis : [],
    methods,
    scores: source.scores && typeof source.scores === "object" ? source.scores : {},
    available_metrics: Array.isArray(source.available_metrics) ? source.available_metrics : [],
    metrics_available: Boolean(source.metrics_available),
    ground_truth_available: Boolean(source.ground_truth_available),
    viewer_build: String(source.viewer_build ?? ""),
  };
}

function configureOptionalUI() {
  const hasMetrics = Boolean(S.index.metrics_available);
  const hasGroundTruth = Boolean(S.index.ground_truth_available);
  document.getElementById("metrics-overview-controls").hidden = !hasMetrics;
  document.getElementById("metrics-overview-panel").hidden = !hasMetrics;
  document.getElementById("metrics-panel").hidden = !hasMetrics;
  document.getElementById("spatial-evaluation-panel").hidden = !hasGroundTruth;
  document.getElementById("overview-ground-truth-control").hidden = !hasGroundTruth;
  document.getElementById("ground-truth-panel").hidden = !hasGroundTruth;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar and selection
// ─────────────────────────────────────────────────────────────────────────────

function buildSidebar() {
  const list = document.getElementById("wsi-list");
  list.innerHTML = "";
  for (const wsi of _orderedWsis()) {
    const item = document.createElement("li");
    item.dataset.stem = wsi.stem;
    item.classList.toggle("active", S.activeStem === wsi.stem);
    item.innerHTML = `
      <span class="wsi-dot ${wsi.has_wsi ? "has-wsi" : ""}"></span>
      <span class="wsi-stem" title="${escapeAttr(wsi.stem)}">${escapeHtml(wsi.stem)}</span>
      ${wsi.has_ground_truth ? '<span class="gt-badge" title="Ground truth available">GT</span>' : ""}
      ${_runStatusBadge(wsi, true)}`;
    item.addEventListener("click", () => selectStem(wsi.stem));
    list.appendChild(item);
  }
}

function _runStatusBadge(wsi, compact) {
  if (!S.index.metrics_available) return "";
  const perSlide = S.index.scores?.[wsi.stem] ?? {};
  const hasSupervised = Object.values(perSlide).some((entry) => Boolean(entry?.metrics?.supervised && Object.keys(entry.metrics.supervised).length));
  const hasUnsupervised = Object.values(perSlide).some((entry) => Boolean(entry?.metrics?.unsupervised && Object.keys(entry.metrics.unsupervised).length));
  const status = hasSupervised ? "supervised" : hasUnsupervised ? "unsupervised" : null;
  const info = status === "supervised"
    ? { label: compact ? "EVAL" : "evaluated", title: "Supervised evaluator metrics are loaded" }
    : status === "unsupervised"
      ? { label: compact ? "UNSUP" : "reference-free", title: "Reference-free evaluator metrics are loaded" }
      : null;
  if (!info) return "";
  return `<span class="run-status status-${escapeAttr(status)}" title="${escapeAttr(info.title)}">${escapeHtml(info.label)}</span>`;
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
  await switchView("inspect");
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

  // Opening a slide does not fetch polygons until a layer is selected.
  S.activeMethodSet = new Set();
  S.gtVisible = false;
  S.evaluationRunId = null;
  S.evaluationVisible = { true_positive: false, false_positive: false, false_negative: false };

  buildMethodToggles(stem);
  if (S.index.ground_truth_available) buildGroundTruthControl(stem);
  if (S.index.ground_truth_available) buildSpatialEvaluationControls(stem);
  if (S.index.metrics_available) buildMetricsTable(stem);

  _refreshMethodToggles();
  if (S.index.ground_truth_available) _refreshGroundTruthControl();

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
// Prediction and optional GT controls
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
    chip.title = `${method.label} — ${method.details || method.run_id}`;
    chip.style.setProperty("--method-color", method.color);

    chip.innerHTML = `
      <span class="chip-swatch" style="background:${method.color}"></span>
      <span class="chip-name">${escapeHtml(method.label || method.name)}</span>
      <span class="chip-params">${escapeHtml(method.details || "—")}</span>
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

function buildGroundTruthControl(stem) {
  const wsi = _getWsi(stem);
  const container = document.getElementById("ground-truth-toggle");
  container.innerHTML = "";

  const button = document.createElement("button");
  button.type = "button";
  button.id = "gt-layer-button";
  button.className = "layer-chip";
  button.style.color = COLORS.ground_truth;
  button.innerHTML = `<span class="chip-swatch" style="background:${COLORS.ground_truth}"></span>Ground truth <span class="chip-eye">○</span>`;
  button.disabled = !Boolean(wsi?.has_ground_truth);
  if (button.disabled) button.title = "No ground-truth GeoJSON is available for this slide";
  button.addEventListener("click", () => toggleGroundTruth(stem));
  container.appendChild(button);
  _refreshGroundTruthControl();
}

async function toggleGroundTruth(stem) {
  if (!(_getWsi(stem)?.has_ground_truth)) return;
  if (S.gtVisible) {
    S.gtVisible = false;
  } else {
    const loaded = await _fetchGroundTruth(stem);
    if (loaded) S.gtVisible = true;
  }
  _refreshGroundTruthControl();
  drawOverlays();
}

function _refreshGroundTruthControl() {
  const button = document.getElementById("gt-layer-button");
  if (!button) return;
  button.classList.toggle("active", S.gtVisible);
  const eye = button.querySelector(".chip-eye");
  if (eye && !button.disabled) eye.textContent = S.gtVisible ? "●" : "○";
}

async function _fetchGroundTruth(stem) {
  if (S.groundTruthCache[stem]) return true;
  const entry = await _fetchGeojson(`/api/ground-truth/${encodeURIComponent(stem)}`);
  if (!entry) return false;
  S.groundTruthCache[stem] = entry;
  return true;
}

async function _fetchPrediction(stem, runId) {
  if (S.predictionCache[stem]?.[runId]) return true;
  const entry = await _fetchGeojson(
    `/api/overlay/${encodeURIComponent(stem)}/${encodeURIComponent(runId)}`
  );
  if (!entry) return false;
  if (!S.predictionCache[stem]) S.predictionCache[stem] = {};
  S.predictionCache[stem][runId] = entry;
  return true;
}

async function _fetchGeojson(url) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      const message = await response.text().catch(() => "");
      throw new Error(`${response.status} ${response.statusText}${message ? `: ${message}` : ""}`);
    }
    const payload = await response.json();
    const features = payload?.type === "FeatureCollection" && Array.isArray(payload.features)
      ? payload.features
      : payload?.type === "Feature"
        ? [payload]
        : [];
    const rawScale = Number(payload?._meta?.scale_to_viewport);
    const scale = Number.isFinite(rawScale) && rawScale > 0 ? rawScale : null;
    if (scale === null) {
      console.warn("Overlay has no valid level-0-to-viewport scale", url, payload?._meta);
    }
    return { features, scale };
  } catch (error) {
    console.error("Could not load GeoJSON overlay", url, error);
    return null;
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Transient TP / FP / FN spatial comparison for the inspected pair
// ─────────────────────────────────────────────────────────────────────────────

function _spatialComparisonRunIds(stem) {
  const wsi = _getWsi(stem);
  if (!wsi?.has_ground_truth) return [];
  return Object.values(S.index.methods)
    .filter((method) => Boolean(S.index.scores?.[stem]?.[method.run_id]?._prediction_available))
    .map((method) => method.run_id);
}

function buildSpatialEvaluationControls(stem) {
  const panel = document.getElementById("spatial-evaluation-panel");
  const select = document.getElementById("evaluation-method-select");
  const toggles = document.getElementById("evaluation-toggles");
  const status = document.getElementById("evaluation-status");
  if (!panel || !select || !toggles || !status) return;

  const runIds = _spatialComparisonRunIds(stem);
  panel.hidden = !runIds.length;
  select.innerHTML = "";
  toggles.innerHTML = "";
  status.textContent = "";
  if (!runIds.length) return;

  for (const runId of runIds) {
    const method = S.index.methods[runId];
    select.append(new Option(method?.label ?? runId, runId));
  }
  S.evaluationRunId = runIds[0];
  select.value = S.evaluationRunId;
  select.onchange = () => {
    S.evaluationRunId = select.value || null;
    S.evaluationVisible = { true_positive: false, false_positive: false, false_negative: false };
    _refreshSpatialEvaluationControls(stem);
    drawOverlays();
  };

  for (const descriptor of ERROR_LAYERS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "layer-chip";
    button.dataset.layer = descriptor.key;
    button.style.color = COLORS[descriptor.key];
    button.innerHTML = `<span class="chip-swatch" style="background:${COLORS[descriptor.key]}"></span>${descriptor.label} <span class="chip-eye">○</span>`;
    button.addEventListener("click", () => toggleEvaluationLayer(stem, descriptor.key));
    toggles.appendChild(button);
  }
  _refreshSpatialEvaluationControls(stem);
}

async function toggleEvaluationLayer(stem, layer) {
  if (!S.evaluationRunId || !_getWsi(stem)?.has_ground_truth) return;
  if (S.evaluationVisible[layer]) {
    S.evaluationVisible[layer] = false;
  } else {
    const loaded = await _fetchEvaluationLayer(stem, S.evaluationRunId, layer);
    if (loaded) S.evaluationVisible[layer] = true;
  }
  _refreshSpatialEvaluationControls(stem);
  drawOverlays();
}

function _refreshSpatialEvaluationControls(stem) {
  const available = Boolean(S.evaluationRunId && _getWsi(stem)?.has_ground_truth);
  document.querySelectorAll("#evaluation-toggles .layer-chip").forEach((button) => {
    const active = available && Boolean(S.evaluationVisible[button.dataset.layer]);
    button.disabled = !available;
    button.classList.toggle("active", active);
    const eye = button.querySelector(".chip-eye");
    if (eye) eye.textContent = available ? (active ? "●" : "○") : "–";
  });
  const status = document.getElementById("evaluation-status");
  if (status) status.textContent = "Computed on demand from the selected prediction and ground truth; cached only in viewer memory.";
}

async function _fetchEvaluationLayer(stem, runId, layer) {
  if (S.evaluationCache[stem]?.[runId]?.[layer]) return true;
  const entry = await _fetchGeojson(`/api/spatial-comparison/${encodeURIComponent(stem)}/${encodeURIComponent(runId)}/${encodeURIComponent(layer)}`);
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
  groundTruthButton.title = "Show or hide ground-truth contours in overview cards";
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
    button.title = method.details || method.run_id;
    button.innerHTML = `<span class="chip-swatch" style="background:${method.color}"></span>${escapeHtml(method.label)}`;
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
    _syncDiceSortControl();
    if (S.overviewSort !== "default") {
      buildSidebar();
      renderOverview();
    } else {
      _updateCardScoreVisibility();
      redrawOverviewCanvases();
    }
    return;
  }
  S.overviewMethodSet.add(runId);
  button.classList.add("active");
  _syncDiceSortControl();
  if (S.overviewSort !== "default") {
    buildSidebar();
    renderOverview();
  } else {
    _updateCardScoreVisibility();
  }
  // Do not fetch prediction GeoJSON for every slide. Load only overlays for
  // currently instantiated overview cards; inspect-view layers are fetched on click.
  await _loadVisibleOverviewOverlays();
  redrawOverviewCanvases();
}

function buildDiceSortControl() {
  const control = document.getElementById("dice-sort-control");
  const select = document.getElementById("dice-sort-select");
  if (!control || !select) return;

  const diceAvailable = (S.index.available_metrics ?? []).includes("dice") && _hasAnySupervisedMetric();
  control.hidden = !diceAvailable;
  if (!diceAvailable) {
    S.overviewSort = "default";
    return;
  }

  select.value = S.overviewSort;
  select.addEventListener("change", (event) => {
    S.overviewSort = event.target.value;
    buildSidebar();
    renderOverview();
    _syncDiceSortControl();
  });
  _syncDiceSortControl();
}

function _syncDiceSortControl() {
  const control = document.getElementById("dice-sort-control");
  const select = document.getElementById("dice-sort-select");
  const context = document.getElementById("dice-sort-context");
  if (!control || !select || control.hidden) return;

  const selected = [...S.overviewMethodSet];
  if (!selected.length) {
    if (S.overviewSort !== "default") {
      S.overviewSort = "default";
      select.value = "default";
      buildSidebar();
      if (S.view === "overview") renderOverview();
    }
    select.disabled = true;
    if (context) context.textContent = "Select prediction method(s)";
    return;
  }

  select.disabled = false;
  if (context) {
    if (selected.length === 1) {
      context.textContent = `Dice: ${S.index.methods[selected[0]]?.label ?? selected[0]}`;
    } else {
      context.textContent = `Mean Dice across ${selected.length} selected methods`;
    }
  }
}

function _diceSortScore(stem) {
  const selected = [...S.overviewMethodSet];
  if (!selected.length) return null;
  const values = selected.map((runId) => _getMetric(S.index.scores, stem, runId, "dice"));
  // A multi-method mean represents all selected methods. If any selected
  // method lacks Dice for this slide, the sort score is missing and the slide
  // is placed after slides with a complete score.
  if (values.some((value) => value === null)) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function _orderedWsis() {
  const rows = [...(S.index?.wsis ?? [])];
  if (S.overviewSort !== "dice-asc" && S.overviewSort !== "dice-desc") return rows;
  if (!S.overviewMethodSet.size) return rows;

  const direction = S.overviewSort === "dice-asc" ? 1 : -1;
  return rows.sort((left, right) => {
    const leftScore = _diceSortScore(left.stem);
    const rightScore = _diceSortScore(right.stem);
    const leftMissing = leftScore === null;
    const rightMissing = rightScore === null;
    if (leftMissing && rightMissing) return left.stem.localeCompare(right.stem, undefined, { sensitivity: "base" });
    if (leftMissing) return 1;
    if (rightMissing) return -1;
    if (leftScore !== rightScore) return direction * (leftScore - rightScore);
    return left.stem.localeCompare(right.stem, undefined, { sensitivity: "base" });
  });
}

async function _loadOverviewSummary() {
  if (!S.index.metrics_available) return null;
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
    return `<tr><th scope="row" title="${escapeAttr(method.run_id)}"><span class="summary-method-dot" style="background:${method.color}"></span>${escapeHtml(method.label)}</th>${cells}</tr>`;
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
  const available = (S.index.available_metrics ?? []).filter((key) => METRIC_LABELS[key]);
  if (!available.length) return;
  if (!available.includes(S.metricKey)) S.metricKey = available[0];
  for (const key of available) {
    select.add(new Option(METRIC_LABELS[key], key, false, key === S.metricKey));
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
  const hasMetrics = Boolean(S.index.metrics_available);
  const metricScope = _isSupervisedMetric(S.metricKey) ? "supervised" : "reference-free";

  for (const wsi of _orderedWsis()) {
    const card = document.createElement("div");
    card.className = "wsi-card";
    card.dataset.stem = wsi.stem;
    card.classList.toggle("active", S.activeStem === wsi.stem);

    const overviewVisual = wsi.has_wsi
      ? '<div class="card-osd"></div>'
      : '<span class="no-thumb">source WSI unavailable</span>';

    const scoreRows = hasMetrics ? methods.map((method) => {
      const value = _getMetric(S.index.scores, wsi.stem, method.run_id, S.metricKey);
      if (value === null) return "";
      const visible = S.overviewMethodSet.has(method.run_id);
      return `<div class="score-row" data-run-id="${escapeAttr(method.run_id)}" style="display:${visible ? "flex" : "none"}">
        <span class="score-dot" style="background:${method.color}"></span>
        <span class="score-name" title="${escapeAttr(method.run_id)}">${escapeHtml(method.label)}</span>
        <span class="score-val">${_fmtVal(value, S.metricKey)}</span>
      </div>`;
    }).join("") : "";

    card.innerHTML = `
      <div class="card-thumb">${overviewVisual}</div>
      <div class="card-body">
        <div class="card-heading">
          <div class="card-stem">${escapeHtml(wsi.stem)}</div>
          ${_runStatusBadge(wsi, false)}
        </div>
        ${hasMetrics ? `<div class="card-scores-wrap" style="display:${S.overviewMethodSet.size ? "block" : "none"}">
          <div class="card-scores-label">${metricScope} metric</div>
          <div class="card-scores">${scoreRows || "<span class='no-scores'>no score for selected metric</span>"}</div>
        </div>` : ""}
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
  if (!S.index.metrics_available) return;
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
    // Prediction availability comes from runner artifacts, independently of metrics.
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
// Self-contained Overview HTML report
// ─────────────────────────────────────────────────────────────────────────────

async function downloadOverviewHtml() {
  const button = document.getElementById("download-overview-html");
  const status = document.getElementById("overview-report-status");
  const wsis = _orderedWsis();
  const runIds = [...S.overviewMethodSet];
  const includeGroundTruth = Boolean(S.overviewGroundTruthVisible && S.index.ground_truth_available);

  if (!runIds.length && !includeGroundTruth) {
    status.textContent = "Select at least one prediction overlay or Ground truth first.";
    return;
  }

  button.disabled = true;
  const oldText = button.textContent;
  button.textContent = "Building report…";
  status.textContent = `Preparing 0/${wsis.length} slides…`;

  try {
    const cards = [];
    for (let index = 0; index < wsis.length; index += 1) {
      const wsi = wsis[index];
      status.textContent = `Preparing ${index + 1}/${wsis.length}: ${wsi.stem}`;
      const card = await _buildOverviewReportCard(wsi, runIds, includeGroundTruth);
      cards.push(card);
    }
    const html = _overviewReportHtml(cards, runIds, includeGroundTruth);
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const experiment = (S.index.output_dir || "segmenteer").split("/").filter(Boolean).at(-1) || "segmenteer";
    anchor.href = url;
    anchor.download = `segmenteer_overview_${_safeFilename(experiment)}.html`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    status.textContent = `Downloaded ${cards.length}-slide HTML report.`;
  } catch (error) {
    console.error("Could not build overview report", error);
    status.textContent = `Report failed: ${String(error?.message || error)}`;
  } finally {
    button.disabled = false;
    button.textContent = oldText;
  }
}

async function _buildOverviewReportCard(wsi, runIds, includeGroundTruth) {
  let width = 1;
  let height = 1;
  let thumbnail = null;
  if (wsi.has_wsi) {
    const [infoResponse, thumbResponse] = await Promise.all([
      fetch(`/api/wsi/${encodeURIComponent(wsi.stem)}/info`, { cache: "no-store" }),
      fetch(`/api/wsi/${encodeURIComponent(wsi.stem)}/thumbnail.jpeg?max_size=900`, { cache: "no-store" }),
    ]);
    if (infoResponse.ok) {
      const info = await infoResponse.json();
      width = Math.max(1, Number(info.width) || 1);
      height = Math.max(1, Number(info.height) || 1);
    }
    if (thumbResponse.ok) thumbnail = await _blobToDataUrl(await thumbResponse.blob());
  }

  const layers = [];
  if (includeGroundTruth && wsi.has_ground_truth) {
    await _fetchGroundTruth(wsi.stem);
    const entry = S.groundTruthCache[wsi.stem];
    if (entry) layers.push({ label: "Ground truth", color: COLORS.ground_truth, dashed: true, features: entry.features });
  }
  for (const runId of runIds) {
    if (!S.index.scores?.[wsi.stem]?.[runId]?._prediction_available) continue;
    await _fetchPrediction(wsi.stem, runId);
    const entry = S.predictionCache[wsi.stem]?.[runId];
    if (!entry) continue;
    const method = S.index.methods[runId];
    layers.push({ label: method?.label || runId, color: method?.color || "#64748b", dashed: false, features: entry.features });
  }
  return { stem: wsi.stem, width, height, thumbnail, layers };
}

function _overviewReportHtml(cards, runIds, includeGroundTruth) {
  const experiment = (S.index.output_dir || "segmenteer").split("/").filter(Boolean).at(-1) || "segmenteer";
  const legend = [];
  if (includeGroundTruth) legend.push(`<span class="legend-item"><i style="background:${COLORS.ground_truth}"></i>Ground truth</span>`);
  for (const runId of runIds) {
    const method = S.index.methods[runId];
    if (!method) continue;
    legend.push(`<span class="legend-item"><i style="background:${escapeAttr(method.color)}"></i>${escapeHtml(method.label)}</span>`);
  }
  const cardHtml = cards.map(_overviewReportCardHtml).join("\n");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>segmenteer overview — ${escapeHtml(experiment)}</title>
<style>
  *{box-sizing:border-box} body{margin:0;background:#f5f8fc;color:#24324a;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
  header{position:sticky;top:0;z-index:2;background:rgba(255,255,255,.96);border-bottom:1px solid #dbe4ef;padding:16px 22px}
  h1{font-size:18px;margin:0 0 9px;font-weight:600} .meta{font-size:12px;color:#71839e;margin-bottom:10px}
  .legend{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:12px}.legend-item{display:inline-flex;align-items:center;gap:6px}.legend-item i{width:10px;height:10px;border-radius:50%;display:inline-block}
  main{padding:18px;display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
  .card{background:#fff;border:1px solid #dbe4ef;border-radius:10px;overflow:hidden;break-inside:avoid}.visual{background:#edf2f8;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center}
  .visual svg{width:100%;height:100%;display:block}.label{padding:10px 12px;font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .missing{color:#8da0ba;font-size:12px} @media print{header{position:static}body{background:white}main{padding:10px;grid-template-columns:repeat(3,1fr);gap:10px}.card{box-shadow:none}}
</style>
</head>
<body>
<header><h1>segmenteer overview · ${escapeHtml(experiment)}</h1><div class="meta">${cards.length} slides · self-contained thumbnail + outline export</div><div class="legend">${legend.join("")}</div></header>
<main>${cardHtml}</main>
</body></html>`;
}

function _overviewReportCardHtml(card) {
  const paths = card.layers.map((layer) => _geojsonSvgPaths(layer.features, layer.color, layer.dashed)).join("");
  const visual = card.thumbnail
    ? `<svg viewBox="0 0 ${card.width} ${card.height}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${escapeAttr(card.stem)} thumbnail with segmentation outlines">
         <image href="${card.thumbnail}" x="0" y="0" width="${card.width}" height="${card.height}" preserveAspectRatio="none"/>
         ${paths}
       </svg>`
    : `<div class="missing">source WSI unavailable</div>`;
  return `<article class="card"><div class="visual">${visual}</div><div class="label" title="${escapeAttr(card.stem)}">${escapeHtml(card.stem)}</div></article>`;
}

function _geojsonSvgPaths(features, color, dashed) {
  const d = [];
  for (const feature of features || []) _geometryToSvgPath(feature?.geometry, d);
  if (!d.length) return "";
  // Match the interactive Overview canvas styling in screen pixels.
  // vector-effect keeps these widths constant as the self-contained report card scales.
  const strokeWidth = dashed ? 2.3 : 1.8;
  const dash = dashed ? ' stroke-dasharray="7 4"' : "";
  return `<path d="${d.join(" ")}" fill="none" stroke="${escapeAttr(color)}" stroke-width="${strokeWidth}" vector-effect="non-scaling-stroke" stroke-linejoin="round" stroke-linecap="round"${dash}/>`;
}

function _geometryToSvgPath(geometry, output) {
  if (!geometry) return;
  if (geometry.type === "Polygon") {
    for (const ring of geometry.coordinates || []) _ringToSvgPath(ring, output);
  } else if (geometry.type === "MultiPolygon") {
    for (const polygon of geometry.coordinates || []) for (const ring of polygon || []) _ringToSvgPath(ring, output);
  } else if (geometry.type === "GeometryCollection") {
    for (const member of geometry.geometries || []) _geometryToSvgPath(member, output);
  }
}

function _ringToSvgPath(ring, output) {
  if (!Array.isArray(ring) || ring.length < 3) return;
  const points = ring.filter((point) => Array.isArray(point) && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])));
  if (points.length < 3) return;
  output.push(`M ${points.map((point) => `${Number(point[0]).toFixed(2)} ${Number(point[1]).toFixed(2)}`).join(" L ")} Z`);
}

function _blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error || new Error("Could not encode thumbnail"));
    reader.readAsDataURL(blob);
  });
}

function _safeFilename(value) {
  return String(value || "segmenteer").replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "") || "segmenteer";
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

  // 3. Transient TP / FP / FN geometry for the selected prediction + GT pair.
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
  return { stroke: COLORS[layer], fill: COLORS[layer], fillAlpha: 0.30, lineWidth: 1.9, dash: [] };
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
  if (!S.index.metrics_available) return;
  const container = document.getElementById("metrics-table");
  const methods = Object.values(S.index.methods).filter((method) => S.index.scores[stem]?.[method.run_id]);
  if (!methods.length) {
    container.innerHTML = '<p class="panel-note">No evaluator metrics are present for this slide.</p>';
    return;
  }

  const supervisedMethods = methods.filter((method) => Boolean(S.index.scores[stem]?.[method.run_id]?.metrics?.supervised));
  const unsupervisedMethods = methods.filter((method) => Boolean(S.index.scores[stem]?.[method.run_id]?.metrics?.unsupervised));
  const tables = [];
  if (supervisedMethods.length) {
    tables.push(_metricTableHtml("Evaluation", stem, supervisedMethods, EVALUATION_COLS));
  }
  if (unsupervisedMethods.length) {
    tables.push(_metricTableHtml("Reference-free metrics", stem, unsupervisedMethods, UNSUPERVISED_COLS));
  }
  container.innerHTML = tables.join("") || '<p class="panel-note">No evaluator metrics loaded for this slide.</p>';
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
        <span class="method-cell-name" title="${escapeAttr(method.run_id)}">${escapeHtml(method.label)}</span>
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
  const hasMetrics = Boolean(S.index.metrics_available);
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
  document.getElementById("view-overview").style.display = view === "overview" ? "" : "none";
  document.getElementById("view-inspect").style.display = view === "inspect" ? "" : "none";

  const layout = document.getElementById("layout");
  const panel = document.getElementById("panel");
  const overviewWithoutMetrics = view === "overview" && !hasMetrics;
  layout.classList.toggle("overview-without-metrics", overviewWithoutMetrics);
  panel.style.display = overviewWithoutMetrics ? "none" : "flex";
  document.getElementById("panel-overview").style.display = view === "overview" && hasMetrics ? "flex" : "none";
  document.getElementById("panel-inspect").style.display = view === "inspect" ? "flex" : "none";

  if (view === "overview") {
    if (hasMetrics && !S.summary) {
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

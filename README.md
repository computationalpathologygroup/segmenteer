<img src="docs/title.svg" alt="segmenteer" height="42">

A repository for tissue segmentation on whole-slide images (WSI).

**Current package version: `0.1.0`.**

## Architecture

Segmenteer separates segmentation, evaluation, and visualization into three independent components:

```text
RUNNER
  WSI files
     ↓
  prediction GeoJSON + config.yaml

EVALUATOR
  runner predictions
  + optional ground-truth GeoJSON
     ↓
  metrics.csv or metrics.sqlite3

VIEWER
  runner output
  + optional WSI directory
  + optional ground-truth GeoJSON
  + optional evaluator metrics
     ↓
  interactive browser + overview HTML export
```

### Runner

The runner discovers the configured WSI dataset and executes the selected segmentation methods.

For each method it writes:

- one prediction GeoJSON per slide;
- the method configuration in `config.yaml`.

At experiment level it writes `dataset_manifest.json` with the number and names of slides in the cohort.

### Evaluator

The evaluator reads prediction GeoJSON files from an existing runner experiment.

- With matching ground-truth GeoJSON files, it computes supervised segmentation metrics.
- Without matching ground truth, it computes reference-free structural metrics.
- Results can be written as CSV or SQLite.

### Viewer

The viewer opens an existing runner experiment and can optionally add:

- source WSI files for thumbnails and interactive slide browsing;
- ground-truth GeoJSON overlays;
- evaluator metrics from CSV or SQLite.

The interface adapts to the supplied inputs. Evaluation summaries, EVAL indicators, and Dice sorting are enabled when usable evaluator metrics are loaded.

---

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) is recommended

### Quick start: install everything

```bash
git clone <repo-url>
cd segmenteer
uv sync --extra all
```

The `all` extra installs the supported segmentation backends plus viewer/runtime dependencies.

You can run the examples below with `python` after activating the environment, or prefix them with `uv run`, for example:

```bash
uv run python run.py
```

### Install selected functionality

```bash
# WSI/runtime support
uv sync --extra wsi
uv sync --extra openslide
uv sync --extra app
uv sync --extra runtime

# Classical methods
uv sync --extra background-subtractor
uv sync --extra od-gmm
uv sync --extra entropymasker
uv sync --extra histomicstk

# Deep-learning methods/backends
uv sync --extra torch
uv sync --extra atlaspatch-sam2
uv sync --extra bigpicture
uv sync --extra cpg
uv sync --extra fastsam
uv sync --extra grandqc
uv sync --extra hest
uv sync --extra pathprofiler
uv sync --extra rtlucassen
uv sync --extra trident

# All supported deep-learning adapters
uv sync --extra dl
```

For the interactive viewer with WSI tile support:

```bash
uv sync --extra runtime
```

Use `uv sync --extra <name>` so versions and pinned Git dependencies declared in `pyproject.toml` are resolved consistently.

---

## 1. Run segmentation

Edit `run.py` and configure:

- `WSI_DIR` — directory containing WSI files;
- `METHODS` — methods/configurations to run;
- optional runtime settings such as device, slide order, and worker count.

Run with a new timestamped experiment directory:

```bash
python run.py
```

Or choose/reuse an experiment directory explicitly:

```bash
python run.py --output-dir outputs/<experiment>
```

Useful runner options:

```bash
python run.py --help

python run.py --new
python run.py --slide-order alphabetical
python run.py --slide-order smallest-first
python run.py --workers 4
```

When an existing Segmenteer 0.1.0 experiment directory is reused, `dataset_manifest.json` restores its saved slide-name cohort and completed prediction artifacts can be reused.

`run.py` is intentionally a thin experiment launcher. Slide discovery, manifest handling,
resume, worker selection, and sharding are implemented by `segmenteer.runner`.

### Runner output

```text
outputs/<experiment>/
├── dataset_manifest.json
├── <method-run-id>/
│   ├── config.yaml
│   └── predictions/
│       ├── slide_001.geojson
│       ├── slide_002.geojson
│       └── ...
└── ...
```

A dataset manifest contains the experiment slide count and names:

```json
{
  "n_slides": 2,
  "slides": [
    "slide_001.svs",
    "slide_002.svs"
  ]
}
```

### Method configuration

Each method directory contains the configuration used to create its predictions. Downstream tools use this saved configuration to identify the method and its parameters.

For wrapped methods such as TRIDENT, the underlying saved model metadata is used in the UI. For example:

```yaml
class: segmenteer.core.base.TRIDENTSegmenter
params:
  model_id: PathProfiler
  target_mag: 4
```

is shown as a PathProfiler/TRIDENT method with target magnification `4×`.

Resolution metadata is displayed directly from saved configuration:

- explicit `mpp` values are shown as µm/px;
- `target_mag` values are shown as magnification.

---

## 2. Evaluate predictions

The evaluator discovers prediction files under:

```text
outputs/<experiment>/<method-run-id>/predictions/*.geojson
```

### Supervised evaluation with GeoJSON ground truth

Ground-truth files are matched to predictions by filename/stem:

```text
ground_truth_geojson/
├── slide_001.geojson
├── slide_002.geojson
└── ...
```

#### CSV

```bash
python -m segmenteer.evaluator \
  --output outputs/<experiment> \
  --ground-truth ground_truth_geojson \
  --format csv \
  --metrics outputs/<experiment>/metrics.csv
```

#### SQLite

```bash
python -m segmenteer.evaluator \
  --output outputs/<experiment> \
  --ground-truth ground_truth_geojson \
  --format sqlite \
  --metrics outputs/<experiment>/metrics.sqlite3
```

If `--metrics` is omitted, the evaluator writes to the experiment directory using the selected format:

```text
outputs/<experiment>/metrics.csv
```

or:

```text
outputs/<experiment>/metrics.sqlite3
```

### Reference-free evaluation

Ground truth is optional:

```bash
python -m segmenteer.evaluator \
  --output outputs/<experiment> \
  --format sqlite
```

Predictions without a matching ground-truth GeoJSON receive reference-free structural metrics.

### Supervised metrics

The supervised evaluator reports:

- Dice;
- IoU;
- precision;
- recall;
- over-segmentation rate;
- under-segmentation rate.

### SQLite resume

SQLite provides resumable evaluation. Unchanged `(slide, method)` inputs can reuse existing rows based on saved file signatures.

CSV is written as a complete snapshot of the current evaluation run.

---

## 3. View an experiment

Launch the viewer from the repository root:

```bash
python -m app --output outputs/<experiment>
```

The viewer opens at:

```text
http://127.0.0.1:8765
```

Use `--port <PORT>` or `--host <HOST>` when needed.

### Predictions

```bash
python -m app \
  --output outputs/<experiment>
```

### Predictions + source WSI

```bash
python -m app \
  --output outputs/<experiment> \
  --data /path/to/wsi/files
```

`--data` enables source WSI thumbnails and interactive WSI tile browsing.

### Predictions + ground truth

```bash
python -m app \
  --output outputs/<experiment> \
  --ground-truth ground_truth_geojson \
  --data /path/to/wsi/files
```

Ground truth appears as a dedicated overlay in the viewer.

### Predictions + metrics

```bash
python -m app \
  --output outputs/<experiment> \
  --metrics outputs/<experiment>/metrics.csv \
  --data /path/to/wsi/files
```

`--metrics` accepts evaluator CSV or SQLite output. In metrics mode, method controls are scoped to runner methods represented by the loaded metric rows.

### Full viewer

```bash
python -m app \
  --output outputs/<experiment> \
  --ground-truth ground_truth_geojson \
  --metrics outputs/<experiment>/metrics.csv \
  --data /path/to/wsi/files
```

Only `--output` is required. The other inputs can be combined as needed.

---

## Viewer

### Overview

The viewer starts on the **Overview** page.

Overview includes:

- the number of slides in the experiment;
- WSI thumbnails when source slides are available;
- prediction overlay toggles;
- optional ground-truth overlay;
- a distinct color for every prediction method;
- a fixed blue dashed ground-truth outline;
- evaluation score cards and summary when metrics are loaded;
- Dice-based slide sorting when supervised Dice metrics are available;
- self-contained HTML export of the current visual overview.

Dice sorting supports:

- **Default order**;
- **Dice ↑**;
- **Dice ↓**.

With one prediction method selected, sorting uses that method's Dice. With multiple methods selected, sorting uses their mean Dice for each slide.

### Inspect

Clicking any slide in the left sidebar opens **Inspect** for that slide.

Inspect can show:

- the source WSI;
- selected prediction outlines;
- ground-truth outline;
- evaluator metrics;
- transient TP/FP/FN spatial comparison layers for the currently selected prediction and ground truth.

The TP/FP/FN visualization is generated on demand from the selected geometries and cached in memory for the running viewer session.

### Method identity and resolution

The viewer uses each method's saved `config.yaml` as its method/configuration metadata source.

For TRIDENT and similar wrappers, the underlying model is shown prominently, for example:

```text
PathProfiler (TRIDENT, 4×)
```

MPP is shown only when an explicit `mpp` value exists in saved configuration.

---

## Download an Overview HTML report

The Overview toolbar contains a subtle **Download overview HTML** action.

The exported file is a self-contained visual snapshot containing:

- the current slide set and ordering;
- overview thumbnails;
- currently selected prediction outlines using their viewer colors;
- the ground-truth outline when enabled.

The exported outline styling matches the interactive Overview page. The downloaded HTML can be opened independently after export.

---

## Typical end-to-end workflow

```bash
# 1. Install
uv sync --extra all

# 2. Configure WSI_DIR and METHODS in run.py, then run segmentation
python run.py --output-dir outputs/my_experiment

# 3. Evaluate against GeoJSON ground truth
python -m segmenteer.evaluator \
  --output outputs/my_experiment \
  --ground-truth ground_truth_geojson \
  --format csv \
  --metrics outputs/my_experiment/metrics.csv

# 4. Launch the viewer
python -m app \
  --output outputs/my_experiment \
  --ground-truth ground_truth_geojson \
  --metrics outputs/my_experiment/metrics.csv \
  --data /path/to/wsi/files
```

For a prediction-only workflow, the runner output can be opened directly with:

```bash
python -m app --output outputs/my_experiment
```

For reference-free evaluation:

```bash
python -m segmenteer.evaluator \
  --output outputs/my_experiment \
  --format sqlite
```

## Citation

> TBD.

When using Segmenteer in a study, please cite the original publication(s) of the segmentation method(s) used.
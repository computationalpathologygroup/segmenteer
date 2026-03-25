<img src="docs/title.svg" alt="segmenteer" height="42">

<h6 style="color: gray;">Currently under development. </h6>

A repository for tissue segmentation on whole-slide images (WSI).
Currently it can run classical and deep-learning methods side-by-side and produce structured output including GeoJSON masks, unsupervised quality metrics, heatmaps.
<p style="color: #aaa; font-size: 0.75em;"><i><a href="https://github.com/computationalpathologygroup/annovert" style="color: #02B0dd; text-decoration: underline;">Annovert</a> can be used to convert the generated GeoJSONs to annotations format of interest including binary masks</i></p>


### How to run

First setup the environment [see below](#setup).

#### Run methods

Edit `run.py` — set your data path and comment out any methods that are not
installed or not wanted to be used — then run:

```bash
python run.py
```

Results are written to `outputs/<timestamp>/`. Each method gets its own
subdirectory containing predictions (GeoJSON), unsupervised metrics, and
heatmaps.

Raw outputs — GeoJSON masks, per-image heatmaps, and metric JSON files — are saved
directly in `outputs/<timestamp>/` and can be inspected at any time. The report and
viewer provide an interactive overview of the same data.

---

Once you have run the methods, the following functionalities are available.
**Ensemble fusion** is independent of report and viewer, but must be run first if you want the fused mask to appear as a method in them.

#### Ensemble fusion

Fuse predictions from multiple methods via soft weighted majority voting.
Edit `postensemble.py` — point `member_dirs` at the method directories you
want to include — then run:

```bash
python postensemble.py
```

The fused mask is written alongside the other method outputs in the same
`outputs/<timestamp>/` directory.

#### Report

Generate a self-contained HTML report with thumbnails, overlays, and
unsupervised metrics for every method:

```bash
python create_report.py --output outputs/<timestamp>
```

#### Viewer

Requires the `app` extra (`fastapi` + `uvicorn`) — install it if not already done:

```bash
uv sync --extra app
```

Launch an interactive viewer with overlay toggling, metric cards, and a
"Download segmentation report" button:

```bash
python -m app --output outputs/<timestamp> --data /path/to/wsi/files
```

Opens at `http://127.0.0.1:8765`. Use `--port <PORT>` to change the port.


### Setup

#### Prerequisites

- Python 3.12+
- preferred [uv](https://docs.astral.sh/uv/)

#### Quick start: install everything

```bash
git clone <repo-url> && cd segmenteer
uv sync --extra all
```

This installs all classical and DL methods with the OpenSlide WSI backend.

#### Install only methods/backends of interest

```bash
# WSI reading backend — pick one
uv sync --extra openslide
uv sync --extra cucim 

# Classical methods
uv sync --extra entropymasker
uv sync --extra background-subtractor
uv sync --extra od-gmm

# Deep learning methods
uv sync --extra hest
uv sync --extra grandqc
uv sync --extra rtlucassen
uv sync --extra bigpicture
uv sync --extra fastsam
uv sync --extra trident

# All DL methods at once
uv sync --extra dl
```

> **Always use `uv sync --extra <name>`, not `pip install` or `uv pip install`.**
> Several packages (Trident, SlideSegmenter, tissue-segmentation) are pinned to
> specific git commits and are only resolved correctly through `uv sync`.

#### HistomicsTK and BigPicture (uv pip installs)

These two methods cannot be managed by `uv sync` — their transitive dependencies
(`mapnik` / `tensorflow-cpu`) have no macOS ARM or Python 3.12 wheels, which breaks
the resolver. Install them directly with `uv pip install` **after** `uv sync`:

```bash
# HistomicsTK — macOS: brew install libtiff openslide first
uv pip install histomicstk large-image-source-tiff large-image-source-openslide

# BigPicture — macOS ARM: use tensorflow-macos + tensorflow-metal (not tensorflow-cpu)
uv pip install tensorflow-macos tensorflow-metal
uv pip install --no-deps "tissue-segmentation @ git+https://github.com/imi-bigpicture/tissue-segmentation.git@6d97a25a8255f591eb2c705611a32a5f56101a25#subdirectory=tissue_segmentation"

# BigPicture — Linux / Windows
uv pip install tensorflow
uv pip install "tissue-segmentation @ git+https://github.com/imi-bigpicture/tissue-segmentation.git@6d97a25a8255f591eb2c705611a32a5f56101a25#subdirectory=tissue_segmentation"
```


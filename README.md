<img src="docs/title.svg" alt="Segmenteer" height="42">

Segmenteer generates tissue annotations from whole-slide images (WSIs). It supports classical image processing and pretrained segmentation models. Predictions can be saved as **GeoJSON** or **ASAP XML**.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra all
```

Some methods need additional packages or pretrained model weights. Install the dependencies for the methods you want to use. Dependency groups are listed in `pyproject.toml`.

## Generate predictions

1. Select the segmentation methods in the `METHODS` list in `run.py`. For example:

   ```python
   METHODS = [
       seg.AtlasPatchSAM2Segmenter(device="cpu"),
   ]
   ```

2. Run segmentation on a directory of slides:

   ```bash
   uv run python run.py \
     --input-dir /path/to/slides \
     --output-dir outputs/tissue-masks \
     --workers 1 \
     --output-format geojson
   ```

Use `--output-format xml` to write ASAP XML annotations instead. Run `uv run python run.py --help` for other options.

Segmenteer saves predictions for each method. It also saves configurations and timing records. Completed predictions can be reused when the input files and method settings have not changed.

## View predictions

The optional browser viewer can display WSIs and prediction overlays:

```bash
uv run python -m app \
  --output outputs/tissue-masks \
  --data /path/to/slides
```

Open `http://127.0.0.1:8765` in your browser. Install the `app` dependency group if the viewer dependencies are not already available.

## Model downloads

Methods with pretrained models may download weights when they first run. Downloaded weights are cached for later runs. Some downloads require explicit approval:

```bash
SEGMENTEER_ALLOW_UNPINNED_DOWNLOADS=1 uv run python run.py --input-dir /path/to/slides
```

Use this setting only with trusted sources. Set `SEGMENTEER_OFFLINE=1` to disable Segmenteer-managed downloads. Make sure the required models are already cached.

## Evaluation (optional)

Segmenteer also supports supervised evaluation when ground-truth annotations are available. Metrics include Dice, IoU, and boundary measures. Evaluation is optional.

```bash
uv run python -m segmenteer.evaluator \
  --output outputs/tissue-masks \
  --ground-truth /path/to/annotations \
  --format csv
```

## Citation

When publishing results, cite the original publications for the segmentation methods and models used.

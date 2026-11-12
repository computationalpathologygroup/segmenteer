<img src="docs/title.svg" alt="Segmenteer" height="42">

Segmenteer generates tissue annotations from whole-slide images (WSIs). It supports classical image processing and pretrained segmentation models. Predictions can be saved as **GeoJSON** or **ASAP XML**.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra all
```

Some methods need additional packages or pretrained model weights. Install the dependencies for the methods you want to use. Dependency groups are listed in `pyproject.toml`.

Another option is to use the provided Dockerfile to build a docker image yourself with

```sh
docker build . --tag segmenteer
```

## Generate predictions

1. Gather the slides you need to run inference on and put them in a `dataset.csv` with only one `path` column. The rows should be absolute or relative paths to the slides.

2. Decide whether you would like to do inference with multiple methods or if one method is enough.
For single-method inference, run
```sh
segmenteer atlaspatch-sam2 \
    --dataset $DATASET_CSV \
    --output-dir $OUTPUT_DIR \
    --workers 1
```
You can substitue `atlaspatch-sam2` with any of the following values:
| Category | Methods |
|---|---|
| Classical methods | li, otsu, yen, morphological, watershed, hsv-threshold, hsv-threshold-v2, fesi, fesi-basic, entropy-masker, histomicstk, double-pass, canny-foreground, background-subtractor |
| Standalone deep-learning methods | atlaspatch-sam2, grandqc, grandqc-patchwise, hest, fastsam, bigpicture, rtlucassen, nki-all-stains, cx-tissue-seg, icgi-nnunetv2, icgi-resenc |
| TRIDENT-integrated methods | trident-cpg, trident-grandqc, trident-hest, trident-pathprofiler, trident-rtlucassen |

For multi-method inference, run
```sh
segmenteer multi-method \
    --dataset $DATASET_CSV \
    --configs $CONFIGS_YAML \
    --device cuda:0 \
    --output-dir $OUTPUT_DIR \
    --workers 1
```

Use `--output-format xml` to write ASAP XML annotations instead.
Run `segmenteer --help` for other options.

Segmenteer saves predictions for each method. It also saves configurations and timing records. Completed predictions can be reused when the input files and method settings have not changed.

These steps are wrapped in pyxis-enabled slurm jobs in `/slurm`.
You need to push your docker build to a repository and reference it in the jobscripts.

## View predictions

The optional browser viewer can display WSIs and prediction overlays:

```bash
segmenteer app \
  --output outputs/tissue-masks \
  --data /path/to/slides
```

Open `http://127.0.0.1:8765` in your browser.
Install the `app` dependency group if the viewer dependencies are not already available.
Use `--host 0.0.0.0` if you would like to reach the app from the network (e.g. if the app is hosted by a cluster machine).

## Model downloads

Methods with pretrained models may download weights when they first run.
Downloaded weights are cached for later runs.

## Evaluation

Segmenteer also supports supervised evaluation when ground-truth annotations are available. Metrics include Dice, IoU, and boundary measures. Evaluation is optional.

```bash
segmenteer evaluate \
  --output $OUTPUT_DIR \
  --annotations-csv $ANNOTATIONS_CSV \
  --format csv
```

## Citation

When publishing results, cite the original publications for the segmentation methods and models used and cite `segmenteer` as
```bibtex
@inproceedings{segmenteer,
  author = {...}
  // TODO
}
```

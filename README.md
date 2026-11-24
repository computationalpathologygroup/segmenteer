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

Note: retry if you see `Prediction or ground truth changed during evaluation; retry this result`.

You can also see the ground truth and the metrics in the app if you add `--metrics $METRICS_CSV` and `--ground-truth $GROUND_TRUTH` to `segmenteer app`. 


## Slurm

Inference, evaluation, and the app are wrapped in pyxis-enabled slurm jobs in `/slurm`.
You need to push your docker build to a repository and reference it in the jobscripts.

## References and citation

We built on many segmentation models and frameworks.
Below we cite them.
We are determined to attribute properly.
Please open an issue to request attribution correction.

| method | url | license | doi/paper      |
|--------|-----|---------|----------|
| CPG    |     |         |          |
| AtlasPatch | [AtlasAnalyticsLab/AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch/) | [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) | [10.48550/arXiv.2602.03998](https://doi.org/10.48550/arXiv.2602.03998) |
| GrandQC | [cpath-ukk/grandqc](https://github.com/cpath-ukk/grandqc/) | [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) | [10.1038/s41467-024-54769-y](https://doi.org/10.1038/s41467-024-54769-y) |
| HEST | [mahmoodlab/hest](https://github.com/mahmoodlab/hest) | [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) | [10.48550/arXiv.2406.16192](https://doi.org/10.48550/arXiv.2406.16192) |
| PathProfiler | [MaryamHaghighat/PathProfiler](https://github.com/MaryamHaghighat/PathProfiler) | [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) | [10.1038/s41598-022-08351-5](https://doi.org/10.1038/s41598-022-08351-5) |
| SlideSegmenter | [RTLucassen/slidesegmenter](https://github.com/RTLucassen/slidesegmenter) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) | [arXiv.2401.13511](https://doi.org/10.48550/arXiv.2401.13511) |
| TRIDENT | [mahmoodlab/Trident](https://github.com/mahmoodlab/Trident) | [CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en) | [10.48550/arXiv.2502.06750](https://doi.org/10.48550/arXiv.2502.06750) |
| FastSAM | [ultralytics/assets](https://github.com/ultralytics/assets/releases#release-v8.4.0) | [AGPL-v3.0](https://www.gnu.org/licenses/agpl-3.0.txt) |  |
| NKI all-stains | [huggingface.co/NKI-AI/tissue-bg-all-stains](https://huggingface.co/NKI-AI/tissue-bg-all-stains) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) |  |
| Bigpicture | [imi-bigpicture/tissue-segmentation](https://github.com/imi-bigpicture/tissue-segmentation) | [MIT](https://mit-license.org/) | [10.7717/peerj.8242](https://doi.org/10.7717/peerj.8242) |
| HistomicsTK | [DigitalSlideArchive/HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) |  |
| icgi | [icgi/Reliable-and-Efficient-Tissue-Segmentation-in-Whole-Slide-Images](https://github.com/icgi/Reliable-and-Efficient-Tissue-Segmentation-in-Whole-Slide-Images) | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | https://openreview.net/forum?id=3gsNUjFi1q |
| CxTissueSeg | https://huggingface.co/conflux-xyz/cx-tissue-seg | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) | | 
| FESI | [NKI-AI/dlup](https://github.com/NKI-AI/dlup/blob/52a65d06a4f636b94cb142d00ccfc2f98b6a966b/dlup/background.py) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) | [10.1007/978-3-662-46224-9_72](https://doi.org/10.1007/978-3-662-46224-9_72) |
| FESI Improved | [NKI-AI/dlup](https://github.com/NKI-AI/dlup/blob/52a65d06a4f636b94cb142d00ccfc2f98b6a966b/dlup/background.py) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) | [10.48550/arXiv.2006.06531](https://doi.org/10.48550/arXiv.2006.06531) |
| EntropyMasker | [CirculatoryHealth/EntropyMasker](https://github.com/CirculatoryHealth/EntropyMasker) | N/A (reimplementation) | [10.1101/2022.09.01.22279487](https://doi.org/10.1101/2022.09.01.22279487) |
| Canny | [icgi/automatic-tumour-segmentation-in-WSIs](https://github.com/icgi/automatic-tumour-segmentation-in-WSIs) | N/A (reimplementation) | [10.1038/s41698-026-01311-6](https://doi.org/10.1038/s41698-026-01311-6) |
| Li |  | N/A (reimplementation) | [10.5220/0011803500003414](https://doi.org/10.5220/0011803500003414) |
| Yen |  | N/A (reimplementation) | [10.1109/83.366472](https://doi.org/10.1109/83.366472)
| Watershed |  | N/A (reimplementation) | [10.1109/34.87344](https://doi.org/10.1109/34.87344) |
| Otsu |  | N/A (reimplementation) | [10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076) |
| HSV |  | N/A (reimplementation) | [10.48550/arXiv.2408.00738](https://doi.org/10.48550/arXiv.2408.00738)
| Double-pass | [aimas-upb/efficient-tissue-detection-wsi](https://github.com/aimas-upb/efficient-tissue-detection-wsi) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.html) | [10.3390/cancers17172918](https://doi.org/10.3390/cancers17172918) |

See the `LICENSES` for license copies.
When publishing results, cite the original publications for the segmentation methods and models used and cite `segmenteer` as
```bibtex
@inproceedings{segmenteer,
  author = {...}
  // TODO. Awaiting arXiv publication.
}
```

# License
This repository is licensed APGLv3 as it depends on APGL code.
See the commit history for changes.

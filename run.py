"""Segmenteer experiment launcher.

Edit the configuration and METHODS list below, then run::

    python run.py

Optional command-line overrides are available with ``python run.py --help``.
"""

from pathlib import Path

import segmenteer as seg


PROJECT_ROOT = Path(__file__).resolve().parent


# =============================================================================
# CONFIGURATION — edit this section
# =============================================================================

# Whole-slide images.
WSI_DIR = Path(
    ""
)

# Runner outputs.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR: Path | None = None  # None -> new timestamped experiment

# Dataset / process settings.
SLIDE_ORDER = "smallest-first"  # "smallest-first" | "alphabetical"
WORKERS: int | str = "auto"     # "auto" | positive integer

# Runtime settings. Apply these before constructing METHODS.
RUNTIME = seg.configure_runtime(
    device="auto",               # "auto" | "cpu" | "mps" | "cuda:0" | ...
    trident_device="mps",        # "auto" follows device above
    models_dir=PROJECT_ROOT / "models",
)


# =============================================================================
# METHODS — add/remove configurations here
# =============================================================================

METHODS = [
    # seg.EntropyMaskerSegmenter(mpp=20),
    # seg.BackgroundSubtractorMOG2Segmenter(mpp=20),
    # seg.FESISegmenter(mpp=20),
    # seg.FESISegmenter(mpp=20, improved=False),
    # seg.FESISegmenter(mpp=10),
    # seg.FESISegmenter(mpp=10, improved=False),
    # seg.HistomicsTKSegmenter(mpp=20),
    # seg.HistomicsTKSegmenter(mpp=10),
    # seg.OtsuSegmenter(mpp=20),
    # seg.BigPictureSegmenter(mpp=8, device=BENCHMARK_DEVICE),
    # seg.BigPictureSegmenter(mpp=8, device=BENCHMARK_DEVICE, select_largest_tissue_objects=True, apply_hole_filling=False),
    # seg.BigPictureSegmenter(mpp=8, device=BENCHMARK_DEVICE),
    # seg.BigPictureSegmenter(
    #     mpp=8,
    #     device=BENCHMARK_DEVICE,
    #     dilation_disk_size=32,
    #     confidence_threshold=0.8,
    #     dilate_mask=True,
    #     apply_hole_filling=True,
    #     select_largest_tissue_objects=True,
    # ),
    # seg.WatershedSegmenter(mpp=20),
    # seg.WatershedSegmenter(mpp=10),
    # seg.WatershedSegmenter(mpp=5),
    # seg.HSVThresholdSegmenter(mpp=20),
    # seg.HSVThresholdSegmenter(mpp=10, min_area=1000),
    # seg.HSVThresholdSegmenter(mpp=5),
    # seg.OtsuSegmenter(mpp=10),
    # seg.OtsuSegmenter(mpp=5),
    # seg.OtsuTissueSegmenter(mpp=20),

    # seg.EntropyMaskerSegmenter(mpp=10),
    # seg.EntropyMaskerSegmenter(mpp=5),
    # seg.FESISegmenter(mpp=5),
    # seg.FESISegmenter(mpp=5, improved=False),
    # seg.HistomicsTKSegmenter(mpp=5),

    # seg.RTLucassenSlideSegmenter(mpp=7.04, device=BENCHMARK_DEVICE),
    # seg.TRIDENTCPGSegmenter,
    # seg.TRIDENTPathProfilerSegmenter,
    # seg.TRIDENTGrandQCSegmenter,
    # seg.TRIDENTHESTSegmenter,

    # # Classical methods
    # seg.LiSegmenter(mpp=20),
    # seg.YenSegmenter(mpp=20),
    # seg.MorphologicalSegmenter(mpp=10),
    # seg.HSVThresholdSegmenter(mpp=10),
    # seg.ODGMMSlideSegmenter(mpp=10),

    # # # Direct deep-learning wrappers (all receive the shared device policy)
    # seg.FastSAMSegmenter(mpp=20, device=BENCHMARK_DEVICE),
    # seg.FastSAMSegmenter(mpp=10, device=BENCHMARK_DEVICE),
    # seg.FastSAMSegmenter(mpp=5, device=BENCHMARK_DEVICE),

    # seg.GrandQCSegmenter(mpp=10, device=BENCHMARK_DEVICE),
    # seg.HESTSegmenter(mpp=20, device=BENCHMARK_DEVICE),
    # seg.RTLucassenSlideSegmenter(mpp=7.04, device=BENCHMARK_DEVICE),
    seg.AtlasPatchSAM2Segmenter(device="cpu"),

    # # # Trident-backed deep-learning methods
    # seg.TRIDENTHESTSegmenter,
    # seg.TRIDENTGrandQCSegmenter,
    # seg.TRIDENTPathProfilerSegmenter,
    # seg.TRIDENTCPGSegmenter,

    # seg.WatershedTissueSegmenter(
    #     mpp=20,
    #     min_distance_um=100,
    # ),
    # seg.LiTissueSegmenter(mpp=10),
    # seg.LiTissueSegmenter(mpp=5),

    # seg.YenTissueSegmenter(mpp=20),
    # seg.YenTissueSegmenter(mpp=10),
    # seg.YenTissueSegmenter(mpp=5),

    # seg.HistomicsTKTissueSegmenter(
    #     mpp=20,
    #     mask_type="simple",
    # ),
    # seg.HistomicsTKTissueSegmenter(
    #     mpp=20,
    #     mask_type="saliency",
    # ),
    # seg.HistomicsTKTissueSegmenter(
    #     mpp=10,
    #     mask_type="saliency",
    # ),
    # seg.OtsuTissueSegmenter(mpp=5),
]


if __name__ == "__main__":
    seg.run_directory_cli(
        METHODS,
        wsi_dir=WSI_DIR,
        output_root=OUTPUT_ROOT,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        slide_order=SLIDE_ORDER,
        workers=WORKERS,
        project_root=PROJECT_ROOT,
    )

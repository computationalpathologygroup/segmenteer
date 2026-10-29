"""Segmenteer experiment launcher.

Edit the configuration and METHODS list below, then run::

    python run.py

Optional command-line overrides are available with ``python run.py --help``.
"""

import segmenteer as seg


# =============================================================================
# CONFIGURATION — edit this section
# =============================================================================

# Dataset / process settings.
SLIDE_ORDER = "smallest-first"  # "smallest-first" | "alphabetical"
WORKERS: int | str = "auto"  # "auto" | positive integer
BENCHMARK_DEVICE = "cpu"


# =============================================================================
# METHODS — add/remove configurations here
# =============================================================================

METHODS = [
    seg.LiTissueSegmenter(mpp=20, min_area=1000),
    seg.LiTissueSegmenter(mpp=10, min_area=1000),
    seg.OtsuTissueSegmenter(mpp=20, min_area=1000),
    seg.OtsuTissueSegmenter(mpp=10, min_area=1000),
    seg.YenTissueSegmenter(mpp=20, min_area=1000),
    seg.YenTissueSegmenter(mpp=10, min_area=1000),
    seg.MorphologicalSegmenter(
        mpp=20,
        disk_size=3,
        min_area=1000,
    ),
    seg.MorphologicalSegmenter(
        mpp=10,
        disk_size=3,
        min_area=1000,
    ),
    seg.WatershedTissueSegmenter(
        mpp=20,
        min_distance_um=200,
        min_area=1000,
        merge_regions=True,
    ),
    seg.WatershedTissueSegmenter(
        mpp=10,
        min_distance_um=200,
        min_area=1000,
        merge_regions=True,
    ),
    seg.HSVThresholdSegmenter(
        mpp=20,
        min_area=1000,
    ),
    seg.HSVThresholdSegmenter(
        mpp=10,
        min_area=1000,
    ),
    seg.HSVThreshold_v2(
        mpp=20,
        min_area=1000,
    ),
    seg.HSVThreshold_v2(
        mpp=10,
        min_area=1000,
    ),
    seg.FESISegmenter(mpp=20),
    seg.FESISegmenter(mpp=20, improved=False),
    seg.FESISegmenter(mpp=10),
    seg.FESISegmenter(mpp=10, improved=False),
    seg.EntropyMaskerSegmenter(mpp=20),
    seg.EntropyMaskerSegmenter(mpp=10),
    seg.HistomicsTKTissueSegmenter(mpp=20),
    seg.HistomicsTKTissueSegmenter(mpp=10),
    seg.DoublePassSegmenter(mpp=10),
    seg.CxTissueSegSegmenter(
        mpp=40,
        device=BENCHMARK_DEVICE,
    ),
    seg.CannyForegroundSegmenter(mpp=5),
    seg.AtlasPatchSAM2Segmenter(device="cpu"),
    seg.GrandQCPatchwiseSegmenter(),
    seg.TRIDENTGrandQCSegmenter,

    seg.NKIAllStainsSegmenter(
        mpp=12,
        device=BENCHMARK_DEVICE,
        batch_size=1,
    ),
    seg.FastSAMSegmenter(mpp=20, device=BENCHMARK_DEVICE),
    seg.BigPictureSegmenter(
        mpp=8,
        device=BENCHMARK_DEVICE,
        select_largest_tissue_objects=True,
        apply_hole_filling=False,
    ),
    seg.RTLucassenSlideSegmenter(mpp=7.04, device=BENCHMARK_DEVICE),
    seg.TRIDENTRTLucassenSlideSegmenterSegmenter,

    seg.TRIDENTHESTSegmenter,
    seg.TRIDENTPathProfilerSegmenter,
    seg.ICGINnUNetV2Segmenter(
        mpp=10,
        device=BENCHMARK_DEVICE,
    ),
    seg.ICGIResEncSegmenter(
        mpp=10,
        device=BENCHMARK_DEVICE,
    ),

    # # seg.TRIDENTCPGSegmenter,
]


if __name__ == "__main__":
    seg.run_directory_cli(
        METHODS,
        slide_order=SLIDE_ORDER,
        workers=WORKERS,
    )

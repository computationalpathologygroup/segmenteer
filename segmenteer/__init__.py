from segmenteer.core import (
    Segmenter,
    SegmentationResult,
    mask_to_geojson,
    geojson_to_mask,
    downsample_image,
    scale_geojson_coordinates,
)
from segmenteer.methods import (
    OtsuSegmenter,
    LiSegmenter,
    YenSegmenter,
    EntropyMaskerSegmenter,
    MorphologicalSegmenter,
    WatershedSegmenter,
    BackgroundSubtractorMOG2Segmenter,
    UnsupervisedClusteringSegmenter,
)
from segmenteer.metrics import (
    SupervisedMetrics,
    UnsupervisedMetrics,
    compute_dice,
    compute_iou,
    compute_hausdorff,
    compute_precision,
    compute_recall,
    compute_all_supervised_metrics,
    compute_unsupervised_metrics,
)
from segmenteer.benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
    export_results_json,
    export_results_csv,
)
from segmenteer.io import (
    load_image,
    save_geojson,
    load_geojson,
    create_timestamped_output_dir,
)
from segmenteer.visualization import (
    create_thumbnail,
    create_heatmap_overlay,
    save_thumbnail,
    save_heatmap_thumbnail,
)
from segmenteer.wsi import (
    WSIMetadata,
    get_wsi_metadata,
    calculate_target_level,
    resample_to_mpp,
    load_wsi_at_mpp,
    estimate_mpp_from_magnification,
    calculate_scale_factor_for_coordinates,
)

__version__ = "0.1.0"


def __getattr__(name):
    if name in ("HESTSegmenter", "GrandQCSegmenter", "FastSAMSegmenter"):
        from segmenteer.methods.dl import HESTSegmenter, GrandQCSegmenter, FastSAMSegmenter, CPGSegmenter

        return {
            "HESTSegmenter": HESTSegmenter,
            "GrandQCSegmenter": GrandQCSegmenter,
            "FastSAMSegmenter": FastSAMSegmenter,
            "CPGSegmenter": CPGSegmenter,
        }[name]
    raise AttributeError(f"module 'segmenteer' has no attribute '{name}'")


__all__ = [
    "Segmenter",
    "SegmentationResult",
    "mask_to_geojson",
    "geojson_to_mask",
    "downsample_image",
    "scale_geojson_coordinates",
    "OtsuSegmenter",
    "LiSegmenter",
    "YenSegmenter",
    "EntropyMaskerSegmenter",
    "MorphologicalSegmenter",
    "WatershedSegmenter",
    "BackgroundSubtractorMOG2Segmenter",
    "UnsupervisedClusteringSegmenter",
    "CPGSegmenter",
    "HESTSegmenter",
    "GrandQCSegmenter",
    "FastSAMSegmenter",
    "SupervisedMetrics",
    "UnsupervisedMetrics",
    "compute_dice",
    "compute_iou",
    "compute_hausdorff",
    "compute_precision",
    "compute_recall",
    "compute_all_supervised_metrics",
    "compute_unsupervised_metrics",
    "BenchmarkRunner",
    "BenchmarkResult",
    "export_results_json",
    "export_results_csv",
    "load_image",
    "save_geojson",
    "load_geojson",
    "create_timestamped_output_dir",
    "create_thumbnail",
    "create_heatmap_overlay",
    "save_thumbnail",
    "save_heatmap_thumbnail",
    "WSIMetadata",
    "get_wsi_metadata",
    "calculate_target_level",
    "resample_to_mpp",
    "load_wsi_at_mpp",
    "estimate_mpp_from_magnification",
    "calculate_scale_factor_for_coordinates",
]

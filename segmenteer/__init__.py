from __future__ import annotations

# Apply the offline policy to supported third-party model hubs.
import os as _os

if _os.environ.get("SEGMENTEER_OFFLINE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
):
    for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        _os.environ[_key] = "1"
del _os
from importlib import import_module
from typing import Any

__version__ = "0.1.0"
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BenchmarkReporter": ("segmenteer.benchmark.console", "BenchmarkReporter"),
    "BenchmarkResult": ("segmenteer.benchmark.runner", "BenchmarkResult"),
    "BenchmarkRunner": ("segmenteer.benchmark.runner", "BenchmarkRunner"),
    "PredictionOutputWriter": ("segmenteer.benchmark.output", "PredictionOutputWriter"),
    "RuntimeSettings": ("segmenteer.core.runtime", "RuntimeSettings"),
    "configure_runtime": ("segmenteer.core.runtime", "configure_runtime"),
    "discover_slides": ("segmenteer.runner", "discover_slides"),
    "run_directory": ("segmenteer.runner", "run_directory"),
    "run_directory_cli": ("segmenteer.runner", "run_directory_cli"),
    "EvaluationSummary": ("segmenteer.benchmark.evaluation", "EvaluationSummary"),
    "evaluate_output_directory": (
        "segmenteer.benchmark.evaluation",
        "evaluate_output_directory",
    ),
    "export_dataset_results_csv": (
        "segmenteer.benchmark.reporting",
        "export_dataset_results_csv",
    ),
    "export_results_csv": ("segmenteer.benchmark.reporting", "export_results_csv"),
    "export_results_json": ("segmenteer.benchmark.reporting", "export_results_json"),
    "make_run_id": ("segmenteer.benchmark.runner", "make_run_id"),
    "make_run_ids": ("segmenteer.benchmark.runner", "make_run_ids"),
    "run_dataset": ("segmenteer.benchmark.workflows", "run_dataset"),
    "run_single_image": ("segmenteer.benchmark.workflows", "run_single_image"),
    "prepare_dataset_output": (
        "segmenteer.benchmark.workflows",
        "prepare_dataset_output",
    ),
    "CannyForegroundSegmenter": (
        "segmenteer.methods.classical.canny_foreground",
        "CannyForegroundSegmenter",
    ),
    "DoublePassSegmenter": (
        "segmenteer.methods.classical.double_pass",
        "DoublePassSegmenter",
    ),
    "SegmentationResult": ("segmenteer.core.base", "SegmentationResult"),
    "Segmenter": ("segmenteer.core.base", "Segmenter"),
    "WSIBackend": ("segmenteer.core.base", "WSIBackend"),
    "downsample_image": ("segmenteer.core.utils", "downsample_image"),
    "geojson_to_mask": ("segmenteer.core.utils", "geojson_to_mask"),
    "load_segmenter": ("segmenteer.core.base", "load_segmenter"),
    "mask_to_geojson": ("segmenteer.core.utils", "mask_to_geojson"),
    "scale_geojson_coordinates": ("segmenteer.core.utils", "scale_geojson_coordinates"),
    "create_timestamped_output_dir": (
        "segmenteer.io.utils",
        "create_timestamped_output_dir",
    ),
    "convert_output_directory": ("segmenteer.io.convert", "convert_output_directory"),
    "save_asap_xml": ("segmenteer.io.asap", "save_asap_xml"),
    "load_asap_xml": ("segmenteer.io.asap", "load_asap_xml"),
    "GroundTruthPairing": ("segmenteer.io.loader", "GroundTruthPairing"),
    "format_ground_truth_pairing_error": (
        "segmenteer.io.loader",
        "format_ground_truth_pairing_error",
    ),
    "inspect_ground_truths": ("segmenteer.io.loader", "inspect_ground_truths"),
    "load_geojson": ("segmenteer.io.loader", "load_geojson"),
    "load_ground_truths": ("segmenteer.io.loader", "load_ground_truths"),
    "load_image": ("segmenteer.io.loader", "load_image"),
    "save_geojson": ("segmenteer.io.loader", "save_geojson"),
    "SupervisedMetrics": ("segmenteer.metrics.supervised", "SupervisedMetrics"),
    "compute_all_supervised_metrics": (
        "segmenteer.metrics.supervised",
        "compute_all_supervised_metrics",
    ),
    "compute_balanced_error_rate": (
        "segmenteer.metrics.supervised",
        "compute_balanced_error_rate",
    ),
    "compute_dice": ("segmenteer.metrics.supervised", "compute_dice"),
    "compute_iou": ("segmenteer.metrics.supervised", "compute_iou"),
    "compute_mae": ("segmenteer.metrics.supervised", "compute_mae"),
    "compute_over_segmentation_rate": (
        "segmenteer.metrics.supervised",
        "compute_over_segmentation_rate",
    ),
    "compute_pixel_accuracy": (
        "segmenteer.metrics.supervised",
        "compute_pixel_accuracy",
    ),
    "compute_precision": ("segmenteer.metrics.supervised", "compute_precision"),
    "compute_recall": ("segmenteer.metrics.supervised", "compute_recall"),
    "compute_under_segmentation_rate": (
        "segmenteer.metrics.supervised",
        "compute_under_segmentation_rate",
    ),
    "create_heatmap_overlay": (
        "segmenteer.visualization.heatmaps",
        "create_heatmap_overlay",
    ),
    "create_thumbnail": ("segmenteer.visualization.thumbnails", "create_thumbnail"),
    "save_heatmap_thumbnail": (
        "segmenteer.visualization.heatmaps",
        "save_heatmap_thumbnail",
    ),
    "save_thumbnail": ("segmenteer.visualization.thumbnails", "save_thumbnail"),
    "WSIMetadata": ("segmenteer.wsi.utils", "WSIMetadata"),
    "calculate_scale_factor_for_coordinates": (
        "segmenteer.wsi.utils",
        "calculate_scale_factor_for_coordinates",
    ),
    "calculate_target_level": ("segmenteer.wsi.utils", "calculate_target_level"),
    "estimate_mpp_from_magnification": (
        "segmenteer.wsi.utils",
        "estimate_mpp_from_magnification",
    ),
    "get_wsi_metadata": ("segmenteer.wsi.utils", "get_wsi_metadata"),
    "load_wsi_at_mpp": ("segmenteer.wsi.utils", "load_wsi_at_mpp"),
    "resample_to_mpp": ("segmenteer.wsi.utils", "resample_to_mpp"),
    "LiTissueSegmenter": (
        "segmenteer.methods.classical.threshold",
        "LiTissueSegmenter",
    ),
    "YenTissueSegmenter": (
        "segmenteer.methods.classical.threshold",
        "YenTissueSegmenter",
    ),
    "EntropyMaskerSegmenter": (
        "segmenteer.methods.classical.threshold",
        "EntropyMaskerSegmenter",
    ),
    "MorphologicalSegmenter": (
        "segmenteer.methods.classical.morphological",
        "MorphologicalSegmenter",
    ),
    "BackgroundSubtractorMOG2Segmenter": (
        "segmenteer.methods.classical.background_subtractor",
        "BackgroundSubtractorMOG2Segmenter",
    ),
    "HSVThresholdSegmenter": (
        "segmenteer.methods.classical.hsv_threshold",
        "HSVThresholdSegmenter",
    ),
    "HSVThreshold_v2": (
        "segmenteer.methods.classical.hsv_threshold_v2",
        "HSVThreshold_v2",
    ),
    "HistomicsTKTissueSegmenter": (
        "segmenteer.methods.classical.histomicstk",
        "HistomicsTKTissueSegmenter",
    ),
    "FESISegmenter": ("segmenteer.methods.classical.fesi", "FESISegmenter"),
    "OtsuTissueSegmenter": (
        "segmenteer.methods.classical.threshold",
        "OtsuTissueSegmenter",
    ),
    "WatershedTissueSegmenter": (
        "segmenteer.methods.classical.morphological",
        "WatershedTissueSegmenter",
    ),
    "AtlasPatchSAM2Segmenter": (
        "segmenteer.methods.dl.atlaspatch_sam2",
        "AtlasPatchSAM2Segmenter",
    ),
    "BigPictureSegmenter": ("segmenteer.methods.dl.bigpicture", "BigPictureSegmenter"),
    "FastSAMSegmenter": ("segmenteer.methods.dl.fastsam", "FastSAMSegmenter"),
    "GrandQCSegmenter": ("segmenteer.methods.dl.grandqc", "GrandQCSegmenter"),
    "GrandQCPatchwiseSegmenter": (
        "segmenteer.methods.dl.grandqc_patchwise",
        "GrandQCPatchwiseSegmenter",
    ),
    "NKIAllStainsSegmenter": (
        "segmenteer.methods.dl.nki_all_stains",
        "NKIAllStainsSegmenter",
    ),
    "HESTSegmenter": ("segmenteer.methods.dl.hest", "HESTSegmenter"),
    "RTLucassenSlideSegmenter": (
        "segmenteer.methods.dl.rtlucassen",
        "RTLucassenSlideSegmenter",
    ),
    "TRIDENTGrandQCSegmenter": (
        "segmenteer.methods.dl.trident",
        "TRIDENTGrandQCSegmenter",
    ),
    "TRIDENTHESTSegmenter": ("segmenteer.methods.dl.trident", "TRIDENTHESTSegmenter"),
    "TRIDENTPathProfilerSegmenter": (
        "segmenteer.methods.dl.trident",
        "TRIDENTPathProfilerSegmenter",
    ),
    "TRIDENTCPGSegmenter": ("segmenteer.methods.dl.trident", "TRIDENTCPGSegmenter"),
    "TRIDENTRTLucassenSlideSegmenterSegmenter": (
        "segmenteer.methods.dl.trident",
        "TRIDENTRTLucassenSlideSegmenterSegmenter",
    ),
    "ICGINnUNetV2Segmenter": ("segmenteer.methods.dl.icgi", "ICGINnUNetV2Segmenter"),
    "ICGIResEncSegmenter": ("segmenteer.methods.dl.icgi", "ICGIResEncSegmenter"),
    "CxTissueSegSegmenter": (
        "segmenteer.methods.dl.cx_tissue_seg",
        "CxTissueSegSegmenter",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'segmenteer' has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["__version__", *_LAZY_EXPORTS]

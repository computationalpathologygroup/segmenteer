"""Resolve architecture display labels independently of experiment identities.

Labels do not affect serialized configurations, run IDs, or fingerprints.
"""

from __future__ import annotations

from collections.abc import Mapping


# Architecture labels are based on the corresponding model implementations.
_BY_CLASS = {
    "HESTSegmenter": "HEST/DeepLabV3-ResNet50",
    "GrandQCSegmenter": "GrandQC/U-Net++ (EfficientNet-B0)",
    "GrandQCPatchwiseSegmenter": "GrandQC patchwise/U-Net++ (EfficientNet-B0)",
    "AtlasPatchSAM2Segmenter": "AtlasPatch/SAM2.1-Hiera-T",
    "FastSAMSegmenter": "FastSAM/YOLOv8-seg",
    "NKIAllStainsSegmenter": "NKI All-Stains/MONAI U-Net",
    "BigPictureSegmenter": "BigPicture/Resolution-agnostic CNN",
    "RTLucassenSlideSegmenter": "RTLucassen/SlideSegmenter CNN",
    "ICGINnUNetV2Segmenter": "ICGI/nnU-Net v2 (10 µm/px)",
    "ICGIResEncSegmenter": "ICGI/nnU-Net v2 ResEnc (10 µm/px)",
    "CannyForegroundSegmenter": "Canny foreground/Canny + morphology (classical, 5 µm/px)",
    "DoublePassSegmenter": "Double-Pass/FilterGrays + downsampled K-means (classical)",
    "CxTissueSegSegmenter": "CxTissueSeg/U-Net (MobileNetV3-Small, 40 µm/px)",
}

# Match stored run-ID prefixes before class-name fallbacks.
_BY_RUN_PREFIX = (
    ("canny_foreground_mpp5", _BY_CLASS["CannyForegroundSegmenter"]),
    ("double_pass_mpp10", _BY_CLASS["DoublePassSegmenter"]),
    ("icgi_resenc_mpp10", _BY_CLASS["ICGIResEncSegmenter"]),
    ("cx_tissue_seg_mpp40", _BY_CLASS["CxTissueSegSegmenter"]),
    ("icgi_nnunetv2_mpp10", _BY_CLASS["ICGINnUNetV2Segmenter"]),
    ("trident_grandqc", "TRIDENT GrandQC/U-Net++"),
    ("trident_hest", "TRIDENT HEST/DeepLabV3-ResNet50"),
    ("trident_pathprofiler", "TRIDENT PathProfiler/U-Net"),
    ("trident_rtlucassen", "TRIDENT RTLucassen/SlideSegmenter CNN"),
    ("nki_all_stains", _BY_CLASS["NKIAllStainsSegmenter"]),
    ("bigpicture", _BY_CLASS["BigPictureSegmenter"]),
    ("rtlucassen", _BY_CLASS["RTLucassenSlideSegmenter"]),
    ("grandqc_patchwise", _BY_CLASS["GrandQCPatchwiseSegmenter"]),
    ("grandqc_", _BY_CLASS["GrandQCSegmenter"]),
    ("hest_deeplabv3", _BY_CLASS["HESTSegmenter"]),
    ("atlaspatch_sam2", _BY_CLASS["AtlasPatchSAM2Segmenter"]),
    ("fastsam_", _BY_CLASS["FastSAMSegmenter"]),
)


def architecture_label(
    *,
    run_id: str = "",
    formal_name: str = "",
    config: Mapping | None = None,
) -> str:
    """Resolve an informative display label without mutating stored artifacts."""
    config = config if isinstance(config, Mapping) else {}
    cls_name = str(config.get("class") or "").rsplit(".", 1)[-1]
    params = config.get("params") if isinstance(config.get("params"), Mapping) else {}
    if cls_name == "TRIDENTSegmenter":
        model_id = str(params.get("model_id", "")).casefold()
        if model_id == "hest":
            return "TRIDENT HEST/DeepLabV3-ResNet50"
        if model_id == "grandqc":
            return "TRIDENT GrandQC/U-Net++"
        if model_id == "pathprofiler":
            return "TRIDENT PathProfiler/U-Net"
        if model_id == "rtlucassen":
            return "TRIDENT RTLucassen/SlideSegmenter CNN"
    if cls_name in _BY_CLASS:
        return _BY_CLASS[cls_name]
    stem = str(run_id).split("__", 1)[0].casefold()
    for prefix, label in _BY_RUN_PREFIX:
        if stem.startswith(prefix):
            return label
    # Older evaluation records may contain class names only.
    short_name = str(formal_name or "").rsplit(".", 1)[-1].strip()
    if short_name in _BY_CLASS:
        return _BY_CLASS[short_name]
    if short_name in ("TRIDENT HEST", "TRIDENTHESTSegmenter"):
        return "TRIDENT HEST/DeepLabV3-ResNet50"
    if short_name in ("TRIDENT GrandQC", "TRIDENTGrandQCSegmenter"):
        return "TRIDENT GrandQC/U-Net++"
    if short_name in ("TRIDENT PathProfiler", "TRIDENTPathProfilerSegmenter"):
        return "TRIDENT PathProfiler/U-Net"
    if "/" in short_name:
        return short_name
    return short_name or cls_name or str(run_id)


def display_name_for_segmenter(segmenter: object) -> str:
    """Resolve a label without changing the method identifier."""
    legacy = getattr(segmenter, "display_name", getattr(segmenter, "name", "Unknown"))
    cls_name = type(segmenter).__name__
    if cls_name == "TRIDENTSegmenter":
        return architecture_label(
            run_id=str(getattr(segmenter, "name", "")),
            formal_name=str(legacy),
            config={
                "class": cls_name,
                "params": {"model_id": getattr(segmenter, "model_id", "")},
            },
        )
    return architecture_label(
        run_id=str(getattr(segmenter, "name", "")),
        formal_name=str(legacy),
        config={"class": cls_name},
    )

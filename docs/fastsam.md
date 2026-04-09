# FastSAM

Fast Segment Anything Model (FastSAM) for image segmentation via the ultralytics FastSAM implementation.

## Overview

FastSAM is a fast, lightweight segmentation model that can perform instance segmentation on images with optional text-based prompting. The implementation provides two modes:

1. **Segment everything** (default): Detects and segments all objects in the image
2. **Text-prompted segmentation**: Detects and segments objects matching a text description

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"FastSAM-x.pt"` | Model checkpoint name. Available: `FastSAM-s.pt`, `FastSAM-m.pt`, `FastSAM-x.pt` |
| `text_prompt` | str or None | `None` | Optional text description for guided segmentation. If `None`, segments all objects |
| `conf` | float | `0.4` | Confidence threshold for object detection (0.0 to 1.0) |
| `iou` | float | `0.9` | IoU (Intersection over Union) threshold for mask post-processing (0.0 to 1.0) |
| `device` | str or None | `None` | Computation device: `"cuda"`, `"cpu"`, or `None` for auto-detection. Defaults to CUDA if available |
| `imgsz` | int | `1024` | Input image size for the model |
| `mpp` | float | `10` | Microns-per-pixel resolution at which the WSI is analyzed |
| `min_area` | int | `10` | Minimum polygon area in pixels at the specified mpp resolution |

## How It Works

1. **Model Loading**: On instantiation, loads the specified FastSAM model from disk or downloads it from the ultralytics repository
   - Models are cached in `<workspace>/models/fastsam/` for reuse
2. **Device Selection**: Automatically detects GPU availability via PyTorch; falls back to CPU if CUDA unavailable
3. **Inference**: Runs the FastSAM model on the RGB image
   - If `text_prompt` is provided, uses text-guided segmentation
   - Otherwise, segments all detected objects
4. **Mask Processing**: 
   - Combines all detected masks via logical OR (union of all segments)
   - Applies confidence and IoU thresholds during inference
   - Resizes mask to match input image dimensions if needed
   - Converts to boolean mask
5. **Output**: Returns binary mask where detected objects are marked as foreground

## Model Variants

| Model | Size | Use Case |
|-------|------|----------|
| `FastSAM-s.pt` | Small | Speed-prioritized tasks |
| `FastSAM-x.pt` | Extra-large | Accuracy-prioritized tasks (default) |

## Dependencies

- `ultralytics` — FastSAM model and inference engine
- `torch` — For CUDA availability detection

Install with:
```
pip install 'segmenteer[fastsam]'
```

## Input Requirements

- Requires RGB input; does not apply grayscale conversion (`APPLY_TO_GRAYSCALE = False`)
- Expects uint8 image arrays

## Notes

- Model downloads are automatic on first use; subsequent uses load from cache
- Text prompts use natural language descriptions (e.g., "tissue", "background", "nuclei")
- The confidence threshold (`conf`) controls detection sensitivity; lower values detect more objects
- The IoU threshold controls mask quality; higher values remove overlapping or fragmented predictions
- GPU acceleration significantly speeds up processing; CPU mode is substantially slower
- All detected masks are combined into a single binary foreground mask
- If no objects are detected, returns an empty mask

## Example Usage

```python
from segmenteer.methods.dl.fastsam import FastSAMSegmenter
from pathlib import Path

# Segment everything
segmenter = FastSAMSegmenter(
    model_name="FastSAM-x.pt",
    mpp=10,
    conf=0.4,
    iou=0.9
)

mask = segmenter.segment(Path("image.svs"))

# Text-prompted segmentation
segmenter_prompted = FastSAMSegmenter(
    model_name="FastSAM-x.pt",
    text_prompt="tissue",
    conf=0.4,
    iou=0.9
)

mask_tissue = segmenter_prompted.segment(Path("image.svs"))
```

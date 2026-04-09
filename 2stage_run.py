"""Two-stage segmentation pipeline.

Stage 1 — Context definers
    Each definer reads a downsampled WSI thumbnail and sends it to an Ollama
    vision model, which returns a short natural-language prompt describing the
    tissue content.

Stage 2 — Prompt-based segmenters
    Each prompt-capable segmenter (CONCH, FastSAM, SAM3) receives the prompt
    produced by stage 1 and uses it to guide segmentation.

The Cartesian product of context_definers × segmenters is built automatically:
each (definer, segmenter) pair becomes one :class:`ContextualSegmenter` entry
that is benchmarked identically to a conventional segmenter in run.py.

Model weights are loaded once per segmenter instance — only the prompt string
changes between images; there is no per-image model-loading overhead.

Usage
-----
    python 2stage_run.py

Configuration
-------------
Edit the three sections below:
  1. CONTEXT_DEFINERS  — your Ollama (or other) context providers
  2. SEGMENTERS        — pre-instantiated prompt-based segmenters
  3. images            — the dataset glob / explicit list
"""

import os
from itertools import product
from pathlib import Path
import warnings
import ctypes

# ---------------------------------------------------------------------------
# Suppress TIFF / libtiff warnings (must precede segmenteer import)
# ---------------------------------------------------------------------------

os.environ.setdefault("LIBTIFF_ERRORS", "0")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
warnings.filterwarnings("ignore", message=".*TIFF.*")
warnings.filterwarnings("ignore", message=".*tags are not sorted.*")

try:
    libc = ctypes.CDLL(None)
    devnull = os.open(os.devnull, os.O_WRONLY)
    original_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
except Exception:
    pass

import segmenteer as seg

try:
    os.dup2(original_stderr, 2)
    os.close(original_stderr)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Reader configuration
# ---------------------------------------------------------------------------

WSI_READER = seg.WSIBackend.OPENSLIDE
os.environ.setdefault("WSI_READER", WSI_READER)
kw = {}  # override backend: from monai.data.wsi_reader import WSIReader; kw = {"reader": WSIReader(WSI_READER)}

# ---------------------------------------------------------------------------
# Stage 1 — Context definers
#
# Each definer reads a downsampled WSI thumbnail and sends it to a locally-
# running Ollama vision model, which returns a short phrase used as the
# segmentation prompt for stage 2.
#
# Required: `ollama serve` must be running and the chosen model pulled:
#   ollama pull llava
#
# OllamaContextDefiner parameters
# --------------------------------
#   model                  Ollama model tag — must be a vision-capable model
#                          (e.g. "llava", "llava-phi3", "moondream")
#   system_prompt          Role / output-format instruction for the LLM
#   user_prompt_template   Message sent per slide; {filename}, {stem}, {mpp} are substituted
#   host                   Ollama server URL  (default: http://localhost:11434)
#   image_mpp              MPP at which to downsample the WSI thumbnail sent to the model;
#                          lower = higher resolution (20 MPP is a good default)
#   options                Dict of Ollama generation options, e.g. {"temperature": 0.1}
#   timeout                HTTP timeout in seconds
# ---------------------------------------------------------------------------

CONTEXT_DEFINERS = [
    seg.OllamaContextDefiner(
        model="gemma4:31b-cloud",
        system_prompt=(
            "You are a pathologist examining a whole-slide image thumbnail. "
            "Your output will be used as a text prompt for a vision segmentation model "
            "(such as FastSAM or CONCH) whose goal is to delineate ALL tissue regions from "
            "background (empty glass) on the slide. "
            "Enumerate every distinct tissue structure and cell type visible — "
            "for example: epidermis, dermis, hair follicles, sebaceous glands, adipose tissue, "
            "blood vessels, stroma, inflammatory infiltrate — whatever is actually present. "
            "The more structures you name, the better the segmentation model can capture them all. "
            "Output ONLY a comma-separated list of the structures present — "
            "no full sentences, no explanation, no preamble."
        ),
        user_prompt_template=(
            "List all tissue structures and cell types visible in this histology slide ({filename})."
        ),
        host="http://localhost:11434",
        image_mpp=10.0,   # MPP of the WSI thumbnail sent to the model
        options={"temperature": 0.1, "num_predict": 256},
    ),

    # --- alternative vision model ---
    # seg.OllamaContextDefiner(
    #     model="llava-phi3",
    #     image_mpp=20.0,
    #     options={"temperature": 0.0, "num_predict": 32},
    # ),
]

# ---------------------------------------------------------------------------
# Stage 2 — Prompt-based segmenters
#
# List pre-instantiated segmenters that support a text prompt.
# Supported classes: CONCHGradCAMSegmenter, FastSAMSegmenter, SAM3Segmenter.
#
# Do NOT set text_prompt / prompt here — the context definer supplies it
# per-image at runtime.
# ---------------------------------------------------------------------------

SEGMENTERS = [
    seg.CONCHGradCAMSegmenter(
        mpp=10,
        activation_threshold=0.2,
    ),

    seg.FastSAMSegmenter(
        model_name="FastSAM-s.pt",
        mpp=10,
        conf=0.4,
        iou=0.9,
    ),

    # seg.SAM3Segmenter(
    #     mpp=10,
    # ),
]

# ---------------------------------------------------------------------------
# Build the pipeline: cross every definer with every segmenter
#
# Each (definer, segmenter) pair becomes a ContextualSegmenter that:
#   1. Calls definer.define(image_path)  → prompt string
#   2. Injects that prompt into the segmenter
#   3. Runs segmenter.segment(image_path)
#
# Output directories follow the usual naming convention:
#   outputs/<timestamp>/<segmenter_name>[<definer_name>]/...
# ---------------------------------------------------------------------------

PIPELINE = [
    seg.ContextualSegmenter(segmenter, definer)
    for definer, segmenter in product(CONTEXT_DEFINERS, SEGMENTERS)
]

# ---------------------------------------------------------------------------
# Context manager to suppress C-level libtiff warnings during processing
# ---------------------------------------------------------------------------


class SuppressLibtiffWarnings:
    def __enter__(self):
        self.original_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        return self

    def __exit__(self, *args):
        os.dup2(self.original_stderr_fd, 2)
        os.close(self.original_stderr_fd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- single image ---
    # seg.run_single_image(PIPELINE, Path("CMU-3.tif"))

    # --- dataset: glob ---
    images = sorted(
        Path("mostwiedzy/dcm2tif/").glob("*.tif")
    )

    # --- dataset: explicit list ---
    # images = [
    #     Path("/path/to/slide1.tif"),
    #     Path("/path/to/slide2.tif"),
    # ]

    with SuppressLibtiffWarnings():
        seg.run_dataset(PIPELINE, images, save_thumbnails=False)

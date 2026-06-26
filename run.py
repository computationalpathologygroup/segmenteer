import os
from pathlib import Path

import segmenteer as seg

# ---------------------------------------------------------------------------
# Reader configuration — change to WSIBackend.CUCIM or WSIBackend.TIFFFILE
# ---------------------------------------------------------------------------

WSI_READER = seg.WSIBackend.OPENSLIDE
os.environ.setdefault("WSI_READER", WSI_READER)
kw = {}  # to override backend: from monai.data.wsi_reader import WSIReader; kw = {"reader": WSIReader(WSI_READER)}

# ---------------------------------------------------------------------------
# Segmenter pool — comment out/remove methods from the list below that you do not want to use or use your own paramaters
# ---------------------------------------------------------------------------

SEGMENTERS = [
    seg.OtsuSegmenter(mpp=5, min_area=0, **kw),
    seg.LiSegmenter(mpp=20, min_area=0, **kw),
    seg.YenSegmenter(mpp=20, min_area=0, **kw),
    seg.EntropyMaskerSegmenter(mpp=10, min_area=0, **kw),
    seg.ODGMMSlideSegmenter(mpp=10, **kw),
    seg.MorphologicalSegmenter(mpp=10, **kw),
    seg.WatershedSegmenter(mpp=10, **kw),
    seg.FESISegmenter(**kw),                     # improved=True, mpp=20
    seg.FESISegmenter(improved=False, **kw),     # different params → separate dir
    seg.BackgroundSubtractorMOG2Segmenter(mpp=20, **kw),
    # seg.HSVThresholdSegmenter(mpp=10, **kw),     # HSV colour-range (H&E purple-pink)
    # seg.HESTSegmenter(mpp=1),
    # seg.GrandQCSegmenter(mpp=8),
    # seg.BigPictureSegmenter(),  # requires: uv pip install tensorflow tissue-segmentation @ git+...
    # seg.TRIDENTHESTSegmenter,
    # seg.TRIDENTGrandQCSegmenter,
    # seg.TRIDENTPathProfilerSegmenter,
    seg.FastSAMSegmenter(mpp=10),
    # seg.RTLucassenSlideSegmenter(),
    # seg.HistomicsTKSegmenter(mpp=10, **kw),
    # seg.TRIDENTCPGSegmenter,
]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- single image, unsupervised ---
    # seg.run_single_image(SEGMENTERS, Path("CMU-3.tif"))

    # --- single image, supervised ---
    # import json
    # gt = json.loads(Path("example.geojson").read_text())
    # seg.run_single_image(SEGMENTERS, Path("example.tiff"), ground_truth=gt)

    # --- dataset, unsupervised ---
    images = sorted(
        Path("/mnt/c/Users/z405155/Downloads/doi-10.34894-zzyu9m/HHG/").glob("*.tiff")
    )
    seg.run_dataset(SEGMENTERS, images)

    # --- dataset, supervised (annotations sit next to images) ---
    # images = sorted(Path("dataset/").glob("*.tiff"))
    # gts = seg.load_ground_truths(images)               # finds <stem>_gt.geojson
    # seg.run_dataset(SEGMENTERS, images, ground_truths=gts)

    # --- dataset, supervised (annotations in a separate folder) ---
    # images = sorted(Path("dataset/wsis/").glob("*.tiff"))
    # gts = seg.load_ground_truths(images, annotation_dir=Path("dataset/annotations/"))
    # seg.run_dataset(SEGMENTERS, images, ground_truths=gts)

    # note: supervised runs have not been yet tested
from pathlib import Path
from typing import Optional

from monai.data.wsi_reader import WSIReader

import segmenteer as seg

# ---------------------------------------------------------------------------
# Reader configuration
# Override here or set the WSI_READER env var (openslide | cucim | tifffile).
# ---------------------------------------------------------------------------
WSI_READER = "openslide"


# ---------------------------------------------------------------------------
# Shared segmenter pool
# ---------------------------------------------------------------------------


def build_segmenters(reader: WSIReader | None = None) -> list:
    if reader is None:
        reader = WSIReader(WSI_READER)
    return [
        # # seg.OtsuSegmenter(mpp=5, min_area=0, reader=reader),
        # # seg.LiSegmenter(mpp=20, min_area=0, reader=reader),
        # # seg.YenSegmenter(mpp=20, min_area=0, reader=reader),
        seg.EntropyMaskerSegmenter(mpp=10, min_area=0, reader=reader),
        # # seg.ODGMMSlideSegmenter(mpp=10, reader=reader),
        # # seg.MorphologicalSegmenter(mpp=10, reader=reader),
        # # seg.WatershedSegmenter(mpp=10, reader=reader),
        # # seg.HistomicsTKSegmenter(mpp=10, reader=reader),
        seg.FESISegmenter(),  # improved=True, mpp=20
        seg.FESISegmenter(improved=False),  # different params → separate dir
        # seg.BackgroundSubtractorMOG2Segmenter(mpp=20, reader=reader),
        seg.HSVThresholdSegmenter(
            mpp=10, reader=reader
        ),  # HSV colour-range (H&E purple-pink)
        # seg.HESTSegmenter(mpp=1),
        # seg.GrandQCSegmenter(mpp=8),
        # seg.RTLucassenSlideSegmenter(),
        # seg.BigPictureSegmenter(),
        # seg.TRIDENTHESTSegmenter,
        # seg.TRIDENTGrandQCSegmenter,
        # seg.TRIDENTPathProfilerSegmenter,
        # seg.FastSAMSegmenter(mpp=10),
        # seg.CPGSegmenter(
        #     docker_image="dodrio1.umcn.nl/daangeijs/tissueseg:latest", min_area=10
        # ),
    ]


# ---------------------------------------------------------------------------
# Single-image benchmark
# ---------------------------------------------------------------------------


def run_single_image(
    path: Path, ground_truth: Optional[dict] = None, reader: WSIReader | None = None
) -> None:
    """Benchmark all methods on one WSI.

    Output layout::

        outputs/<timestamp>/
            ensemble_manifest.json
            original_thumbnail.png
            results.csv / results.json
            <method__params>/              ← one dir per method+hyperparams
                annotations.geojson   ← polygons
                metrics.json          ← unsupervised (+ supervised) scores
                metadata.json         ← timing, image path, run_id
                heatmap.png
    """
    output_dir = seg.create_timestamped_output_dir("outputs")
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    seg.save_thumbnail(path, thumbnails_dir / f"{path.stem}.png")

    segmenters = build_segmenters(reader)
    reporter = seg.BenchmarkReporter()
    writer = seg.EnsembleOutputWriter(output_dir, image_path=path)

    runner = seg.BenchmarkRunner(reporter=reporter, result_callback=writer)
    results = runner.run_multiple(segmenters, path, ground_truth_geojson=ground_truth)

    seg.export_results_csv(results, output_dir / "results.csv")
    seg.export_results_json(results, output_dir / "results.json")
    manifest = writer.finalize(results, image_path=path)

    reporter.print_summary(results)
    reporter.print_saved(
        output_dir,
        [f"thumbnails/{path.stem}.png", "results.csv", "results.json", manifest.name]
        + [f"{r.run_id}/" for r in results],
    )


# ---------------------------------------------------------------------------
# Dataset benchmark
# ---------------------------------------------------------------------------


def run_dataset(
    images: list[Path],
    ground_truths: Optional[dict[Path, dict]] = None,
    reader: WSIReader | None = None,
) -> None:
    """Benchmark all methods across a set of WSIs.

    Output layout::

        outputs/<timestamp>/
            ensemble_manifest.json
            thumbnails/
                <image_stem>.png
            <method__params>/        ← one dir per method+hyperparams
                config.yaml
                predictions/
                    <image_stem>.geojson
                eval/
                    scores/
                        <image_stem>.json
                    heatmaps/
                        <image_stem>.png

    After the run, load members for downstream ensembling::

        members = seg.load_ensemble_members(manifest, image_stem="CMU-1")
        geojsons = [m["geojson"] for m in members]
    """
    output_dir = seg.create_timestamped_output_dir("outputs")
    thumbnails_dir = output_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Save thumbnails per image
    for img in images:
        seg.save_thumbnail(img, thumbnails_dir / f"{img.stem}.png")

    segmenters = build_segmenters(reader)
    reporter = seg.BenchmarkReporter()
    writer = seg.EnsembleOutputWriter(output_dir)

    runner = seg.BenchmarkRunner(reporter=reporter, result_callback=writer)
    all_results = runner.run_dataset(segmenters, images, ground_truths=ground_truths)

    manifest = writer.finalize_dataset(all_results)

    # Print summary for each image
    for img_path, results in all_results.items():
        reporter.print_image_header(img_path, 0, 0)  # section label only
        reporter.print_summary(results)

    reporter.print_saved(output_dir, [manifest.name, "thumbnails/"])


# ---------------------------------------------------------------------------
# Post-run loader (use after any benchmark run)
# ---------------------------------------------------------------------------


def load_for_ensembling(
    manifest_path: Path,
    image_stem: Optional[str] = None,
) -> list[dict]:
    """Load saved members for downstream ensemble fusion.

    Parameters
    ----------
    manifest_path:
        Path to ``ensemble_manifest.json``.
    image_stem:
        For dataset runs: restrict to one image.  ``None`` loads all images.

    Returns a list of member dicts with ``geojson``, ``metrics``, ``metadata``.
    """
    members = seg.load_ensemble_members(manifest_path, image_stem=image_stem)
    for m in members:
        u = m["metrics"]["unsupervised"]
        print(
            f"  {m['run_id']:40s}  "
            f"objects={u['num_objects']:4d}  "
            f"coverage={u['coverage_ratio']:.3f}"
        )
    return members


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # # --- single image, unsupervised ---
    # run_single_image(Path("/Users/agatapolejowska/histopathobiome-s/data/cropped_regions/006_M2.tiff"))

    # --- single image, supervised ---
    # import json
    # gt = json.loads(Path("CMU-3_gt.geojson").read_text())
    # run_single_image(Path("CMU-3.tiff"), ground_truth=gt)

    # --- dataset, unsupervised ---
    # /Users/agatapolejowska/histopathobiome-s/data/cropped_regions
    images = sorted(
        Path("/Users/agatapolejowska/histopathobiome-s/data/cropped_regions").glob(
            "*.tiff"
        )
    )
    run_dataset(images)

    # --- dataset, supervised (annotations sit next to images) ---
    # images = sorted(Path("dataset/").glob("*.tiff"))
    # gts = seg.load_ground_truths(images)               # finds <stem>_gt.geojson
    # run_dataset(images, ground_truths=gts)

    # --- dataset, supervised (annotations in a separate folder) ---
    # images = sorted(Path("dataset/wsis/").glob("*.tiff"))
    # gts = seg.load_ground_truths(images, annotation_dir=Path("dataset/annotations/"))
    # run_dataset(images, ground_truths=gts)

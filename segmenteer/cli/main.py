from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Iterable
from pathlib import Path
import sys


def _parse_segmenter_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _require_segmenter(obj: object, name: str) -> object:
    if obj is None:
        raise ImportError(f"{name} is not available")
    return obj


def _instantiate_segmenters(
    factories: dict[str, Callable[[], object]],
    names: Iterable[str],
) -> tuple[list[object], list[str]]:
    segmenters: list[object] = []
    skipped: list[str] = []

    for name in names:
        factory = factories[name]
        try:
            segmenter = factory()
        except ImportError as exc:
            skipped.append(f"{name} ({exc})")
            continue
        segmenters.append(segmenter)

    return segmenters, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="segmenteer",
        description="Tissue segmentation benchmarking CLI",
    )
    parser.add_argument(
        "--wsi-reader",
        default=None,
        choices=["openslide", "cucim", "tifffile"],
        help="MONAI WSI backend to use (overrides WSI_READER env var)",
    )
    parser.add_argument(
        "--segmenters",
        default="default",
        metavar="LIST",
        help=(
            "Comma-separated list of segmenters to run, or one of: "
            "default, classic, all"
        ),
    )
    parser.add_argument(
        "--list-segmenters",
        action="store_true",
        help="List available segmenter names and exit",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", metavar="PATH", help="Run on a single image")
    target.add_argument(
        "--dataset",
        metavar="DIR",
        help="Run on a directory of images (default: dataset/wsis)",
    )
    parser.add_argument(
        "--pattern",
        default="*.tiff",
        metavar="GLOB",
        help="Glob pattern to match images inside --dataset",
    )
    parser.add_argument(
        "--ground-truth",
        metavar="PATH",
        help="GeoJSON ground-truth file for --image",
    )
    parser.add_argument(
        "--annotations-dir",
        metavar="DIR",
        help="Directory of <stem>_gt.geojson files for dataset runs",
    )
    parser.add_argument(
        "--annotation-suffix",
        default="_gt.geojson",
        metavar="SUFFIX",
        help="Filename suffix for dataset annotations (default: _gt.geojson)",
    )
    parser.add_argument(
        "--mpp",
        default=None,
        help="Override microns-per-pixel (mpp) for all segmenters",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    args = parser.parse_args(argv)

    if args.wsi_reader:
        os.environ["WSI_READER"] = args.wsi_reader

    import segmenteer as seg

    kw: dict = {}

    factories: dict[str, Callable[[], object]] = {
        "otsu": lambda: seg.OtsuSegmenter(mpp=args.mpp, min_area=0, **kw),
        "li": lambda: seg.LiSegmenter(mpp=args.mpp, min_area=0, **kw),
        "yen": lambda: seg.YenSegmenter(mpp=args.mpp, min_area=0, **kw),
        "entropy-masker": lambda: seg.EntropyMaskerSegmenter(mpp=args.mpp, min_area=0, **kw),
        "od-gmm": lambda: seg.ODGMMSlideSegmenter(mpp=args.mpp, **kw),
        "morphological": lambda: seg.MorphologicalSegmenter(mpp=args.mpp, **kw),
        "watershed": lambda: seg.WatershedSegmenter(mpp=args.mpp, **kw),
        "fesi": lambda: seg.FESISegmenter(mpp=args.mpp, **kw),
        "fesi-basic": lambda: seg.FESISegmenter(mpp=args.mpp, improved=False, **kw),
        "background-subtractor": lambda: seg.BackgroundSubtractorMOG2Segmenter(
            mpp=args.mpp, **kw
        ),
        "hsv-threshold": lambda: seg.HSVThresholdSegmenter(mpp=args.mpp, **kw),
        "hest": lambda: seg.HESTSegmenter(mpp=args.mpp),
        "grandqc": lambda: seg.GrandQCSegmenter(mpp=args.mpp),
        "bigpicture": lambda: seg.BigPictureSegmenter(mpp=args.mpp),
        "trident-hest": lambda: _require_segmenter(
            getattr(seg, "TRIDENTHESTSegmenter"), "TRIDENTHESTSegmenter"
        ),  # TODO: add mpp argument to TRIDENTHESTSegmenter
        "trident-grandqc": lambda: _require_segmenter(
            getattr(seg, "TRIDENTGrandQCSegmenter"), "TRIDENTGrandQCSegmenter"
        ),  # TODO: add mpp argument to TRIDENTGrandQCSegmenter
        "trident-pathprofiler": lambda: _require_segmenter(
            getattr(seg, "TRIDENTPathProfilerSegmenter"), "TRIDENTPathProfilerSegmenter"
        ),  # TODO: add mpp argument to TRIDENTPathProfilerSegmenter
        "fastsam": lambda: seg.FastSAMSegmenter(mpp=args.mpp),
        "rtlucassen": lambda: seg.RTLucassenSlideSegmenter(mpp=args.mpp),
        "histomicstk": lambda: seg.HistomicsTKSegmenter(mpp=args.mpp, **kw),
        "trident-cpg": lambda: _require_segmenter(
            getattr(seg, "TRIDENTCPGSegmenter"), "TRIDENTCPGSegmenter"
        ),  # TODO: add mpp argument to TRIDENTCPGSegmenter
    }

    classic_names = [
        "otsu",
        "li",
        "yen",
        "entropy-masker",
        "od-gmm",
        "morphological",
        "watershed",
        "fesi",
        "fesi-basic",
        "background-subtractor",
        "hsv-threshold",
        "histomicstk",
    ]
    default_names = classic_names + [
        "hest",
        "grandqc",
        "bigpicture",
        "trident-hest",
        "trident-grandqc",
        "trident-pathprofiler",
        "fastsam",
        "rtlucassen",
        "trident-cpg",
    ]

    if args.list_segmenters:
        for name in sorted(factories):
            print(name)
        return

    if args.segmenters == "default":
        requested = default_names
    elif args.segmenters == "classic":
        requested = classic_names
    elif args.segmenters == "all":
        requested = list(factories.keys())
    else:
        requested = _parse_segmenter_names(args.segmenters)

    unknown = [name for name in requested if name not in factories]
    if unknown:
        raise SystemExit(f"Unknown segmenters: {', '.join(unknown)}")

    segmenters, skipped = _instantiate_segmenters(factories, requested)
    if not segmenters:
        raise SystemExit("No segmenters available. Install optional dependencies.")
    if skipped:
        print("Skipped segmenters missing optional dependencies:")
        for item in skipped:
            print(f"  - {item}")

    if args.image:
        image_path = Path(args.image)
        ground_truth = None
        if args.ground_truth:
            ground_truth = seg.load_geojson(args.ground_truth)
        seg.run_single_image(segmenters, image_path, ground_truth=ground_truth)
        return

    dataset_dir = Path(args.dataset) if args.dataset else Path("dataset/wsis")
    images = sorted(dataset_dir.glob(args.pattern))
    if not images:
        raise SystemExit(
            f"No images found in {dataset_dir} matching pattern {args.pattern}"
        )

    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else None
    ground_truths = seg.load_ground_truths(
        images, annotation_dir=annotations_dir, suffix=args.annotation_suffix
    )
    seg.run_dataset(segmenters, images, ground_truths=ground_truths or None)


if __name__ == "__main__":
    main()

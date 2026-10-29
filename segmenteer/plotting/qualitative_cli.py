import argparse
from pathlib import Path

from .qualitative import (
    generate_qualitative_figures,
    DEFAULT_GT_ALPHA,
    DEFAULT_TP_ALPHA,
    DEFAULT_FN_ALPHA,
    DEFAULT_FP_ALPHA,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate metric-selected qualitative WSI segmentation comparisons from saved evaluation results"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-methods", type=int, default=3)
    parser.add_argument("--min-method-slides", type=int, default=100)
    parser.add_argument("--min-matched-slides", type=int, default=20)
    parser.add_argument("--max-image-size", type=int, default=1000)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg", "pdf"), default=("png", "svg")
    )
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--gt-opacity", type=float, default=DEFAULT_GT_ALPHA)
    parser.add_argument("--tp-opacity", type=float, default=DEFAULT_TP_ALPHA)
    parser.add_argument("--fn-opacity", type=float, default=DEFAULT_FN_ALPHA)
    parser.add_argument("--fp-opacity", type=float, default=DEFAULT_FP_ALPHA)
    parser.add_argument("--outline-width", type=int, default=1)
    args = parser.parse_args(argv)
    root = args.output_dir or args.metrics.parent / "figures" / "qualitative"
    result = generate_qualitative_figures(
        metrics=args.metrics,
        predictions=args.predictions,
        ground_truth=args.ground_truth,
        data=args.data,
        output_dir=root,
        top_methods=args.top_methods,
        min_method_slides=args.min_method_slides,
        min_matched_slides=args.min_matched_slides,
        max_image_size=args.max_image_size,
        formats=args.formats,
        dpi=args.dpi,
        gt_alpha=args.gt_opacity,
        tp_alpha=args.tp_opacity,
        fn_alpha=args.fn_opacity,
        fp_alpha=args.fp_opacity,
        outline_width=args.outline_width,
    )
    print(f"Qualitative figure: {', '.join(result['figure'])}")
    print(f"Selected cases and method scores: {root}")


if __name__ == "__main__":
    main()

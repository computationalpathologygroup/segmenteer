import argparse
from pathlib import Path

from . import generate_performance_figures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate independent metadata-stratified segmentation performance figures from completed evaluation outputs"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bootstrap-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metric", choices=["nsd", "dice"], default="nsd")
    parser.add_argument("--min-cell-slides", type=int, default=5)
    parser.add_argument("--min-group-slides", type=int, default=5)
    parser.add_argument("--min-method-slides", type=int, default=1)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "svg", "pdf"], default=["png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args(argv)
    default_output = args.metrics.parent / "figures"
    result = generate_performance_figures(
        metrics=args.metrics,
        metadata=args.metadata,
        bootstrap_summary=args.bootstrap_summary,
        output_dir=args.output_dir or default_output,
        metric=args.metric,
        min_cell_slides=args.min_cell_slides,
        min_group_slides=args.min_group_slides,
        min_method_slides=args.min_method_slides,
        matched=args.matched,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
        formats=args.formats,
        dpi=args.dpi,
    )
    print(
        f"Plotting complete: {result['methods']} methods, {result['evaluated_slides']} evaluated slides, {result['missing_metadata_slides']} without metadata"
    )
    for dimension, files in result["figures"].items():
        print(f"{dimension}: {', '.join(files) if files else 'no eligible subgroups'}")
    print(f"Rankings and coverage: {(args.output_dir or default_output).resolve()}")


if __name__ == "__main__":
    main()

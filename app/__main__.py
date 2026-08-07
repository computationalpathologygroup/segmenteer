from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.loader import DEFAULT_HOLE_MODE, DEFAULT_OUTPUT_PREFIX, DEFAULT_RESULTS_DIR, HOLE_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app_official",
        description=(
            "Serve a segmenteer viewer restricted to an official-results cohort. "
            "Only metrics__<hole-mode>__<method>.csv files define the methods/slides exposed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Original output run root containing <method>/predictions directories.",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_RESULTS_DIR,
        metavar="DIR",
        help="Directory containing official metrics CSVs.",
    )
    parser.add_argument(
        "--hole-mode",
        choices=HOLE_MODES,
        default=DEFAULT_HOLE_MODE,
        help="Official metrics and ground-truth interpretation to use.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix in metrics CSV names, e.g. metrics__holes_are_not_tissue__method.csv.",
    )
    parser.add_argument(
        "--ground-truth",
        metavar="DIR",
        help=(
            "GeoJSON ground-truth directory. For holes_are_not_tissue, tissue/hole groups are "
            "subtracted in memory for visualization. Defaults to <output>/ground_truth."
        ),
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        help="Directory containing source WSI files. Required to browse WSI imagery.",
    )
    parser.add_argument(
        "--tissue-group",
        action="append",
        dest="tissue_groups",
        metavar="NAME",
        help="GeoJSON part_of_group to treat as tissue for holes_are_not_tissue. Repeatable.",
    )
    parser.add_argument(
        "--hole-group",
        action="append",
        dest="hole_groups",
        metavar="NAME",
        help="GeoJSON part_of_group to subtract as holes for holes_are_not_tissue. Repeatable.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser()
    metrics = Path(args.metrics).expanduser()
    ground_truth = Path(args.ground_truth).expanduser() if args.ground_truth else None

    if not output.is_dir():
        raise SystemExit(f"--output does not exist or is not a directory: {output}")
    if not metrics.is_dir():
        raise SystemExit(f"--metrics does not exist or is not a directory: {metrics}")
    if args.data and not Path(args.data).expanduser().is_dir():
        print(f"Warning: --data is not mounted or not a directory: {args.data}")
    if ground_truth and not ground_truth.is_dir():
        print(f"Warning: --ground-truth is not mounted or not a directory: {ground_truth}")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise SystemExit("The viewer needs the runtime extra. Run: uv sync --extra runtime") from exc

    from app.server import create_app

    app = create_app(
        output_dir=str(output),
        metrics_dir=str(metrics),
        data_dir=args.data,
        ground_truth_dir=str(ground_truth) if ground_truth else None,
        hole_mode=args.hole_mode,
        output_prefix=args.output_prefix,
        tissue_groups=set(args.tissue_groups or ["tissue"]),
        hole_groups=set(args.hole_groups or ["hole", "holes"]),
    )
    print(f"\n  segmenteer official-results viewer  →  http://{args.host}:{args.port}")
    print(f"  Output run: {output}")
    print(f"  Official metrics: {metrics}")
    print(f"  Hole mode: {args.hole_mode}")
    print("  Only evaluated methods/slides from the selected official CSVs are shown.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

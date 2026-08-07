from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description=(
            "Serve the Segmenteer viewer from runner predictions. Ground truth "
            "and evaluator metrics are optional independent inputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", required=True, metavar="DIR", help="Runner experiment directory containing <method>/predictions/.")
    parser.add_argument("--ground-truth", metavar="DIR", help="Optional GeoJSON ground-truth directory.")
    parser.add_argument("--metrics", metavar="FILE", help="Optional evaluator CSV or SQLite metrics file.")
    parser.add_argument("--data", metavar="DIR", help="Optional source WSI directory for image browsing.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser()
    if not output.is_dir():
        raise SystemExit(f"--output does not exist or is not a directory: {output}")
    if args.metrics and not Path(args.metrics).expanduser().is_file():
        raise SystemExit(f"--metrics does not exist or is not a file: {args.metrics}")
    if args.ground_truth and not Path(args.ground_truth).expanduser().is_dir():
        raise SystemExit(f"--ground-truth does not exist or is not a directory: {args.ground_truth}")
    if args.data and not Path(args.data).expanduser().is_dir():
        print(f"Warning: --data is not mounted or not a directory: {args.data}")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The viewer needs FastAPI/uvicorn runtime dependencies.") from exc

    from app.server import create_app

    app = create_app(
        output_dir=str(output),
        metrics_file=args.metrics,
        data_dir=args.data,
        ground_truth_dir=args.ground_truth,
    )
    print(f"\n  segmenteer viewer  →  http://{args.host}:{args.port}")
    print(f"  Runner output: {output}")
    print(f"  Ground truth: {args.ground_truth or 'not provided'}")
    print(f"  Metrics: {args.metrics or 'not provided'}")
    print("  Viewer build: 20260807_18 (static/API cache disabled)")
    index_data = getattr(app.state, "index_data", None)
    if args.metrics and index_data is not None:
        metric_entries = sum(
            1
            for per_slide in index_data.scores.values()
            for entry in per_slide.values()
            if isinstance(entry, dict) and entry.get("metrics")
        )
        metric_names = ", ".join(index_data.available_metrics) or "none"
        print(f"  Metrics matched: {metric_entries} slide/method row(s); fields: {metric_names}")
    print()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

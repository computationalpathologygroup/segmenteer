"""CLI entry point.

Usage
-----
    python -m app --output outputs/1903_1622
    python -m app --output outputs/1903_1622 --data /path/to/tiffs
    python -m app --output outputs/1903_1622 --port 8765

Dependencies
------------
Install the [app] extra:

    uv pip install -e ".[app]"
    # or: pip install fastapi "uvicorn[standard]"
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="segmenteer results viewer",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Path to a segmenteer output directory (e.g. outputs/1903_1622)",
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="DIR",
        help="Optional: directory containing the source TIFF/WSI files.  "
        "Only needed when the stored image_path is no longer valid.",
    )
    parser.add_argument("--host", default="127.0.0.1", metavar="HOST")
    parser.add_argument("--port", type=int, default=8765, metavar="PORT")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is required.  Run: pip install 'uvicorn[standard]'")

    from app.server import create_app

    app = create_app(output_dir=args.output, data_dir=args.data)

    print(f"\n  segmenteer viewer  →  http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

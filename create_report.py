"""Standalone script to generate a self-contained HTML segmentation report.

This produces the same output as clicking "Download segmentation report" in the
app UI — no running server required.

Usage
-----
    python create_report.py --output outputs/1903_1622
    python create_report.py --output outputs/1903_1622 --data /path/to/tiffs
    python create_report.py --output outputs/1903_1622 --out report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_older.loader import load_index
from app_older.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML segmentation report.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="segmenteer output directory (e.g. outputs/1903_1622)",
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="DIR",
        help="Optional directory containing the source WSI files (only needed "
        "when the stored image_path is no longer valid).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Path to write the HTML report to.  "
        "Defaults to segmenteer_report_<output-dir-name>.html in the current directory.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    data_path = Path(args.data).resolve() if args.data else None

    if not output_path.exists():
        parser.error(f"Output directory not found: {output_path}")

    out_file = Path(args.out) if args.out else Path(f"segmenteer_report_{output_path.name}.html")

    print(f"Loading index from {output_path} …")
    index = load_index(output_path, data_path)

    print(f"Generating report for {len(index.wsis)} slide(s) × {len(index.methods)} method(s) …")
    html_content = generate_report(index, output_path)

    out_file.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_file.resolve()}")


if __name__ == "__main__":
    main()

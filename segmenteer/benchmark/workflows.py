"""High-level benchmark workflows.

Provides ``run_single_image`` and ``run_dataset`` so that user scripts only
need to supply a segmenter list and a data path — everything else is handled
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from segmenteer.benchmark.console import BenchmarkReporter
from segmenteer.benchmark.ensemble import (EnsembleOutputWriter,
                                           load_ensemble_members)
from segmenteer.benchmark.reporting import (export_results_csv,
                                            export_results_json)
from segmenteer.benchmark.runner import BenchmarkRunner
from segmenteer.io import create_timestamped_output_dir
from segmenteer.visualization import save_thumbnail


def run_single_image(
    segmenters: list,
    path: Path,
    ground_truth: Optional[dict] = None,
    save_thumbnails: bool = True,
) -> None:
    """Benchmark *segmenters* on a single WSI.

    Output layout::

        outputs/<timestamp>/
            ensemble_manifest.json
            thumbnails/<stem>.png       (if save_thumbnails=True)
            results.csv / results.json
            <method__params>/
                predictions/<stem>.geojson
                eval/scores/<stem>.json
                eval/heatmaps/<stem>.png

    Parameters
    ----------
    segmenters : list
        List of segmenter instances to benchmark.
    path : Path
        Path to the WSI file.
    ground_truth : dict, optional
        GeoJSON dict with ground truth annotations.
    save_thumbnails : bool, default=True
        Whether to save image thumbnails to the output directory.
    """
    path = Path(path)
    output_dir = create_timestamped_output_dir("outputs")

    if save_thumbnails:
        thumbnails_dir = output_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        save_thumbnail(path, thumbnails_dir / f"{path.stem}.png")

    reporter = BenchmarkReporter()
    writer = EnsembleOutputWriter(output_dir, image_path=path)
    runner = BenchmarkRunner(reporter=reporter, result_callback=writer)

    results = runner.run_multiple(segmenters, path, ground_truth_geojson=ground_truth)

    export_results_csv(results, output_dir / "results.csv")
    export_results_json(results, output_dir / "results.json")
    manifest = writer.finalize(results, image_path=path)

    reporter.print_summary(results)
    reporter.print_saved(
        output_dir,
        (
            [
                f"thumbnails/{path.stem}.png",
                "results.csv",
                "results.json",
                manifest.name,
            ]
            if save_thumbnails
            else ["results.csv", "results.json", manifest.name]
        )
        + [f"{r.run_id}/" for r in results],
    )


def run_dataset(
    segmenters: list,
    images: list[Path],
    ground_truths: Optional[dict[Path, dict]] = None,
    save_thumbnails: bool = True,
) -> None:
    """Benchmark *segmenters* across a set of WSIs.

    Output layout::

        outputs/<timestamp>/
            ensemble_manifest.json
            thumbnails/<stem>.png       (if save_thumbnails=True)
            <method__params>/
                predictions/<stem>.geojson
                eval/scores/<stem>.json
                eval/heatmaps/<stem>.png

    Parameters
    ----------
    segmenters : list
        List of segmenter instances to benchmark.
    images : list[Path]
        List of paths to WSI files.
    ground_truths : dict[Path, dict], optional
        Mapping of image paths to ground truth GeoJSON dicts.
    save_thumbnails : bool, default=True
        Whether to save image thumbnails to the output directory.
    """
    output_dir = create_timestamped_output_dir("outputs")

    if save_thumbnails:
        thumbnails_dir = output_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            save_thumbnail(img, thumbnails_dir / f"{img.stem}.png")

    reporter = BenchmarkReporter()
    writer = EnsembleOutputWriter(output_dir)
    runner = BenchmarkRunner(reporter=reporter, result_callback=writer)

    all_results = runner.run_dataset(segmenters, images, ground_truths=ground_truths)
    manifest = writer.finalize_dataset(all_results)

    for img_path, results in all_results.items():
        reporter.print_image_header(img_path, 0, 0)
        reporter.print_summary(results)

    reporter.print_saved(
        output_dir,
        [manifest.name] + (["thumbnails/"] if save_thumbnails else []),
    )

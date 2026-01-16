from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
import time
import numpy as np
from segmenteer.core.base import Segmenter
from skimage.transform import resize
from segmenteer.metrics.evaluation import (
    SupervisedMetrics,
    compute_all_supervised_metrics,
)
from segmenteer.metrics.unsupervised import (
    UnsupervisedMetrics,
    compute_unsupervised_metrics,
)
from segmenteer.core.utils import mask_to_geojson
import pyvips


@dataclass
class BenchmarkResult:
    method_name: str
    geojson: dict
    execution_time: float
    seconds_per_pixel: float
    unsupervised_metrics: UnsupervisedMetrics
    supervised_metrics: Optional[SupervisedMetrics] = None


class BenchmarkRunner:
    def __init__(self, verbose: bool = True, result_callback: Optional[Callable[[BenchmarkResult], None]] = None):
        self.results = []
        self.verbose = verbose
        self.result_callback = result_callback

    def _log(self, message: str, end: str = "\n"):
        if self.verbose:
            print(message, end=end, flush=True)

    def run_single(
        self,
        segmenter: Segmenter,
        image_path: Path | None = None,
        ground_truth_geojson: Optional[dict] = None,
    ) -> BenchmarkResult:
        self._log(f"Running {segmenter.name}...")

        start_time = time.perf_counter()
        result = segmenter.segment(image_path)
        execution_time = time.perf_counter() - start_time

        self._log(f"  Segmentation completed in {execution_time:.4f}s")

        image = pyvips.Image.tiffload(image_path, access="sequential", page=0, n=1)

        image_shape = (image.height, image.width)
        num_pixels = image_shape[0] * image_shape[1]
        seconds_per_pixel = execution_time / num_pixels

        self._log(f"  Computing unsupervised metrics...")
        image_area = num_pixels
        unsupervised = compute_unsupervised_metrics(result, image_area)
        self._log(f"  Found {unsupervised.num_objects} objects")

        supervised = None
        if ground_truth_geojson is not None:
            self._log(f"  Computing supervised metrics...")
            supervised = compute_all_supervised_metrics(
                result, ground_truth_geojson, image_shape
            )
            self._log(f"  Dice: {supervised.dice:.4f}, IoU: {supervised.iou:.4f}")

        result = BenchmarkResult(
            method_name=segmenter.name,
            geojson=result,
            execution_time=execution_time,
            seconds_per_pixel=seconds_per_pixel,
            unsupervised_metrics=unsupervised,
            supervised_metrics=supervised,
        )

        self.results.append(result)
        self._log(f"  ✓ {segmenter.name} complete\n")
        
        if self.result_callback:
            self.result_callback(result)
        
        return result

    def run_multiple(
        self,
        segmenters: list[Segmenter],
        image_path: Path,
        ground_truth_geojson: Optional[dict] = None,
    ) -> list[BenchmarkResult]:
        self._log(
            f"\nBenchmarking {len(segmenters)} methods on image {image_path}"
        )
        self._log("=" * 60 + "\n")

        results = []
        for i, segmenter in enumerate(segmenters, 1):
            self._log(f"[{i}/{len(segmenters)}] ", end="")
            result = self.run_single(segmenter, image_path, ground_truth_geojson)
            results.append(result)

        self._log("=" * 60)
        self._log(f"All {len(segmenters)} methods completed!\n")
        return results

    def clear_results(self):
        self.results = []

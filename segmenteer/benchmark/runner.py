from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
from segmenteer.core.base import Segmenter
from segmenteer.metrics.evaluation import (
    SupervisedMetrics,
    compute_all_supervised_metrics,
)
from segmenteer.metrics.unsupervised import (
    UnsupervisedMetrics,
    compute_unsupervised_metrics,
)


@dataclass
class BenchmarkResult:
    method_name: str
    geojson: dict
    execution_time: float
    seconds_per_pixel: float
    unsupervised_metrics: UnsupervisedMetrics
    supervised_metrics: Optional[SupervisedMetrics] = None


class BenchmarkRunner:
    def __init__(self, verbose: bool = True):
        self.results = []
        self.verbose = verbose

    def _log(self, message: str, end: str = "\n"):
        if self.verbose:
            print(message, end=end, flush=True)

    def run_single(
        self,
        segmenter: Segmenter,
        image: np.ndarray,
        ground_truth_geojson: Optional[dict] = None,
    ) -> BenchmarkResult:
        self._log(f"Running {segmenter.name}...")

        start_time = time.perf_counter()
        geojson_result = segmenter.segment(image)
        execution_time = time.perf_counter() - start_time

        self._log(f"  Segmentation completed in {execution_time:.4f}s")

        num_pixels = image.shape[0] * image.shape[1]
        seconds_per_pixel = execution_time / num_pixels

        self._log(f"  Computing unsupervised metrics...")
        image_area = float(image.shape[0] * image.shape[1])
        unsupervised = compute_unsupervised_metrics(geojson_result, image_area)
        self._log(f"  Found {unsupervised.num_objects} objects")

        supervised = None
        if ground_truth_geojson is not None:
            self._log(f"  Computing supervised metrics...")
            supervised = compute_all_supervised_metrics(
                geojson_result, ground_truth_geojson
            )
            self._log(f"  Dice: {supervised.dice:.4f}, IoU: {supervised.iou:.4f}")

        result = BenchmarkResult(
            method_name=segmenter.name,
            geojson=geojson_result,
            execution_time=execution_time,
            seconds_per_pixel=seconds_per_pixel,
            unsupervised_metrics=unsupervised,
            supervised_metrics=supervised,
        )

        self.results.append(result)
        self._log(f"  ✓ {segmenter.name} complete\n")
        return result

    def run_multiple(
        self,
        segmenters: list[Segmenter],
        image: np.ndarray,
        ground_truth_geojson: Optional[dict] = None,
    ) -> list[BenchmarkResult]:
        self._log(
            f"\nBenchmarking {len(segmenters)} methods on {image.shape[0]}x{image.shape[1]} image"
        )
        self._log("=" * 60 + "\n")

        results = []
        for i, segmenter in enumerate(segmenters, 1):
            self._log(f"[{i}/{len(segmenters)}] ", end="")
            result = self.run_single(segmenter, image, ground_truth_geojson)
            results.append(result)

        self._log("=" * 60)
        self._log(f"All {len(segmenters)} methods completed!\n")
        return results

    def clear_results(self):
        self.results = []

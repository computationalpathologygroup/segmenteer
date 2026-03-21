from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from segmenteer.core.base import Segmenter, segmenter_config_dict
from segmenteer.metrics.supervised import (SupervisedMetrics,
                                           compute_all_supervised_metrics)
from segmenteer.metrics.unsupervised import (UnsupervisedMetrics,
                                             compute_unsupervised_metrics)

if TYPE_CHECKING:
    from segmenteer.benchmark.console import BenchmarkReporter


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of one segmentation run on one image.

    ``run_id`` is a filesystem-safe unique slug that encodes the method name
    *and* its hyperparameters, so two instances of the same class with
    different params are always stored in separate directories.
    ``image_path`` is set when results come from a dataset run, allowing the
    writer to namespace outputs per image.
    """

    method_name: str
    run_id: str
    geojson: dict
    execution_time: float
    seconds_per_pixel: float
    unsupervised_metrics: UnsupervisedMetrics
    image_path: Optional[Path] = None
    supervised_metrics: Optional[SupervisedMetrics] = None
    segmenter_config: Optional[dict] = None
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


# ---------------------------------------------------------------------------
# run_id helpers
# ---------------------------------------------------------------------------

# Attrs that carry no useful hyperparameter information
_SKIP_ATTRS = frozenset({"reader", "to_gray_func", "results"})
# Characters that are invalid or ugly in directory names
_SLUG_RE = re.compile(r"[^\w\-.]")


def _slug(value: object) -> str:
    """Convert a parameter value to a compact filesystem-safe string."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # "20.0" → "20", "0.5" → "0.5"
        return f"{value:g}"
    raw = str(value)
    return _SLUG_RE.sub("", raw)[:24]  # cap length to keep names sane


def make_run_id(segmenter: Segmenter, used_ids: set[str]) -> str:
    """Return a unique, human-readable id for *segmenter* within this run.

    The id is ``<name>`` when the segmenter has no public hyperparameters, or
    ``<name>__<param>=<val>_<param>=<val>…`` otherwise (underscores in param
    names are replaced with hyphens).  If the same string was already produced
    for a different segmenter in this run it is suffixed with ``_2``, ``_3``,
    … until unique.
    """
    base = segmenter.name

    params = {
        k: v
        for k, v in vars(segmenter).items()
        if not k.startswith("_") and k not in _SKIP_ATTRS and not callable(v)
    }
    if params:
        param_str = "_".join(
            f"{k.replace('_', '-')}={_slug(v)}" for k, v in sorted(params.items())
        )
        candidate = f"{base}__{param_str}"
    else:
        candidate = base

    original = candidate
    counter = 2
    while candidate in used_ids:
        candidate = f"{original}_{counter}"
        counter += 1

    used_ids.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    def __init__(
        self,
        verbose: bool = True,
        result_callback: Optional[Callable[[BenchmarkResult], None]] = None,
        reporter: Optional[BenchmarkReporter] = None,
    ) -> None:
        self.results: list[BenchmarkResult] = []
        self.verbose = verbose
        self.result_callback = result_callback
        self.reporter = reporter

    def _log(self, message: str, end: str = "\n") -> None:
        """Plain-text fallback logger used when no reporter is attached."""
        if self.verbose and self.reporter is None:
            print(message, end=end, flush=True)

    # ------------------------------------------------------------------
    # Single image, single segmenter
    # ------------------------------------------------------------------

    def run_single(
        self,
        segmenter: Segmenter,
        image: Path,
        ground_truth_geojson: Optional[dict] = None,
        *,
        run_id: Optional[str] = None,
        image_path: Optional[Path] = None,
        _index: int = 0,
        _total: int = 0,
    ) -> BenchmarkResult:
        """Segment *image* with *segmenter* and return a :class:`BenchmarkResult`.

        Parameters
        ----------
        run_id:
            Pre-computed unique id for this segmenter.  When ``None`` a fresh
            one is generated (useful when calling ``run_single`` directly).
        image_path:
            When called from a dataset run, pass the image path here so it is
            set on the result *before* the result_callback fires.
        """
        if run_id is None:
            run_id = make_run_id(segmenter, set())

        display = run_id  # human-readable label for logs

        if self.reporter and _index and _total:
            self.reporter.print_method_start(display, _index, _total)
        else:
            self._log(f"Running {display}...")

        start_time = time.perf_counter()
        try:
            geojson_result = segmenter.segment(image)
            execution_time = time.perf_counter() - start_time

            self._log(f"  Segmentation completed in {execution_time:.4f}s")

            # TODO: this doesn't look clean and is repeated elsewhere.
            from monai.data.wsi_reader import WSIReader

            from segmenteer.core.base import WSI_READER

            reader = WSIReader(WSI_READER)
            wsi = reader.read(str(image))

            shape = reader.get_size(wsi, 0)
            area = shape[0] * shape[1]
            seconds_per_pixel = execution_time / area

            self._log("  Computing unsupervised metrics...")
            unsupervised = compute_unsupervised_metrics(geojson_result, area)
            self._log(f"  Found {unsupervised.num_objects} objects")

            supervised = None
            if ground_truth_geojson is not None:
                self._log("  Computing supervised metrics...")
                supervised = compute_all_supervised_metrics(
                    geojson_result, ground_truth_geojson, shape
                )
                self._log(f"  Dice: {supervised.dice:.4f}, IoU: {supervised.iou:.4f}")

            result = BenchmarkResult(
                method_name=segmenter.name,
                run_id=run_id,
                geojson=geojson_result,
                execution_time=execution_time,
                seconds_per_pixel=seconds_per_pixel,
                unsupervised_metrics=unsupervised,
                supervised_metrics=supervised,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path,
            )

        except Exception as exc:  # noqa: BLE001
            execution_time = time.perf_counter() - start_time
            error_msg = f"{type(exc).__name__}: {exc}"
            self._log(f"  ✗ FAILED after {execution_time:.2f}s — {error_msg}")
            if self.reporter:
                self.reporter.print_method_failed(display, error_msg)
            result = BenchmarkResult(
                method_name=segmenter.name,
                run_id=run_id,
                geojson={"type": "FeatureCollection", "features": []},
                execution_time=execution_time,
                seconds_per_pixel=0.0,
                unsupervised_metrics=compute_unsupervised_metrics(
                    {"type": "FeatureCollection", "features": []}, 1
                ),
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path,
                error=error_msg,
            )
            self.results.append(result)
            if self.result_callback:
                self.result_callback(result)
            return result

        self.results.append(result)

        if self.reporter:
            self.reporter.print_method_done(result)
        else:
            self._log(f"  ✓ {display} complete\n")

        if self.result_callback:
            self.result_callback(result)

        return result

    # ------------------------------------------------------------------
    # Single image, many segmenters
    # ------------------------------------------------------------------

    def run_multiple(
        self,
        segmenters: list[Segmenter],
        image: Path,
        ground_truth_geojson: Optional[dict] = None,
    ) -> list[BenchmarkResult]:
        """Run all *segmenters* on a single *image*."""
        if self.reporter:
            self.reporter.print_header(len(segmenters), image)
        else:
            self._log(f"\nBenchmarking {len(segmenters)} methods on image {image}")
            self._log("=" * 60 + "\n")

        # Build run_ids once so uniqueness is guaranteed across the full list
        used_ids: set[str] = set()
        run_ids = [make_run_id(s, used_ids) for s in segmenters]

        results = []
        for i, (segmenter, run_id) in enumerate(zip(segmenters, run_ids), 1):
            if not self.reporter:
                self._log(f"[{i}/{len(segmenters)}] ", end="")
            result = self.run_single(
                segmenter,
                image,
                ground_truth_geojson,
                run_id=run_id,
                _index=i,
                _total=len(segmenters),
            )
            results.append(result)

        if self.reporter:
            n_failed = sum(1 for r in results if r.failed)
            self.reporter.print_run_complete(len(segmenters), n_failed=n_failed)
        else:
            self._log("=" * 60)
            self._log(f"All {len(segmenters)} methods completed!\n")

        return results

    # ------------------------------------------------------------------
    # Dataset: many images, many segmenters
    # ------------------------------------------------------------------

    def run_dataset(
        self,
        segmenters: list[Segmenter],
        images: list[Path],
        ground_truths: Optional[dict[Path, dict]] = None,
    ) -> dict[Path, list[BenchmarkResult]]:
        """Run all *segmenters* on every image in *images*.

        Parameters
        ----------
        segmenters:
            List of segmenter instances to benchmark.
        images:
            List of WSI paths — the dataset.
        ground_truths:
            Optional mapping of ``image_path → geojson dict``.  When an image
            has a matching entry supervised metrics are computed for it.

        Returns
        -------
        dict mapping each image path to its list of :class:`BenchmarkResult`.
        All results also carry ``image_path`` set for downstream writers.
        """
        ground_truths = ground_truths or {}

        if self.reporter:
            self.reporter.print_dataset_header(len(segmenters), len(images))
        else:
            self._log(
                f"\nDataset benchmark: {len(images)} images × {len(segmenters)} methods"
            )
            self._log("=" * 60 + "\n")

        # run_ids are shared across the whole dataset so member dirs are
        # consistent and comparable across images.
        used_ids: set[str] = set()
        run_ids = [make_run_id(s, used_ids) for s in segmenters]

        all_results: dict[Path, list[BenchmarkResult]] = {}

        for img_idx, image in enumerate(images, 1):
            gt = ground_truths.get(image)

            if self.reporter:
                self.reporter.print_image_header(image, img_idx, len(images))
            else:
                self._log(f"\n[Image {img_idx}/{len(images)}] {image.name}")
                self._log("-" * 60)

            img_results = []
            for i, (segmenter, run_id) in enumerate(zip(segmenters, run_ids), 1):
                if not self.reporter:
                    self._log(f"  [{i}/{len(segmenters)}] ", end="")
                result = self.run_single(
                    segmenter,
                    image,
                    gt,
                    run_id=run_id,
                    image_path=image,
                    _index=i,
                    _total=len(segmenters),
                )
                img_results.append(result)

            all_results[image] = img_results

            if self.reporter:
                n_failed = sum(1 for r in img_results if r.failed)
                self.reporter.print_run_complete(len(segmenters), n_failed=n_failed)
            else:
                self._log(f"  ✓ {image.name} done\n")

        return all_results

    # ------------------------------------------------------------------

    def clear_results(self) -> None:
        self.results = []

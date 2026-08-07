from __future__ import annotations

import math
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from segmenteer.benchmark.resume import fingerprint
from segmenteer.core.base import Segmenter, get_wsi_reader, segmenter_config_dict
if TYPE_CHECKING:
    from segmenteer.benchmark.console import BenchmarkReporter
    from segmenteer.metrics.supervised import SupervisedMetrics
    from segmenteer.metrics.unsupervised import UnsupervisedMetrics


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
    image_path: Optional[Path] = None
    # Metric calculation is intentionally deferred to ``evaluate_outputs.py``.
    # These optional fields exist only to retain API compatibility with older
    # completed artifacts and callers that supply precomputed values.
    unsupervised_metrics: Optional["UnsupervisedMetrics"] = None
    supervised_metrics: Optional["SupervisedMetrics"] = None
    # Preserved for output writers so the viewer can show the exact same GT
    # geometry used for the scalar supervised metrics.
    ground_truth_geojson: Optional[dict] = None
    segmenter_config: Optional[dict] = None
    # Level-0 physical pixel size in µm/px, captured from the reader used for
    # benchmark accounting. ``None`` means the input metadata was unavailable.
    native_spacing: Optional[tuple[float, float]] = None
    # Model target magnification when declared (for example ``"4x"`` for a
    # Trident adapter), otherwise the configured inference spacing such as
    # ``"10 µm/px"``. It is a report label, never a fabricated scan objective.
    magnification_used: Optional[str] = None
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


# ---------------------------------------------------------------------------
# run_id helpers
# ---------------------------------------------------------------------------

# Attrs that carry no useful hyperparameter information
_SKIP_ATTRS = frozenset({"reader", "to_gray_func", "results", "segmenter", "segmenter_func"})
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


def make_run_id(segmenter: Segmenter, used_ids: set[str] | None = None) -> str:
    """Return a deterministic, filesystem-safe directory id for *segmenter*.

    The readable method/parameter slug remains convenient for humans.  A stable
    configuration fingerprint is appended so independently launched workers
    cannot map distinct configurations onto the same method directory merely
    because two parameter values stringify or truncate to the same slug.

    ``used_ids`` keeps the historical support for duplicate segmenter entries
    in one invocation; it is optional for callers that only need the canonical
    cross-process ID.
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
        readable = f"{base}__{param_str}"
    else:
        readable = base

    config_digest = fingerprint(segmenter_config_dict(segmenter))[:12]
    candidate = f"{readable}__cfg={config_digest}"

    if used_ids is None:
        return candidate

    original = candidate
    counter = 2
    while candidate in used_ids:
        candidate = f"{original}_{counter}"
        counter += 1

    used_ids.add(candidate)
    return candidate


def make_run_ids(segmenters: list[Segmenter]) -> list[str]:
    """Build the stable run IDs shared by an entire dataset worker."""
    used_ids: set[str] = set()
    return [make_run_id(segmenter, used_ids) for segmenter in segmenters]


# ---------------------------------------------------------------------------
# Result metadata helpers
# ---------------------------------------------------------------------------


def describe_magnification_used(segmenter: Segmenter) -> str | None:
    """Describe the inference scale without mislabelling MPP as objective power.

    Trident exposes an explicit target objective magnification. Most other
    built-in segmenters are parameterised by micrometres per pixel, which is
    recorded verbatim because it is the actual request made to the WSI reader.
    AtlasPatch intentionally operates on its service thumbnail and does not
    expose a fixed objective magnification.
    """
    target_mag = getattr(segmenter, "target_mag", None)
    if isinstance(target_mag, (int, float)) and not isinstance(target_mag, bool):
        target_mag = float(target_mag)
        if math.isfinite(target_mag) and target_mag > 0:
            return f"{target_mag:g}x"

    mpp = getattr(segmenter, "mpp", None)
    if isinstance(mpp, (int, float)) and not isinstance(mpp, bool):
        mpp = float(mpp)
        if math.isfinite(mpp) and mpp > 0:
            return f"{mpp:g} µm/px"

    if getattr(segmenter, "name", "") == "atlaspatch_sam2":
        return "AtlasPatch service thumbnail"
    return None


def _native_spacing_or_none(reader, wsi) -> tuple[float, float] | None:
    """Read level-0 MPP for reporting without making a completed run fail."""
    try:
        x_mpp, y_mpp = reader.get_mpp(wsi, 0)
        x_mpp, y_mpp = float(x_mpp), float(y_mpp)
    except Exception:  # metadata is optional reporting information
        return None
    if not all(math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp)):
        return None
    return x_mpp, y_mpp


def _native_spacing_from_score(score: dict) -> tuple[float, float] | None:
    """Restore persisted spacing metadata without reopening a resumed WSI."""
    raw = score.get("native_spacing_um_per_px")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        x_mpp, y_mpp = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp)):
        return None
    return x_mpp, y_mpp


def _magnification_used_from_config(config: dict) -> str | None:
    """Recover a report label for legacy resume artifacts from saved params."""
    params = config.get("params") if isinstance(config, dict) else None
    if not isinstance(params, dict):
        return None
    target_mag = params.get("target_mag")
    if isinstance(target_mag, (int, float)) and not isinstance(target_mag, bool):
        target_mag = float(target_mag)
        if math.isfinite(target_mag) and target_mag > 0:
            return f"{target_mag:g}x"
    mpp = params.get("mpp")
    if isinstance(mpp, (int, float)) and not isinstance(mpp, bool):
        mpp = float(mpp)
        if math.isfinite(mpp) and mpp > 0:
            return f"{mpp:g} µm/px"
    class_path = str(config.get("class", ""))
    if class_path.endswith("AtlasPatchSAM2Segmenter"):
        return "AtlasPatch service thumbnail"
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    def __init__(
        self,
        verbose: bool = True,
        result_callback: Optional[Callable[[BenchmarkResult], None]] = None,
        reporter: Optional[BenchmarkReporter] = None,
        compute_pixel_metrics: bool = False,
    ) -> None:
        self.results: list[BenchmarkResult] = []
        self.verbose = verbose
        self.result_callback = result_callback
        self.reporter = reporter
        # Retained as a harmless compatibility argument. Pixel metrics are now
        # deliberately computed only by the explicit post-run evaluator.
        self.compute_pixel_metrics = compute_pixel_metrics

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
            run_id = make_run_id(segmenter)

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

            # Reopen only enough of the WSI to obtain level-0 dimensions and
            # physical spacing.  This does not rasterise prediction/GT polygons
            # or calculate quality metrics; evaluation is a separate post-run
            # step in ``evaluate_outputs.py``.
            reader = getattr(segmenter, "reader", None) or get_wsi_reader()
            wsi = reader.read(str(image))
            width, height = reader.get_size(wsi, 0)
            area = width * height
            native_spacing = _native_spacing_or_none(reader, wsi)
            magnification_used = describe_magnification_used(segmenter)
            seconds_per_pixel = execution_time / area if area else 0.0

            result = BenchmarkResult(
                method_name=getattr(segmenter, "display_name", segmenter.name),
                run_id=run_id,
                geojson=geojson_result,
                execution_time=execution_time,
                seconds_per_pixel=seconds_per_pixel,
                ground_truth_geojson=ground_truth_geojson,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path,
                native_spacing=native_spacing,
                magnification_used=magnification_used,
            )

        except Exception as exc:  # noqa: BLE001
            execution_time = time.perf_counter() - start_time
            full_tb = traceback.format_exc()
            error_msg = f"{type(exc).__name__}: {exc}\n\n{full_tb}"
            self._log(f"  ✗ FAILED after {execution_time:.2f}s — {type(exc).__name__}: {exc}")
            if self.reporter:
                self.reporter.print_method_failed(display, f"{type(exc).__name__}: {exc}")
            result = BenchmarkResult(
                method_name=getattr(segmenter, "display_name", segmenter.name),
                run_id=run_id,
                geojson={"type": "FeatureCollection", "features": []},
                execution_time=execution_time,
                seconds_per_pixel=0.0,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path,
                magnification_used=describe_magnification_used(segmenter),
                ground_truth_geojson=ground_truth_geojson,
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

        # Build deterministic run IDs once for the full method list.
        run_ids = make_run_ids(segmenters)

        results = []
        for i, (segmenter, run_id) in enumerate(zip(segmenters, run_ids), 1):
            if not self.reporter:
                self._log(f"[{i}/{len(segmenters)}] ", end="")
            result = self.run_single(
                segmenter,
                image,
                ground_truth_geojson,
                run_id=run_id,
                image_path=image,
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

    @staticmethod
    def _result_from_resumed_artifact(
        artifact,
        *,
        image_path: Path,
        ground_truth_geojson: Optional[dict],
        expected_config: dict,
    ) -> BenchmarkResult | None:
        """Rehydrate a validated result without requiring stored metrics.

        Older output folders may contain metric payloads.  They are restored
        when valid, but new lightweight runs persist only prediction metadata.
        """
        score = artifact.score
        try:
            unsupervised = None
            supervised = None
            metrics = score.get("metrics")
            if isinstance(metrics, dict):
                unsupervised_payload = metrics.get("unsupervised")
                if isinstance(unsupervised_payload, dict):
                    from segmenteer.metrics.unsupervised import UnsupervisedMetrics

                    unsupervised = UnsupervisedMetrics(**unsupervised_payload)
                supervised_payload = metrics.get("supervised")
                if isinstance(supervised_payload, dict):
                    from segmenteer.metrics.supervised import SupervisedMetrics

                    supervised_payload = dict(supervised_payload)
                    # JSON sanitisation represents an infinite Hausdorff
                    # distance as null; restore the in-memory representation.
                    if supervised_payload.get("hausdorff") is None:
                        supervised_payload["hausdorff"] = float("inf")
                    supervised = SupervisedMetrics(**supervised_payload)

            return BenchmarkResult(
                method_name=str(score["method_name"]),
                run_id=str(score["run_id"]),
                geojson=artifact.geojson,
                execution_time=float(score.get("execution_time_s", 0.0)),
                seconds_per_pixel=float(score.get("seconds_per_pixel", 0.0)),
                unsupervised_metrics=unsupervised,
                supervised_metrics=supervised,
                ground_truth_geojson=ground_truth_geojson,
                segmenter_config=expected_config,
                image_path=image_path,
                native_spacing=_native_spacing_from_score(score),
                magnification_used=(
                    score.get("magnification_used")
                    or _magnification_used_from_config(expected_config)
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Dataset: many images, many segmenters
    # ------------------------------------------------------------------

    def run_dataset(
        self,
        segmenters: list[Segmenter],
        images: list[Path],
        ground_truths: Optional[dict[Path, dict]] = None,
        *,
        resume_store=None,
        run_ids: list[str] | None = None,
    ) -> dict[Path, list[BenchmarkResult]]:
        """Run all *segmenters* on every image in *images*.

        When ``resume_store`` is supplied, each slide/method pair is reused
        only after its persisted prediction, score metadata, and configuration
        validate against the current invocation.  Failed, partial, corrupt, or
        incompatible outputs are recomputed.

        Parameters
        ----------
        segmenters:
            List of segmenter instances to benchmark.
        images:
            List of WSI paths — the dataset.
        ground_truths:
            Optional mapping of ``image_path → geojson dict``. When an image
            has a matching entry, the label is persisted for later post-run
            evaluation; no metrics are computed during this benchmark call.
        resume_store:
            Internal resume validator created by :func:`run_dataset` in
            :mod:`segmenteer.benchmark.workflows`.

        Returns
        -------
        dict mapping each image path to its list of :class:`BenchmarkResult`.
        Reused outputs are rehydrated into the same result type as newly run
        methods, so post-run summaries cover the complete dataset.
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
        # consistent and comparable across images.  Workflows may supply a
        # precomputed list after claiming the matching process locks.
        if run_ids is None:
            run_ids = make_run_ids(segmenters)
        else:
            run_ids = list(run_ids)
            if len(run_ids) != len(segmenters):
                raise ValueError("run_ids must contain exactly one ID per segmenter.")
            if len(set(run_ids)) != len(run_ids):
                raise ValueError("run_ids must be unique within one dataset worker.")
        segmenter_configs = {
            run_id: segmenter_config_dict(segmenter)
            for segmenter, run_id in zip(segmenters, run_ids)
        }
        if resume_store is not None:
            resume_store.prepare(segmenter_configs)

        all_results: dict[Path, list[BenchmarkResult]] = {}

        for img_idx, image in enumerate(images, 1):
            gt = ground_truths.get(image)

            if self.reporter:
                self.reporter.print_image_header(
                    image,
                    img_idx,
                    len(images),
                    evaluation_mode="supervised" if gt is not None else "prediction-only",
                )
            else:
                self._log(f"\n[Image {img_idx}/{len(images)}] {image.name}")
                self._log("-" * 60)

            img_results = []
            for i, (segmenter, run_id) in enumerate(zip(segmenters, run_ids), 1):
                expected_config = segmenter_configs[run_id]
                display = getattr(segmenter, "display_name", segmenter.name)

                if resume_store is not None:
                    artifact, reason = resume_store.load_completed(
                        run_id=run_id,
                        image_path=image,
                        expected_config=expected_config,
                        expected_method_name=display,
                        ground_truth_geojson=gt,
                    )
                    if artifact is not None:
                        resumed = self._result_from_resumed_artifact(
                            artifact,
                            image_path=image,
                            ground_truth_geojson=gt,
                            expected_config=expected_config,
                        )
                        if resumed is not None:
                            self.results.append(resumed)
                            img_results.append(resumed)
                            if self.reporter:
                                self.reporter.print_method_skipped(run_id, i, len(segmenters))
                            else:
                                self._log(f"  [{i}/{len(segmenters)}] {run_id}  reused completed output")
                            continue
                        reason = "saved metrics cannot be reconstructed"

                    # A score/prediction pair that did not validate must never
                    # remain beside the newly computed result for this slide.
                    if reason != "not yet written":
                        resume_store.discard_partial(run_id, image.stem)
                        if not self.reporter:
                            self._log(
                                f"  [{i}/{len(segmenters)}] {run_id}  recomputing ({reason})"
                            )

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

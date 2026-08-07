from __future__ import annotations

import math
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from segmenteer.benchmark.resume import fingerprint
from segmenteer.core.base import Segmenter, get_wsi_reader, segmenter_config_dict

if TYPE_CHECKING:
    from segmenteer.benchmark.console import BenchmarkReporter


@dataclass
class BenchmarkResult:
    """One runner result: prediction plus inference/reproducibility metadata only."""

    method_name: str
    run_id: str
    geojson: dict
    execution_time: float
    seconds_per_pixel: float
    image_path: Optional[Path] = None
    segmenter_config: Optional[dict] = None
    native_spacing: Optional[tuple[float, float]] = None
    magnification_used: Optional[str] = None
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


_SKIP_ATTRS = frozenset({"reader", "to_gray_func", "results", "segmenter", "segmenter_func"})
_SLUG_RE = re.compile(r"[^\w\-.]")


def _slug(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return _SLUG_RE.sub("", str(value))[:24]


def make_run_id(segmenter: Segmenter, used_ids: set[str] | None = None) -> str:
    """Return a deterministic method/configuration directory id."""
    base = segmenter.name
    params = {
        key: value
        for key, value in vars(segmenter).items()
        if not key.startswith("_") and key not in _SKIP_ATTRS and not callable(value)
    }
    if params:
        param_str = "_".join(
            f"{key.replace('_', '-')}={_slug(value)}" for key, value in sorted(params.items())
        )
        readable = f"{base}__{param_str}"
    else:
        readable = base

    digest = fingerprint(segmenter_config_dict(segmenter))[:12]
    candidate = f"{readable}__cfg={digest}"
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
    used_ids: set[str] = set()
    return [make_run_id(segmenter, used_ids) for segmenter in segmenters]


def describe_magnification_used(segmenter: Segmenter) -> str | None:
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
        return "AtlasPatch service input scale"
    return None


def _native_spacing_or_none(reader, wsi) -> tuple[float, float] | None:
    try:
        x_mpp, y_mpp = reader.get_mpp(wsi, 0)
        x_mpp, y_mpp = float(x_mpp), float(y_mpp)
    except Exception:
        return None
    if not all(math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp)):
        return None
    return x_mpp, y_mpp


def _native_spacing_from_score(score: dict) -> tuple[float, float] | None:
    raw = score.get("native_spacing_um_per_px")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        values = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) and value > 0 for value in values) else None


def _magnification_used_from_config(config: dict) -> str | None:
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
        return "AtlasPatch service input scale"
    return None


class BenchmarkRunner:
    """Inference-only runner.

    This component only invokes segmentation methods and returns predictions
    plus runtime/configuration metadata. Evaluation and visualization are
    intentionally external consumers.
    """

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
        if self.verbose and self.reporter is None:
            print(message, end=end, flush=True)

    def run_single(
        self,
        segmenter: Segmenter,
        image: Path,
        *,
        run_id: Optional[str] = None,
        image_path: Optional[Path] = None,
        _index: int = 0,
        _total: int = 0,
    ) -> BenchmarkResult:
        if run_id is None:
            run_id = make_run_id(segmenter)
        display = run_id

        if self.reporter and _index and _total:
            self.reporter.print_method_start(display, _index, _total)
        else:
            self._log(f"Running {display}...")

        start_time = time.perf_counter()
        try:
            prediction = segmenter.segment(image)
            execution_time = time.perf_counter() - start_time

            reader = getattr(segmenter, "reader", None) or get_wsi_reader()
            wsi = reader.read(str(image))
            width, height = reader.get_size(wsi, 0)
            area = width * height
            result = BenchmarkResult(
                method_name=getattr(segmenter, "display_name", segmenter.name),
                run_id=run_id,
                geojson=prediction,
                execution_time=execution_time,
                seconds_per_pixel=execution_time / area if area else 0.0,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path or Path(image),
                native_spacing=_native_spacing_or_none(reader, wsi),
                magnification_used=describe_magnification_used(segmenter),
            )
        except Exception as exc:  # noqa: BLE001
            execution_time = time.perf_counter() - start_time
            error_msg = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            if self.reporter:
                self.reporter.print_method_failed(display, f"{type(exc).__name__}: {exc}")
            else:
                self._log(f"  ✗ FAILED after {execution_time:.2f}s — {type(exc).__name__}: {exc}")
            result = BenchmarkResult(
                method_name=getattr(segmenter, "display_name", segmenter.name),
                run_id=run_id,
                geojson={"type": "FeatureCollection", "features": []},
                execution_time=execution_time,
                seconds_per_pixel=0.0,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path or Path(image),
                magnification_used=describe_magnification_used(segmenter),
                error=error_msg,
            )

        self.results.append(result)
        if self.reporter and not result.failed:
            self.reporter.print_method_done(result)
        elif not self.reporter and not result.failed:
            self._log(f"  ✓ {display} complete\n")
        if self.result_callback:
            self.result_callback(result)
        return result

    def run_multiple(self, segmenters: list[Segmenter], image: Path) -> list[BenchmarkResult]:
        if self.reporter:
            self.reporter.print_header(len(segmenters), image)
        else:
            self._log(f"\nRunning {len(segmenters)} methods on image {image}")
            self._log("=" * 60 + "\n")

        run_ids = make_run_ids(segmenters)
        results: list[BenchmarkResult] = []
        for index, (segmenter, run_id) in enumerate(zip(segmenters, run_ids), 1):
            results.append(
                self.run_single(
                    segmenter,
                    image,
                    run_id=run_id,
                    image_path=Path(image),
                    _index=index,
                    _total=len(segmenters),
                )
            )
        if self.reporter:
            self.reporter.print_run_complete(len(segmenters), sum(result.failed for result in results))
        return results

    @staticmethod
    def _result_from_resumed_artifact(
        artifact,
        *,
        image_path: Path,
        expected_config: dict,
        expected_method_name: str,
        run_id: str,
    ) -> BenchmarkResult:
        # Runtime timing is intentionally not persisted by RUNNER. Reused
        # predictions therefore report zero execution time while retaining the
        # method configuration needed for reproducibility/display.
        return BenchmarkResult(
            method_name=expected_method_name,
            run_id=run_id,
            geojson=artifact.geojson,
            execution_time=0.0,
            seconds_per_pixel=0.0,
            segmenter_config=expected_config,
            image_path=image_path,
            native_spacing=None,
            magnification_used=_magnification_used_from_config(expected_config),
        )

    def run_dataset(
        self,
        segmenters: list[Segmenter],
        images: list[Path],
        *,
        resume_store=None,
        run_ids: list[str] | None = None,
    ) -> dict[Path, list[BenchmarkResult]]:
        if self.reporter:
            self.reporter.print_dataset_header(len(segmenters), len(images))

        if run_ids is None:
            run_ids = make_run_ids(segmenters)
        elif len(run_ids) != len(segmenters) or len(set(run_ids)) != len(run_ids):
            raise ValueError("run_ids must contain one unique ID per segmenter")

        configs = {
            run_id: segmenter_config_dict(segmenter)
            for segmenter, run_id in zip(segmenters, run_ids)
        }
        if resume_store is not None:
            resume_store.prepare(configs)

        all_results: dict[Path, list[BenchmarkResult]] = {}
        for image_index, image in enumerate(images, 1):
            image = Path(image)
            if self.reporter:
                self.reporter.print_image_header(image, image_index, len(images))

            image_results: list[BenchmarkResult] = []
            for method_index, (segmenter, run_id) in enumerate(zip(segmenters, run_ids), 1):
                expected_config = configs[run_id]
                display_name = getattr(segmenter, "display_name", segmenter.name)
                if resume_store is not None:
                    artifact, reason = resume_store.load_completed(
                        run_id=run_id,
                        image_path=image,
                        expected_config=expected_config,
                        expected_method_name=display_name,
                    )
                    if artifact is not None:
                        result = self._result_from_resumed_artifact(
                            artifact,
                            image_path=image,
                            expected_config=expected_config,
                            expected_method_name=display_name,
                            run_id=run_id,
                        )
                        self.results.append(result)
                        image_results.append(result)
                        if self.reporter:
                            self.reporter.print_method_skipped(run_id, method_index, len(segmenters))
                        continue
                    if reason != "not yet written":
                        resume_store.discard_partial(run_id, image.stem)

                image_results.append(
                    self.run_single(
                        segmenter,
                        image,
                        run_id=run_id,
                        image_path=image,
                        _index=method_index,
                        _total=len(segmenters),
                    )
                )
            all_results[image] = image_results
            if self.reporter:
                self.reporter.print_run_complete(
                    len(segmenters), sum(result.failed for result in image_results)
                )
        return all_results

    def clear_results(self) -> None:
        self.results = []

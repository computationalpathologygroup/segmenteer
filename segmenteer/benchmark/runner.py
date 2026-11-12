from __future__ import annotations
import math
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
from segmenteer.benchmark.resume import fingerprint
from segmenteer.benchmark.slide_identity import validate_unique_slide_stems
from segmenteer.methods.identity import display_name_for_segmenter
from segmenteer.core.base import Segmenter, get_wsi_reader, segmenter_config_dict
from segmenteer.benchmark.profiling import PhaseRecorder, active_recorder

if TYPE_CHECKING:
    from segmenteer.benchmark.console import BenchmarkReporter


@dataclass
class BenchmarkResult:
    method_name: str
    run_id: str
    geojson: dict
    execution_time: float | None
    seconds_per_pixel: float | None
    image_path: Optional[Path] = None
    segmenter_config: Optional[dict] = None
    native_spacing: Optional[tuple[float, float]] = None
    magnification_used: Optional[str] = None
    error: Optional[str] = None
    native_pixels: int | None = None
    device: str | None = None
    measurement_phase: str | None = None
    reused: bool = False
    prediction_protocol: str | None = None
    initialization_time_s: float | None = None
    algorithm_time_s: float | None = None
    segmentation_call_time_s: float | None = None
    total_processing_time_s: float | None = None
    image_loading_time_s: float | None = None
    preprocessing_time_s: float | None = None
    polygon_conversion_time_s: float | None = None
    serialization_time_s: float | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


_SKIP_ATTRS = frozenset(
    {"reader", "to_gray_func", "results", "segmenter", "segmenter_func"}
)
_SLUG_RE = re.compile("[^\\w\\-.]")
_MAX_RUN_ID_BYTES = 160
_MOG2_VISIBLE_PARAMS = ("mpp", "history", "var_threshold", "training_frames")


def _slug(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    return _SLUG_RE.sub("", str(value))[:24]


def _bounded_run_id(readable: str, digest: str) -> str:
    suffix = f"__cfg={digest}"
    capacity = _MAX_RUN_ID_BYTES - len(suffix.encode("utf-8"))
    if len(readable.encode("utf-8")) > capacity:
        readable = (
            readable.encode("utf-8")[:capacity]
            .decode("utf-8", errors="ignore")
            .rstrip("._-")
        )
    return f"{readable}{suffix}"


def make_run_id(segmenter: Segmenter, used_ids: set[str] | None = None) -> str:
    base = segmenter.name
    params = {
        key: value
        for key, value in vars(segmenter).items()
        if not key.startswith("_") and key not in _SKIP_ATTRS and (not callable(value))
    }
    if type(segmenter).__name__ == "BackgroundSubtractorMOG2Segmenter":
        params = {key: params[key] for key in _MOG2_VISIBLE_PARAMS if key in params}
    if params:
        param_str = "_".join(
            f"{key.replace('_', '-')}={_slug(value)}"
            for key, value in sorted(params.items())
        )
        readable = f"{base}__{param_str}"
    else:
        readable = base
    digest = fingerprint(segmenter_config_dict(segmenter))[:12]
    candidate = _bounded_run_id(readable, digest)
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
    seen: dict[str, int] = {}
    result: list[str] = []
    for index, segmenter in enumerate(segmenters, 1):
        key = fingerprint(segmenter_config_dict(segmenter))
        if key in seen:
            raise ValueError(
                f"Duplicate segmentation configuration at positions {seen[key]} and {index}: {segmenter.name}."
            )
        seen[key] = index
        result.append(make_run_id(segmenter, used_ids))
    return result


def describe_magnification_used(segmenter: Segmenter) -> str | None:
    target_mag = getattr(segmenter, "target_mag", None)
    if isinstance(target_mag, (int, float)) and (not isinstance(target_mag, bool)):
        target_mag = float(target_mag)
        if math.isfinite(target_mag) and target_mag > 0:
            return f"{target_mag:g}x"
    mpp = getattr(segmenter, "mpp", None)
    if isinstance(mpp, (int, float)) and (not isinstance(mpp, bool)):
        mpp = float(mpp)
        if math.isfinite(mpp) and mpp > 0:
            return f"{mpp:g} µm/px"
    if getattr(segmenter, "name", "") == "atlaspatch_sam2":
        return "AtlasPatch service input scale"
    return None


def _native_spacing_or_none(reader, wsi) -> tuple[float, float] | None:
    try:
        x_mpp, y_mpp = reader.get_mpp(wsi, 0)
        x_mpp, y_mpp = (float(x_mpp), float(y_mpp))
    except Exception:
        return None
    if not all((math.isfinite(value) and value > 0 for value in (x_mpp, y_mpp))):
        return None
    return (x_mpp, y_mpp)


def _native_spacing_from_score(score: dict) -> tuple[float, float] | None:
    raw = score.get("native_spacing_um_per_px")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        values = (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None
    return (
        values
        if all((math.isfinite(value) and value > 0 for value in values))
        else None
    )


def _magnification_used_from_config(config: dict) -> str | None:
    params = config.get("params") if isinstance(config, dict) else None
    if not isinstance(params, dict):
        return None
    target_mag = params.get("target_mag")
    if isinstance(target_mag, (int, float)) and (not isinstance(target_mag, bool)):
        target_mag = float(target_mag)
        if math.isfinite(target_mag) and target_mag > 0:
            return f"{target_mag:g}x"
    mpp = params.get("mpp")
    if isinstance(mpp, (int, float)) and (not isinstance(mpp, bool)):
        mpp = float(mpp)
        if math.isfinite(mpp) and mpp > 0:
            return f"{mpp:g} µm/px"
    class_path = str(config.get("class", ""))
    if class_path.endswith("AtlasPatchSAM2Segmenter"):
        return "AtlasPatch service input scale"
    return None


def _sync_accelerator(segmenter: Segmenter) -> None:
    device = str(getattr(segmenter, "device", "")).casefold()
    if not (device.startswith("cuda") or device.startswith("mps")):
        return
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)
    elif (
        device.startswith("mps")
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.synchronize()


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
        self._executed_run_ids: set[str] = set()
        self._initialized_objects: set[int] = set()

    @staticmethod
    def _read_slide_geometry(
        segmenter: Segmenter, image: Path
    ) -> tuple[int | None, tuple[float, float] | None]:
        try:
            reader = getattr(segmenter, "reader", None) or get_wsi_reader()
            wsi = reader.read(str(image))
            width, height = reader.get_size(wsi, 0)
            area = width * height if width > 0 and height > 0 else None
            return area, _native_spacing_or_none(reader, wsi)
        except Exception:
            return None, None

    def run_single(
        self,
        segmenter: Segmenter,
        image: Path,
        *,
        run_id: Optional[str] = None,
        image_path: Optional[Path] = None,
        _index: int = 0,
        _total: int = 0,
        _slide_geometry: tuple[int | None, tuple[float, float] | None] | None = None,
    ) -> BenchmarkResult:
        if run_id is None:
            run_id = make_run_id(segmenter)
        phase = (
            "first_call_in_process"
            if run_id not in self._executed_run_ids
            else "subsequent_call_in_process"
        )
        self._executed_run_ids.add(run_id)
        device = str(getattr(segmenter, "device", "cpu"))
        recorder = PhaseRecorder(lambda: _sync_accelerator(segmenter))
        first_use = id(segmenter) not in self._initialized_objects
        self._initialized_objects.add(id(segmenter))
        constructor_time = (
            getattr(segmenter, "_constructor_time_s", None) if first_use else None
        )
        if constructor_time is not None:
            constructor_time = float(constructor_time)
        execution_time = None
        started = None
        try:
            _sync_accelerator(segmenter)
            started = time.perf_counter()
            with active_recorder(recorder):
                prediction = segmenter.segment(image)
            _sync_accelerator(segmenter)
            execution_time = time.perf_counter() - started
            area, spacing = (
                _slide_geometry
                if _slide_geometry is not None
                else self._read_slide_geometry(segmenter, Path(image))
            )
            spans = recorder.measurements
            setup = spans.get("model_initialization")
            initialization_time = (
                (constructor_time or 0.0) + (setup or 0.0)
                if constructor_time is not None or setup is not None
                else None
            )
            algorithm = spans.get("algorithm")
            algorithm_time = (
                max(0.0, algorithm - (setup or 0.0)) if algorithm is not None else None
            )
            result = BenchmarkResult(
                method_name=display_name_for_segmenter(segmenter),
                run_id=run_id,
                geojson=prediction,
                prediction_protocol=getattr(segmenter, "prediction_protocol", None),
                execution_time=execution_time,
                segmentation_call_time_s=execution_time,
                seconds_per_pixel=execution_time / area if area else None,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path or Path(image),
                native_spacing=spacing,
                magnification_used=describe_magnification_used(segmenter),
                native_pixels=area,
                device=device,
                measurement_phase=phase,
                initialization_time_s=initialization_time,
                algorithm_time_s=algorithm_time,
                image_loading_time_s=spans.get("image_loading"),
                preprocessing_time_s=spans.get("preprocessing"),
                polygon_conversion_time_s=spans.get("polygon_conversion"),
            )
        except Exception as exc:
            execution_time = (
                time.perf_counter() - started if started is not None else None
            )
            error_msg = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            result = BenchmarkResult(
                method_name=display_name_for_segmenter(segmenter),
                run_id=run_id,
                geojson={"type": "FeatureCollection", "features": []},
                execution_time=execution_time,
                seconds_per_pixel=None,
                segmenter_config=segmenter_config_dict(segmenter),
                image_path=image_path or Path(image),
                magnification_used=describe_magnification_used(segmenter),
                device=device,
                measurement_phase=phase,
                error=error_msg,
            )
            if self.reporter:
                self.reporter.print_failure(
                    Path(image).name, result.method_name, str(exc)
                )
            else:
                print(
                    f"FAILED {Path(image).name} · {result.method_name}: {exc}",
                    flush=True,
                )
        self.results.append(result)
        if not result.failed and self.result_callback:
            self.result_callback(result)
        if not result.failed and result.total_processing_time_s is None:
            result.total_processing_time_s = result.segmentation_call_time_s
        return result

    def run_multiple(
        self, segmenters: list[Segmenter], image: Path
    ) -> list[BenchmarkResult]:
        run_ids = make_run_ids(segmenters)
        return [
            self.run_single(segmenter, image, run_id=run_id, image_path=Path(image))
            for segmenter, run_id in zip(segmenters, run_ids)
        ]

    @staticmethod
    def _result_from_resumed_artifact(
        artifact,
        *,
        image_path: Path,
        expected_config: dict,
        expected_method_name: str,
        run_id: str,
    ) -> BenchmarkResult:
        timing = artifact.timing
        speed = timing.get("seconds_per_native_megapixel")
        spacing = timing.get("native_spacing_um_per_px")
        return BenchmarkResult(
            method_name=expected_method_name,
            run_id=run_id,
            geojson=artifact.geojson,
            execution_time=timing["segmentation_call_time_s"],
            segmentation_call_time_s=timing["segmentation_call_time_s"],
            seconds_per_pixel=speed / 1000000 if speed is not None else None,
            segmenter_config=expected_config,
            image_path=image_path,
            native_spacing=tuple(spacing) if spacing else None,
            magnification_used=timing.get("magnification_used")
            or _magnification_used_from_config(expected_config),
            native_pixels=timing.get("native_pixels"),
            device=timing.get("device"),
            measurement_phase=timing.get("measurement_phase"),
            reused=True,
            prediction_protocol=timing.get("prediction_protocol"),
            **{
                field: timing.get(field)
                for field in (
                    "initialization_time_s",
                    "algorithm_time_s",
                    "total_processing_time_s",
                    "image_loading_time_s",
                    "preprocessing_time_s",
                    "polygon_conversion_time_s",
                    "serialization_time_s",
                )
            },
        )

    def run_dataset(
        self,
        segmenters: list[Segmenter],
        images: list[Path],
        *,
        resume_store=None,
        run_ids: list[str] | None = None,
        retain_geometries: bool = True,
    ) -> dict[Path, list[BenchmarkResult]]:
        validate_unique_slide_stems(images, context="benchmark runner")
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
        from segmenteer.benchmark.console import BenchmarkProgress

        all_results: dict[Path, list[BenchmarkResult]] = {}
        progress_console = self.reporter.console if self.reporter else None
        with BenchmarkProgress(
            slides=len(images),
            methods=len(segmenters),
            console=progress_console,
            interactive=self.reporter is not None,
        ) as progress:
            for image_index, image in enumerate(images, 1):
                image = Path(image)
                progress.begin_slide(slide_index=image_index, slide=image.name)
                geometry = None
                image_results: list[BenchmarkResult] = []
                for method_index, (segmenter, run_id) in enumerate(
                    zip(segmenters, run_ids), 1
                ):
                    expected_config = configs[run_id]
                    display_name = display_name_for_segmenter(segmenter)
                    progress.begin_method(
                        slide_index=image_index,
                        slide=image.name,
                        method_index=method_index,
                        method=display_name,
                    )
                    if resume_store is not None:
                        artifact, reason = resume_store.load_completed(
                            run_id=run_id,
                            image_path=image,
                            expected_config=expected_config,
                            expected_method_name=display_name,
                            retain_geometry=retain_geometries,
                            expected_prediction_protocol=getattr(
                                segmenter, "prediction_protocol", None
                            ),
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
                            if not retain_geometries:
                                result.geojson = {
                                    "type": "FeatureCollection",
                                    "features": [],
                                }
                            image_results.append(result)
                            progress.advance(
                                reused=True,
                                method=display_name,
                                slide_index=image_index,
                                method_index=method_index,
                            )
                            continue
                        if reason != "not yet written":
                            resume_store.discard_partial(run_id, image.stem)
                    if geometry is None:
                        geometry = self._read_slide_geometry(segmenter, image)
                    result = self.run_single(
                        segmenter,
                        image,
                        run_id=run_id,
                        image_path=image,
                        _index=method_index,
                        _total=len(segmenters),
                        _slide_geometry=geometry,
                    )
                    if not retain_geometries:
                        result.geojson = {"type": "FeatureCollection", "features": []}
                    image_results.append(result)
                    progress.advance(
                        failed=result.failed,
                        method=display_name,
                        slide_index=image_index,
                        method_index=method_index,
                    )
                all_results[image] = image_results
        return all_results

    def clear_results(self) -> None:
        self.results = []

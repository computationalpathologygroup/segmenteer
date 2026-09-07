"""Lazy public API for the three decoupled benchmark-facing components."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BenchmarkReporter": ("segmenteer.benchmark.console", "BenchmarkReporter"),
    "BenchmarkResult": ("segmenteer.benchmark.runner", "BenchmarkResult"),
    "BenchmarkRunner": ("segmenteer.benchmark.runner", "BenchmarkRunner"),
    "PredictionOutputWriter": ("segmenteer.benchmark.output", "PredictionOutputWriter"),
    "EvaluationSummary": ("segmenteer.benchmark.evaluation", "EvaluationSummary"),
    "evaluate_output_directory": ("segmenteer.benchmark.evaluation", "evaluate_output_directory"),
    "EnsembleOutputWriter": ("segmenteer.benchmark.ensemble", "EnsembleOutputWriter"),
    "load_ensemble_members": ("segmenteer.benchmark.ensemble", "load_ensemble_members"),
    "make_ensemble_run_id": ("segmenteer.benchmark.ensemble", "make_ensemble_run_id"),
    "make_run_id": ("segmenteer.benchmark.runner", "make_run_id"),
    "make_run_ids": ("segmenteer.benchmark.runner", "make_run_ids"),
    "export_results_json": ("segmenteer.benchmark.reporting", "export_results_json"),
    "export_results_csv": ("segmenteer.benchmark.reporting", "export_results_csv"),
    "export_dataset_results_csv": ("segmenteer.benchmark.reporting", "export_dataset_results_csv"),
    "run_single_image": ("segmenteer.benchmark.workflows", "run_single_image"),
    "run_dataset": ("segmenteer.benchmark.workflows", "run_dataset"),
    "prepare_dataset_output": ("segmenteer.benchmark.workflows", "prepare_dataset_output"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = list(_LAZY_EXPORTS)

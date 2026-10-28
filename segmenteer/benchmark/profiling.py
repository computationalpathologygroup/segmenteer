from __future__ import annotations
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PhaseRecorder:
    synchronize: Callable[[], None]
    measurements: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - start
            self.measurements[name] = self.measurements.get(name, 0.0) + elapsed


_ACTIVE: ContextVar[PhaseRecorder | None] = ContextVar(
    "segmenteer_benchmark_phases", default=None
)


@contextmanager
def active_recorder(recorder: PhaseRecorder):
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


@contextmanager
def measure_phase(name: str):
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
    else:
        with recorder.measure(name):
            yield

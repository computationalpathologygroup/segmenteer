from __future__ import annotations
import threading
import time
from pathlib import Path
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class EvaluationProgress:
    def __init__(
        self,
        total: int,
        *,
        enabled: bool = True,
        heartbeat_seconds: float = 15.0,
        console: Console | None = None,
    ):
        self.total = total
        self.enabled = enabled
        self.console = console or Console(stderr=True)
        self.interactive = enabled and self.console.is_terminal
        self.heartbeat_seconds = heartbeat_seconds
        self.completed = 0
        self.supervised = 0
        self.skipped = 0
        self.failed = 0
        self.reused = 0
        self.started_at = time.monotonic()
        self.active = ""
        self.stage_name = ""
        self.active_since = 0.0
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._progress = None
        self._task_id = None

    def __enter__(self):
        if not self.enabled:
            return self
        if self.interactive:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Evaluating[/bold cyan]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[detail]}"),
                console=self.console,
                refresh_per_second=4,
                expand=True,
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "Evaluating", total=self.total, detail="Preparing first prediction"
            )
        else:
            self.console.print(
                f"Evaluation: {self.total} prediction(s), starting.", highlight=False
            )
            if self.heartbeat_seconds > 0:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat, daemon=True
                )
                self._heartbeat_thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(self.heartbeat_seconds):
            if self.active:
                elapsed = time.monotonic() - self.active_since
                self.console.print(
                    f"Evaluation still running [{self.completed + 1}/{self.total}]: "
                    f"{self.active} · {self.stage_name} · current item {elapsed:.0f}s",
                    highlight=False,
                )

    def begin(self, index: int, path: Path, run_id: str):
        if not self.enabled:
            return
        self.active = f"{path.name} · {run_id[:40]}"
        self.active_since = time.monotonic()
        self.stage("Locating ground truth")
        if not self.interactive:
            self.console.print(
                f"Evaluating [{index}/{self.total}]: {self.active}", highlight=False
            )

    def stage(self, name: str):
        if not self.enabled:
            return
        self.stage_name = name
        if self.interactive:
            self._progress.update(
                self._task_id, detail=f"{self.active[:47]} · {name}", refresh=True
            )

    def advance(self, mode: str, *, error: str = ""):
        if not self.enabled:
            return
        self.completed += 1
        if mode == "supervised":
            self.supervised += 1
        elif mode == "skipped":
            self.skipped += 1
        elif mode == "reused":
            self.reused += 1
        else:
            self.failed += 1
        self.active = ""
        elapsed = time.monotonic() - self.started_at
        remaining = (
            elapsed / self.completed * (self.total - self.completed)
            if self.completed
            else 0.0
        )
        detail = f"Supervised {self.supervised} · Skipped {self.skipped} · Reused {self.reused} · Failed {self.failed}"
        if self.interactive:
            self._progress.update(self._task_id, advance=1, detail=detail, refresh=True)
            if mode == "failed":
                self.console.print(
                    f"[red]FAILED[/red] {self.completed}/{self.total}: {error}",
                    highlight=False,
                )
        else:
            self.console.print(
                f"Evaluated [{self.completed}/{self.total} · {self.completed / self.total:.1%}]: "
                f"{mode.upper()} · elapsed {elapsed:.0f}s · ETA {remaining:.0f}s · {detail}"
                + (f" · {error}" if error else ""),
                highlight=False,
            )

    def finish_stage(self, stage: str):
        if not self.enabled:
            return
        self.active = stage
        self.active_since = time.monotonic()
        self.stage_name = stage
        if self.interactive:
            self._progress.update(self._task_id, detail=stage, refresh=True)
        else:
            self.console.print(stage, highlight=False)

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=0.5)
        if self._progress is not None:
            self._progress.stop()
        if self.enabled:
            elapsed = time.monotonic() - self.started_at
            self.console.print(
                f"Evaluation progress: {self.completed}/{self.total} · supervised {self.supervised} · "
                f"skipped {self.skipped} · reused {self.reused} · failed {self.failed} · elapsed {elapsed:.1f}s",
                highlight=False,
            )
        return False

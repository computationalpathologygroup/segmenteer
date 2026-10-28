from __future__ import annotations
from pathlib import Path
import os
import time
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from collections import Counter
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

_THEME = Theme({"header": "bold cyan"})


class BenchmarkReporter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(theme=_THEME)

    def print_dataset_header(self, n_methods: int, n_images: int) -> None:
        self.console.print(
            Panel.fit(
                f"[header]segmenteer[/header] · benchmark\nSlides: {n_images}  Methods: {n_methods}  Runs: {n_images * n_methods}",
                border_style="cyan",
            )
        )

    def print_failure(self, slide: str, method: str, error: str) -> None:
        self.console.print(f"[red]FAILED[/red] {slide} · {method}: {error}")

    @staticmethod
    def _summary_metric(row: dict, name: str, compact: bool) -> str:
        mean = row.get(f"{name}_mean")
        std = row.get(f"{name}_std")
        if mean is None:
            return "N/A"
        precision = 6 if mean < 0.001 else 3
        mean_text = f"{mean:.{precision}f}"
        if std is None:
            return f"{mean_text}(—)" if compact else f"{mean_text} ± —"
        sd_text = f"{std:.{precision}f}"
        return f"{mean_text}({sd_text})" if compact else f"{mean_text} ± {sd_text}"

    def print_summary(
        self, summaries: list[dict], *, expected_slides: int | None = None
    ) -> None:
        compact = self.console.width < 118
        table = Table(
            title="Benchmark by method · mean and sample SD",
            box=box.SIMPLE if compact else box.ROUNDED,
            border_style="cyan",
            show_edge=not compact,
        )
        table.add_column(
            "Method",
            style="bold",
            overflow="ellipsis",
            no_wrap=True,
            max_width=29 if compact else None,
        )
        if not compact:
            table.add_column("Config", style="dim", no_wrap=True)
        table.add_column("N", justify="right", no_wrap=True)
        table.add_column("Init s", justify="right", no_wrap=True)
        table.add_column("Alg s", justify="right", no_wrap=True)
        table.add_column("Total s", justify="right", no_wrap=True)
        if not compact:
            table.add_column("s/native Mpx", justify="right", no_wrap=True)
        if not summaries:
            self.console.print(
                "[yellow]No completed predictions with valid benchmark records.[/yellow]"
            )
            return
        name_counts = Counter(row["segmentation_method"] for row in summaries)
        for row in summaries:
            config = (
                row["run_id"].rsplit("__cfg=", 1)[-1]
                if "__cfg=" in row["run_id"]
                else row["run_id"]
            )
            values = [
                (
                    f"[{config[:6]}] {row['segmentation_method']}"
                    if compact and name_counts[row["segmentation_method"]] > 1
                    else str(row["segmentation_method"])
                ),
                *([config[:10]] if not compact else []),
                f"{row['n_slides']}/{expected_slides}"
                if expected_slides is not None
                else str(row["n_slides"]),
                self._summary_metric(row, "initialization_time_s", compact),
                self._summary_metric(row, "algorithm_time_s", compact),
                self._summary_metric(row, "total_processing_time_s", compact),
                *(
                    [self._summary_metric(row, "seconds_per_native_megapixel", compact)]
                    if not compact
                    else []
                ),
            ]
            table.add_row(*values)
        self.console.print(table)
        self.console.print(
            "[dim]Values: mean (SD) in compact view, mean ± SD in wide view. — = one sample. Initialization samples are per worker, not per slide. See benchmark_summary.csv for sample counts and native-Mpx rates.[/dim]"
        )

    def print_saved(self, output_dir: Path | str, files: list[str]) -> None:
        self.console.print(
            Panel(
                "\n".join(f"  ✓ {item}" for item in files),
                title=f"saved → {Path(output_dir)}",
                border_style="dim",
            )
        )


class BenchmarkProgress:
    def __init__(
        self,
        *,
        slides: int,
        methods: int,
        console: Console | None = None,
        interactive: bool = True,
    ) -> None:
        self.console = console or Console()
        self.slides = slides
        self.methods = methods
        self.total = slides * methods
        self.completed = 0
        self.saved = 0
        self.reused = 0
        self.failed = 0
        self.started_at = time.monotonic()
        self.interactive = bool(interactive and self.console.is_terminal)
        self.worker_label = os.environ.get("SEGMENTEER_WORKER_LABEL", "").strip()
        self._progress = None
        self._task_id = None

    def __enter__(self):
        if self.interactive:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Segmenting[/bold cyan]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[detail]}"),
                console=self.console,
                expand=True,
                refresh_per_second=4,
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "Segmenting", total=self.total, detail="Preparing first slide"
            )
        else:
            prefix = f"{self.worker_label} " if self.worker_label else ""
            self.console.print(
                f"{prefix}Progress: 0/{self.total} runs, 0/{self.slides} slides, starting.",
                highlight=False,
            )
        return self

    def begin_slide(self, *, slide_index: int, slide: str) -> None:
        if self.interactive:
            self._progress.update(
                self._task_id,
                detail=f"Opening slide {slide_index}/{self.slides}: {slide[:32]}",
                refresh=True,
            )
        else:
            prefix = f"{self.worker_label} " if self.worker_label else ""
            self.console.print(
                f"{prefix}Slide {slide_index}/{self.slides}: {slide}", highlight=False
            )

    def begin_method(
        self, *, slide_index: int, slide: str, method_index: int, method: str
    ) -> None:
        if self.interactive:
            detail = f"{slide_index}/{self.slides} {slide[:28]} · {method_index}/{self.methods} {method[:22]}"
            self._progress.update(self._task_id, detail=detail, refresh=True)

    def advance(
        self,
        *,
        reused: bool = False,
        failed: bool = False,
        method: str = "",
        slide_index: int = 0,
        method_index: int = 0,
    ) -> None:
        self.completed += 1
        if reused:
            self.reused += 1
        elif failed:
            self.failed += 1
        else:
            self.saved += 1
        if self.interactive:
            detail = f"Saved {self.saved} · Reused {self.reused} · Failed {self.failed}"
            self._progress.update(self._task_id, advance=1, detail=detail, refresh=True)
        else:
            elapsed = max(0.0, time.monotonic() - self.started_at)
            remaining = (
                (elapsed / self.completed) * (self.total - self.completed)
                if self.completed
                else 0.0
            )
            status = "REUSED" if reused else "FAILED" if failed else "SAVED"
            prefix = f"{self.worker_label} " if self.worker_label else ""
            self.console.print(
                f"{prefix}[{self.completed}/{self.total} · {self.completed / self.total:.1%}] "
                f"Slide {slide_index}/{self.slides}, method {method_index}/{self.methods}: "
                f"{method} — {status} · elapsed {elapsed:.0f}s · ETA {remaining:.0f}s · "
                f"saved {self.saved}, reused {self.reused}, failed {self.failed}",
                highlight=False,
            )

    def __exit__(self, exc_type, exc, tb):
        if self._progress is not None:
            self._progress.stop()
        elapsed = max(0.0, time.monotonic() - self.started_at)
        prefix = f"{self.worker_label} " if self.worker_label else ""
        self.console.print(
            f"{prefix}Progress: {self.completed}/{self.total} runs · "
            f"saved {self.saved} · reused {self.reused} · failed {self.failed} · "
            f"elapsed {elapsed:.1f}s",
            highlight=False,
        )
        return False

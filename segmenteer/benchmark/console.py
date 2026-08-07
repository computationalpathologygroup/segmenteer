from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from segmenteer.benchmark.runner import BenchmarkResult

__all__ = ["BenchmarkReporter"]

_SEGMENTEER_THEME = Theme(
    {
        "header": "bold cyan",
        "step": "bold white",
        "label": "dim white",
        "value": "white",
        "good": "bold green",
        "warn": "bold yellow",
        "method": "bold magenta",
        "time": "cyan",
        "coverage": "green",
        "metric": "yellow",
        "muted": "dim",
        "title": "bold white",
    }
)


def _bar(ratio: float, width: int = 12, filled: str = "█", empty: str = "░") -> str:
    filled_n = round(ratio * width)
    return filled * filled_n + empty * (width - filled_n)


def _coverage_color(ratio: float) -> str:
    if ratio >= 0.6:
        return "green"
    if ratio >= 0.3:
        return "yellow"
    return "red"


def _time_color(seconds: float) -> str:
    if seconds < 5:
        return "green"
    if seconds < 30:
        return "yellow"
    return "red"


class BenchmarkReporter:
    """Rich-powered reporter for benchmark runs.

    Usage::

        reporter = BenchmarkReporter()
        runner = seg.BenchmarkRunner(reporter=reporter)
        results = runner.run_multiple(segmenters, path)
        reporter.print_summary(results)
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(theme=_SEGMENTEER_THEME)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def print_header(self, n_methods: int, image_path: Path | str) -> None:
        """Print a welcome banner before a single-image benchmark starts."""
        image_path = Path(image_path)
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[header]segmenteer[/header]  [muted]·[/muted]  Tissue Segmentation Benchmark\n\n"
                f"  [label]Image   [/label] [value]{image_path.name}[/value]\n"
                f"  [label]Methods [/label] [value]{n_methods}[/value]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        self.console.print()

    def print_dataset_header(self, n_methods: int, n_images: int) -> None:
        """Print a welcome banner before a dataset benchmark starts."""
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[header]segmenteer[/header]  [muted]·[/muted]  Dataset Benchmark\n\n"
                f"  [label]Images  [/label] [value]{n_images}[/value]\n"
                f"  [label]Methods [/label] [value]{n_methods}[/value]\n"
                f"  [label]Total   [/label] [value]{n_images * n_methods} runs[/value]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        self.console.print()

    def print_image_header(
        self,
        image_path: Path | str,
        index: int,
        total: int,
        evaluation_mode: str | None = None,
    ) -> None:
        """Print a section rule when moving to a new image in a dataset run."""
        image_path = Path(image_path)
        counter = f"  [muted]{index}/{total}[/muted]" if index and total else ""
        mode = (
            f"  [muted]· {evaluation_mode}[/muted]"
            if evaluation_mode is not None
            else ""
        )
        self.console.print(
            Rule(f"[header]{image_path.name}[/header]{counter}{mode}", style="cyan")
        )
        self.console.print()

    def print_method_start(self, name: str, index: int, total: int) -> None:
        """Print a line announcing that a method is starting."""
        counter = Text(f"[{index}/{total}]", style="muted")
        method = Text(f" {name}", style="method")
        self.console.print(counter + method + Text("  running…", style="muted"))

    def print_method_skipped(self, name: str, index: int, total: int) -> None:
        """Print a compact line for a validated result reused during resume."""
        counter = Text(f"[{index}/{total}]", style="muted")
        method = Text(f" {name}", style="method")
        self.console.print(counter + method + Text("  reused completed output", style="muted"))
        self.console.print()

    def print_method_done(self, result: BenchmarkResult) -> None:
        """Print a compact one-line summary of a just-finished result."""
        if result.error:
            # failure is already printed by print_method_failed; just add spacing
            self.console.print()
            return
        t_color = _time_color(result.execution_time)
        if result.unsupervised_metrics is None:
            line = (
                Text("  ✓ ", style="good")
                + Text(f"{result.execution_time:7.2f}s", style=t_color)
                + Text("  prediction + metadata saved", style="muted")
            )
            self.console.print(line)
            self.console.print()
            return

        # Legacy/precomputed metrics remain displayable without forcing a new
        # benchmark to calculate them.
        c_ratio = result.unsupervised_metrics.coverage_ratio
        c_color = _coverage_color(c_ratio)
        bar = _bar(c_ratio)
        line = (
            Text("  ✓ ", style="good")
            + Text(f"{result.execution_time:7.2f}s", style=t_color)
            + Text("  objects: ", style="muted")
            + Text(f"{result.unsupervised_metrics.num_objects:4d}", style="value")
            + Text("  coverage: ", style="muted")
            + Text(f"{bar} {c_ratio * 100:5.1f}%", style=c_color)
        )
        self.console.print(line)
        self.console.print()

    def print_method_failed(self, name: str, error_msg: str) -> None:
        """Print an error banner when a method raises an exception."""
        # Truncate very long error strings so they don't swamp the console
        short = error_msg if len(error_msg) <= 120 else error_msg[:117] + "…"
        self.console.print(
            Text("  ✗ FAILED  ", style="bold red") + Text(short, style="dim red")
        )

    def print_run_complete(self, n: int, n_failed: int = 0) -> None:
        ok = n - n_failed
        if n_failed:
            msg = f"[good]✓ {ok} ok[/good]  [bold red]✗ {n_failed} failed[/bold red]"
        else:
            msg = f"[good]✓ {n} method{'s' if n != 1 else ''} complete[/good]"
        self.console.print(Rule(msg, style="dim"))
        self.console.print()

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------

    def print_summary(self, results: list[BenchmarkResult]) -> None:
        """Print the full post-run summary (performance + quality tables)."""
        self.console.print()
        self.console.print(Rule("[title]Results Summary[/title]", style="cyan"))
        self.console.print()
        self._print_performance_table(results)
        self.console.print()
        if any(result.unsupervised_metrics is not None for result in results):
            self._print_quality_table(results)
        else:
            self.console.print(
                "[muted]Quality metrics are deferred. Run evaluate_outputs.py "
                "against this output folder when evaluation is required.[/muted]"
            )
        self.console.print()
        failures = [r for r in results if r.failed]
        if failures:
            self._print_failures_section(failures)

    def _print_failures_section(self, failures: list[BenchmarkResult]) -> None:
        self.console.print(Rule("[bold red]Failures[/bold red]", style="red"))
        self.console.print()
        for r in failures:
            self.console.print(
                Panel(
                    r.error or "(no details)",
                    title=f"[bold red]✗ {r.method_name}[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
        self.console.print()

    def _print_performance_table(self, results: list[BenchmarkResult]) -> None:
        table = Table(
            title="Performance",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            show_lines=False,
            pad_edge=True,
        )
        table.add_column("Method", style="method", no_wrap=True, min_width=22)
        table.add_column("Time (s)", style="time", justify="right")
        table.add_column("s / Mpx", justify="right", style="muted")
        table.add_column("Speed", justify="left", min_width=14)

        # Rank by execution time for relative bar widths
        max_time = max((r.execution_time for r in results), default=1.0)

        for result in results:
            t = result.execution_time
            if result.error:
                table.add_row(
                    result.method_name,
                    Text("FAILED", style="bold red"),
                    "—",
                    Text("░" * 12, style="dim red"),
                )
                continue
            bar_width = round((t / max_time) * 12) if max_time > 0 else 0
            bar = "█" * bar_width + "░" * (12 - bar_width)
            spp_mpx = result.seconds_per_pixel * 1_000_000
            table.add_row(
                result.method_name,
                f"{t:.3f}",
                f"{spp_mpx:.4f}",
                Text(bar, style=_time_color(t)),
            )

        self.console.print(table, justify="left")

    def _print_quality_table(self, results: list[BenchmarkResult]) -> None:
        has_supervised = any(r.supervised_metrics for r in results)

        table = Table(
            title="Segmentation Quality",
            box=box.ROUNDED,
            border_style="cyan",
            header_style="bold cyan",
            show_lines=False,
            pad_edge=True,
        )
        table.add_column("Method", style="method", no_wrap=True, min_width=22)
        table.add_column("Objects", justify="right")
        table.add_column("Coverage", justify="right")
        table.add_column("", justify="left", min_width=14)  # coverage bar
        table.add_column("Mean Area", justify="right", style="muted")
        table.add_column("Compact.", justify="right", style="muted")
        table.add_column("Solidity", justify="right", style="muted")

        if has_supervised:
            table.add_column("Dice", justify="right", style="metric")
            table.add_column("IoU", justify="right", style="metric")

        for result in results:
            u = result.unsupervised_metrics
            if result.error:
                fail_row = [
                    Text("✗ " + result.method_name, style="bold red"),
                    Text("FAILED", style="bold red"),
                    "—",
                    Text("░" * 12, style="dim red"),
                    "—",
                    "—",
                    "—",
                ]
                if has_supervised:
                    fail_row += ["—", "—"]
                table.add_row(*fail_row)
                continue
            c_color = _coverage_color(u.coverage_ratio)
            bar = _bar(u.coverage_ratio)

            row = [
                result.method_name,
                str(u.num_objects),
                f"{u.coverage_ratio * 100:.1f}%",
                Text(bar, style=c_color),
                f"{u.mean_area:,.0f}",
                f"{u.mean_compactness:.3f}",
                f"{u.mean_solidity:.3f}",
            ]

            if has_supervised:
                s = result.supervised_metrics
                if s:
                    row += [f"{s.dice:.3f}", f"{s.iou:.3f}"]
                else:
                    row += ["—", "—"]

            table.add_row(*row)

        self.console.print(table, justify="left")

    # ------------------------------------------------------------------
    # Saved-files confirmation
    # ------------------------------------------------------------------

    def print_saved(self, output_dir: Path | str, files: list[str]) -> None:
        """Print a tidy summary of files written to disk."""
        output_dir = Path(output_dir)
        lines = "\n".join(f"  [good]✓[/good] [value]{f}[/value]" for f in files)
        self.console.print(
            Panel(
                lines,
                title=f"[muted]saved → {output_dir}[/muted]",
                border_style="dim",
                padding=(0, 2),
            )
        )
        self.console.print()

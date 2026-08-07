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

_THEME = Theme({"header": "bold cyan", "method": "bold magenta", "muted": "dim", "good": "bold green"})


class BenchmarkReporter:
    """Console reporting for the inference-only runner."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(theme=_THEME)

    def print_header(self, n_methods: int, image_path: Path | str) -> None:
        image_path = Path(image_path)
        self.console.print(Panel.fit(
            f"[header]segmenteer[/header] · runner\n\nImage: {image_path.name}\nMethods: {n_methods}",
            border_style="cyan",
        ))

    def print_dataset_header(self, n_methods: int, n_images: int) -> None:
        self.console.print(Panel.fit(
            f"[header]segmenteer[/header] · runner\n\nSlides: {n_images}\nMethods: {n_methods}\nRuns: {n_images * n_methods}",
            border_style="cyan",
        ))

    def print_image_header(self, image_path: Path | str, index: int, total: int) -> None:
        self.console.print(Rule(f"[header]{Path(image_path).name}[/header] [muted]{index}/{total}[/muted]", style="cyan"))

    def print_method_start(self, name: str, index: int, total: int) -> None:
        self.console.print(Text(f"[{index}/{total}] ", style="muted") + Text(name, style="method") + Text(" running…", style="muted"))

    def print_method_skipped(self, name: str, index: int, total: int) -> None:
        self.console.print(Text(f"[{index}/{total}] ", style="muted") + Text(name, style="method") + Text(" reused prediction", style="muted"))

    def print_method_done(self, result: BenchmarkResult) -> None:
        self.console.print(f"  [good]✓[/good] {result.execution_time:.2f}s  [muted]prediction saved[/muted]")

    def print_method_failed(self, name: str, error_msg: str) -> None:
        short = error_msg if len(error_msg) <= 120 else error_msg[:117] + "…"
        self.console.print(f"  [bold red]✗ {name}[/bold red] [dim red]{short}[/dim red]")

    def print_run_complete(self, n: int, n_failed: int = 0) -> None:
        self.console.print(Rule(f"{n - n_failed} completed, {n_failed} failed", style="dim"))

    def print_summary(self, results: list[BenchmarkResult]) -> None:
        table = Table(title="Runner summary", box=box.ROUNDED, border_style="cyan")
        table.add_column("Method")
        table.add_column("Time (s)", justify="right")
        table.add_column("s / Mpx", justify="right")
        table.add_column("Status")
        for result in results:
            table.add_row(
                result.method_name,
                f"{result.execution_time:.3f}",
                f"{result.seconds_per_pixel * 1_000_000:.4f}",
                "FAILED" if result.failed else "prediction",
            )
        self.console.print(table)

    def print_saved(self, output_dir: Path | str, files: list[str]) -> None:
        lines = "\n".join(f"  ✓ {item}" for item in files)
        self.console.print(Panel(lines, title=f"saved → {Path(output_dir)}", border_style="dim"))

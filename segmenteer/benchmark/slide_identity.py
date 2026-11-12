"""Validate slide identifiers used by predictions, timings, and evaluation.

Existing stems remain stable; case-insensitive collisions are rejected.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
from unicodedata import normalize


def validate_unique_slide_stems(
    slides: Iterable[str | Path], *, context: str = "slide cohort"
) -> None:
    """Reject slide sets with duplicate output identifiers."""
    by_stem: dict[str, list[str]] = defaultdict(list)
    for slide in slides:
        name = Path(slide).name
        key = normalize("NFC", Path(name).stem).casefold()
        by_stem[key].append(name)
    clashes = [names for names in by_stem.values() if len(names) > 1]
    if clashes:
        rendered = "; ".join(
            " / ".join(repr(name) for name in names) for names in clashes[:5]
        )
        remaining = f" (and {len(clashes) - 5} more)" if len(clashes) > 5 else ""
        raise ValueError(
            f"Ambiguous slide stems in {context}: {rendered}{remaining}. "
            "Predictions, timing records, and evaluation use the slide stem as an identifier. "
            "Rename the conflicting slides to distinct stems or use separate experiment directories; "
            "no outputs were written."
        )

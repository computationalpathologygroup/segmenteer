"""Resolve ground-truth filenames consistently across workflows.

Accept ``<stem>`` and ``<stem>_gt`` by default, or a configured suffix.
Ambiguous matches raise an error.
"""

from __future__ import annotations

from pathlib import Path

ANNOTATION_EXTENSIONS = (".geojson", ".xml", ".json")
DEFAULT_STEM_SUFFIXES = ("", "_gt")


def annotation_candidates(
    root: Path | str, stem: str, suffix: str | None = None
) -> list[Path]:
    root = Path(root)
    if not stem or Path(stem).name != stem or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid slide stem: {stem!r}")
    if suffix is None:
        return [
            root / f"{stem}{postfix}{extension}"
            for postfix in DEFAULT_STEM_SUFFIXES
            for extension in ANNOTATION_EXTENSIONS
        ]
    if (
        not isinstance(suffix, str)
        or "/" in suffix
        or "\\" in suffix
        or not suffix
        or suffix in {".", ".."}
    ):
        raise ValueError(
            "Annotation suffix must be a non-empty filename suffix without path separators"
        )
    extension = Path(suffix).suffix.casefold()
    if extension:
        if extension not in ANNOTATION_EXTENSIONS:
            raise ValueError(
                f"Unsupported annotation suffix {suffix!r}; expected .geojson, .json or .xml"
            )
        return [root / f"{stem}{suffix}"]
    return [root / f"{stem}{suffix}{ext}" for ext in ANNOTATION_EXTENSIONS]


def resolve_ground_truth(
    root: Path | str, stem: str, suffix: str | None = None
) -> Path | None:
    candidates = annotation_candidates(root, stem, suffix)
    matches = [path for path in candidates if path.is_file()]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous ground truth for {stem!r}: {', '.join(str(p) for p in matches)}. Remove duplicates or specify an annotation suffix."
        )
    return matches[0] if matches else None

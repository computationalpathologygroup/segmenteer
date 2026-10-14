from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_timestamped_output_dir(base_dir: str | Path = "outputs") -> Path:
    """Create a collision-safe result folder with second-level versioning."""
    root = Path(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

"""Post-run ensemble fusion for segmenteer benchmark outputs.

Fuses method outputs saved by :class:`EnsembleOutputWriter` via soft weighted
majority voting.  Edit the ``__main__`` block and run::

    python postensemble.py

Fusion logic lives in :mod:`segmenteer.methods.ensemble`.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401 — used in member_dirs / manifest_path

import segmenteer as seg

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seg.run_ensemble(
        manifest_path=None,  # or: Path("outputs/.../ensemble_manifest.json")
        threshold=0.5,       # 0.0 = union  |  0.5 = majority  |  1.0 = intersection
        max_size=2048,
        min_area=50,
        image_stem=None,
        # The output directory is the parent of the first member_dirs entry.
        # When using manifest_path instead, it is the manifest's parent directory.
        member_dirs=[
            # "outputs/2503_1322/entropy_masker__footprint=None_min-area=0_mpp=10",
            # "outputs/2503_1322/od_gmm_slide__epsilon=1e-06_min-area=10_morph-kernel-size=5_mpp=10_n-components=2_n-samples=100000",
            "outputs/2503_1624/hsv-threshold__lower=908103_min-area=10_mpp=10_upper=180255255",
            "outputs/2503_1624/trident_libtridentpathprofilersegmenter",
            "outputs/2503_1624/trident_grandqcsegmenter",
        ],
        # max_members=5,  # optional: keep only top-N methods by quality weight
    )

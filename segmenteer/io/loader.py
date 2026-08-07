import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Union

import geojson
import numpy as np

from segmenteer.core.utils import scale_geojson_coordinates
from segmenteer.io.asap import load_asap_xml
from segmenteer.io.atomic import write_json_atomic


def is_dicom_directory(path: Path) -> bool:
    if not path.is_dir():
        return False

    dicom_indicators = ["1_Map", "2", "3_0", "DICOMDIR"]

    dir_contents = [f.name for f in path.iterdir()]

    return any(indicator in dir_contents for indicator in dicom_indicators)


def load_dicom_wsi(path: Path, level: int = 0, verbose: bool = True) -> np.ndarray:
    from wsidicom import WsiDicom

    wsi = WsiDicom.open(path)

    num_levels = len(wsi.levels)

    if verbose:
        print("DICOM WSI Info:")
        print(f"  Available pyramid levels: {num_levels}")
        for i, lvl in enumerate(wsi.levels):
            print(f"    Level {i}: {lvl.size.width}x{lvl.size.height}")

    if level >= num_levels:
        original_level = level
        level = num_levels - 1
        if verbose:
            print(
                f"  Warning: Requested level {original_level} not available, using level {level}"
            )

    if verbose:
        print(
            f"  Loading level {level}: {wsi.levels[level].size.width}x{wsi.levels[level].size.height}"
        )

    level_size = wsi.levels[level].size
    size_tuple = (level_size.width, level_size.height)

    region = wsi.read_region((0, 0), level, size_tuple)

    image_array = np.array(region)

    wsi.close()

    return image_array


def load_openslide_wsi(path: Path, level: int = 0, verbose: bool = True) -> np.ndarray:
    try:
        import openslide
    except ImportError:
        raise ImportError(
            f"OpenSlide is required to read {path.suffix} files. "
            "Install with: pip install openslide-python openslide-bin"
        )

    slide = openslide.OpenSlide(str(path))

    num_levels = slide.level_count

    if verbose:
        print("OpenSlide WSI Info:")
        print(f"  Format: {slide.detect_format(str(path))}")
        print(f"  Available pyramid levels: {num_levels}")
        for i in range(num_levels):
            w, h = slide.level_dimensions[i]
            print(f"    Level {i}: {w}x{h}")

    if level >= num_levels:
        original_level = level
        level = num_levels - 1
        if verbose:
            print(
                f"  Warning: Requested level {original_level} not available, using level {level}"
            )

    if verbose:
        w, h = slide.level_dimensions[level]
        print(f"  Loading level {level}: {w}x{h}")

    level_dimensions = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, level_dimensions)
    image_array = np.array(region.convert("RGB"))

    slide.close()

    return image_array


def load_image(path: Union[str, Path], level: int = 0) -> np.ndarray:
    path = Path(path)

    if path.is_dir() and is_dicom_directory(path):
        return load_dicom_wsi(path, level=level, verbose=True)

    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        try:
            import tifffile
        except ImportError:
            raise ImportError(
                "tifffile is required to read TIFF files.\n"
                "Install the wsi extra: pip install 'segmenteer[wsi]'"
            ) from None
        return tifffile.imread(path)
    elif suffix == ".mrxs":
        try:
            return load_openslide_wsi(path, level=level, verbose=True)
        except ImportError:
            raise ImportError(
                "OpenSlide is required for MRXS files. "
                "Install with: pip install openslide-python openslide-bin"
            )
    else:
        try:
            import openslide

            if openslide.OpenSlide.detect_format(str(path)):
                return load_openslide_wsi(path, level=level, verbose=True)
        except Exception:
            pass

        try:
            import pyvips
        except ImportError:
            raise ImportError(
                "pyvips is required to read this image format.\n"
                "Install the wsi extra: pip install 'segmenteer[wsi]'"
            ) from None
        return np.array(pyvips.Image.new_from_file(path))


def save_geojson(geojson_data: dict, path: Union[str, Path], scale_factor: float = 1.0):
    """Persist GeoJSON atomically so readers never observe a partial file."""
    path = Path(path)

    if scale_factor != 1.0:
        geojson_data = scale_geojson_coordinates(geojson_data, scale_factor)

    write_json_atomic(path, geojson_data)


def load_geojson(path: Union[str, Path]) -> dict:
    path = Path(path)

    with open(path, "r") as f:
        return geojson.load(f)


@dataclass(frozen=True)
class GroundTruthPairing:
    """Pairing outcome for a set of WSI paths and external annotations.

    ``ground_truths`` contains only valid, non-empty annotations.  Missing
    files and files that become empty after group filtering are retained
    separately so callers can choose a policy without treating partial labels
    as a fatal error.
    """

    ground_truths: dict[Path, dict]
    missing: dict[Path, Path]
    empty: dict[Path, Path]

    @property
    def paired_images(self) -> list[Path]:
        """Images with a usable annotation, in deterministic path order."""
        return sorted(self.ground_truths, key=lambda path: str(path).casefold())

    @property
    def unpaired_images(self) -> list[Path]:
        """Images whose annotation was missing or unusable."""
        return sorted(
            {*self.missing, *self.empty}, key=lambda path: str(path).casefold()
        )

    @property
    def has_unpaired_images(self) -> bool:
        return bool(self.missing or self.empty)

    def as_manifest(self) -> dict:
        """Return JSON-safe, per-slide pairing information for run manifests."""
        return {
            "paired_images": [str(path) for path in self.paired_images],
            "missing_annotations": [
                {"image": str(image), "expected_annotation": str(annotation)}
                for image, annotation in sorted(
                    self.missing.items(), key=lambda item: str(item[0]).casefold()
                )
            ],
            "empty_annotations": [
                {"image": str(image), "annotation": str(annotation)}
                for image, annotation in sorted(
                    self.empty.items(), key=lambda item: str(item[0]).casefold()
                )
            ],
        }


def inspect_ground_truths(
    images: list,
    annotation_dir: Union[Path, None] = None,
    suffix: str = "_gt.geojson",
    groups: Iterable[str] | None = None,
) -> GroundTruthPairing:
    """Inspect ground-truth availability without imposing a run policy.

    Pairing is by identical filename stem.  A missing annotation and an empty
    annotation (including one emptied by ASAP group filtering) are distinct:
    both are unsuitable for supervised evaluation, but both remain eligible
    for prediction-only inference when requested by the caller.
    """
    ground_truths: dict[Path, dict] = {}
    missing: dict[Path, Path] = {}
    empty: dict[Path, Path] = {}

    for raw_image in images:
        image = Path(raw_image)
        search_dir = Path(annotation_dir) if annotation_dir is not None else image.parent
        annotation_path = search_dir / f"{image.stem}{suffix}"
        if not annotation_path.exists():
            missing[image] = annotation_path
            continue

        extension = annotation_path.suffix.casefold()
        if extension == ".xml":
            ground_truth = load_asap_xml(annotation_path, groups=groups)
        elif extension in {".geojson", ".json"}:
            ground_truth = load_geojson(annotation_path)
        else:
            raise ValueError(
                f"Unsupported annotation format for {annotation_path}. "
                "Supported formats are .xml, .geojson, and .json."
            )

        if not ground_truth.get("features"):
            empty[image] = annotation_path
            continue
        ground_truths[image] = ground_truth

    return GroundTruthPairing(ground_truths=ground_truths, missing=missing, empty=empty)


def format_ground_truth_pairing_error(
    pairing: GroundTruthPairing,
    *,
    limit: int = 8,
) -> str:
    """Compact error message for strict pairing without dumping huge path lists."""
    parts: list[str] = []
    if pairing.missing:
        examples = ", ".join(
            str(path)
            for path in list(
                sorted(pairing.missing.values(), key=lambda value: str(value).casefold())
            )[:limit]
        )
        suffix = " …" if len(pairing.missing) > limit else ""
        parts.append(f"{len(pairing.missing)} missing annotation file(s): {examples}{suffix}")
    if pairing.empty:
        examples = ", ".join(
            str(path)
            for path in list(
                sorted(pairing.empty.values(), key=lambda value: str(value).casefold())
            )[:limit]
        )
        suffix = " …" if len(pairing.empty) > limit else ""
        parts.append(
            f"{len(pairing.empty)} empty annotation file(s) after format/group filtering: "
            f"{examples}{suffix}"
        )
    if not parts:
        return "Ground-truth pairing is complete."
    return "Ground-truth pairing failed (" + "; ".join(parts) + ")"


def load_ground_truths(
    images: list,
    annotation_dir: Union[Path, None] = None,
    suffix: str = "_gt.geojson",
    groups: Iterable[str] | None = None,
    strict: bool = False,
) -> dict:
    """Load valid paired GeoJSON or ASAP XML ground truth for each image.

    ``strict=False`` (the default) returns the valid subset, allowing callers
    to run prediction-only inference on unlabeled slides.  ``strict=True``
    raises when any matching annotation is missing or empty.
    """
    pairing = inspect_ground_truths(
        images,
        annotation_dir=annotation_dir,
        suffix=suffix,
        groups=groups,
    )
    if strict and pairing.has_unpaired_images:
        raise FileNotFoundError(format_ground_truth_pairing_error(pairing))
    return pairing.ground_truths

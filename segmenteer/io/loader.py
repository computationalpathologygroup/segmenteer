import json
from pathlib import Path
from typing import Union

import geojson
import numpy as np

from segmenteer.core.utils import scale_geojson_coordinates


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
    path = Path(path)

    if scale_factor != 1.0:
        geojson_data = scale_geojson_coordinates(geojson_data, scale_factor)

    with open(path, "w") as f:
        json.dump(geojson_data, f, indent=2)


def load_geojson(path: Union[str, Path]) -> dict:
    path = Path(path)

    with open(path, "r") as f:
        return geojson.load(f)


def load_ground_truths(
    images: list,
    annotation_dir: Union[Path, None] = None,
    suffix: str = "_gt.geojson",
) -> dict:
    """Discover and load GeoJSON ground-truth annotations for a list of images.

    Looks for ``<image_stem><suffix>`` either next to each image (default) or
    inside *annotation_dir* when provided.  Images without a matching file are
    silently skipped and will run in unsupervised mode.

    Parameters
    ----------
    images:
        List of WSI :class:`~pathlib.Path` objects.
    annotation_dir:
        Directory containing annotation files.  Defaults to each image's own
        parent directory.
    suffix:
        Filename suffix appended to the image stem (default ``_gt.geojson``).

    Returns
    -------
    ``dict`` mapping each image :class:`~pathlib.Path` to its loaded GeoJSON
    ``dict``.  Only images that have a matching annotation are included.

    Examples
    --------
    Annotations next to images::

        gts = seg.load_ground_truths(images)

    Annotations in a separate folder::

        gts = seg.load_ground_truths(images, annotation_dir=Path("annotations/"))
    """
    ground_truths: dict = {}
    for img in images:
        img = Path(img)
        search_dir = Path(annotation_dir) if annotation_dir is not None else img.parent
        ann_path = search_dir / f"{img.stem}{suffix}"
        if ann_path.exists():
            ground_truths[img] = load_geojson(ann_path)
    return ground_truths

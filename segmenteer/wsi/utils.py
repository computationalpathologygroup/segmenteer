from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from skimage.transform import rescale


class WSIMetadata:
    def __init__(
        self,
        width: int,
        height: int,
        mpp: Optional[float] = None,
        num_levels: Optional[int] = None,
        level_dimensions: Optional[list] = None,
        level_downsamples: Optional[list] = None,
    ):
        self.width = width
        self.height = height
        self.mpp = mpp
        self.num_levels = num_levels
        self.level_dimensions = level_dimensions or []
        self.level_downsamples = level_downsamples or []

    def __repr__(self):
        return (
            f"WSIMetadata(width={self.width}, height={self.height}, "
            f"mpp={self.mpp}, num_levels={self.num_levels})"
        )


def get_wsi_metadata(path: Union[str, Path]) -> WSIMetadata:
    path = Path(path)

    if path.is_dir():
        return _get_dicom_metadata(path)

    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        return _get_tiff_metadata(path)
    elif suffix == ".mrxs":
        return _get_openslide_metadata(path)
    else:
        try:
            import openslide

            if openslide.OpenSlide.detect_format(str(path)):
                return _get_openslide_metadata(path)
        except:
            pass

        return _get_pillow_metadata(path)


def _get_dicom_metadata(path: Path) -> WSIMetadata:
    from wsidicom import WsiDicom

    wsi = WsiDicom.open(path)

    try:
        mpp_obj = wsi.mpp
        if hasattr(mpp_obj, "width") and hasattr(mpp_obj, "height"):
            mpp = (mpp_obj.width + mpp_obj.height) / 2.0
        else:
            mpp = float(mpp_obj)
    except:
        mpp = None

    level_dimensions = [(lvl.size.width, lvl.size.height) for lvl in wsi.levels]

    level_downsamples = [
        wsi.levels[0].size.width / lvl.size.width for lvl in wsi.levels
    ]

    metadata = WSIMetadata(
        width=wsi.levels[0].size.width,
        height=wsi.levels[0].size.height,
        mpp=mpp,
        num_levels=len(wsi.levels),
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
    )

    wsi.close()
    return metadata


def _get_tiff_metadata(path: Path) -> WSIMetadata:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        width = page.imagewidth
        height = page.imagelength

        mpp = None
        if page.tags.get("XResolution") and page.tags.get("YResolution"):
            x_res = page.tags["XResolution"].value
            y_res = page.tags["YResolution"].value

            unit = page.tags.get("ResolutionUnit")
            if unit and unit.value == 3:
                mpp = 10000.0 / x_res[0] * x_res[1]

        num_levels = len(tif.pages)
        level_dimensions = [(p.imagewidth, p.imagelength) for p in tif.pages]
        level_downsamples = [width / p.imagewidth for p in tif.pages]

        return WSIMetadata(
            width=width,
            height=height,
            mpp=mpp,
            num_levels=num_levels,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
        )


def _get_openslide_metadata(path: Path) -> WSIMetadata:
    try:
        import openslide
    except ImportError:
        raise ImportError(
            f"OpenSlide is required to read {path.suffix} files. "
            "Install with: pip install openslide-python"
        )

    slide = openslide.OpenSlide(str(path))

    width, height = slide.dimensions

    mpp = None
    if "openslide.mpp-x" in slide.properties:
        mpp = float(slide.properties["openslide.mpp-x"])

    num_levels = slide.level_count
    level_dimensions = slide.level_dimensions
    level_downsamples = slide.level_downsamples

    slide.close()

    return WSIMetadata(
        width=width,
        height=height,
        mpp=mpp,
        num_levels=num_levels,
        level_dimensions=list(level_dimensions),
        level_downsamples=list(level_downsamples),
    )


def _get_pillow_metadata(path: Path) -> WSIMetadata:
    img = Image.open(path)
    width, height = img.size

    mpp = None
    if "dpi" in img.info:
        dpi = img.info["dpi"][0]
        mpp = 25400.0 / dpi

    img.close()

    return WSIMetadata(width=width, height=height, mpp=mpp, num_levels=1)


def calculate_target_level(
    metadata: WSIMetadata, target_mpp: float
) -> Tuple[int, float]:
    if metadata.mpp is None:
        raise ValueError("WSI does not have MPP metadata")

    target_downsample = target_mpp / metadata.mpp

    if metadata.num_levels is None or metadata.num_levels == 1:
        return 0, target_downsample

    best_level = 0
    best_diff = float("inf")

    for i, downsample in enumerate(metadata.level_downsamples):
        diff = abs(downsample - target_downsample)
        if diff < best_diff:
            best_diff = diff
            best_level = i

    level_downsample = metadata.level_downsamples[best_level]
    additional_scale = target_downsample / level_downsample

    return best_level, additional_scale


def resample_to_mpp(
    image: np.ndarray, current_mpp: float, target_mpp: float
) -> np.ndarray:
    if current_mpp is None or target_mpp is None:
        raise ValueError("Both current_mpp and target_mpp must be specified")

    scale_factor = current_mpp / target_mpp

    if abs(scale_factor - 1.0) < 0.01:
        return image

    if image.ndim == 3:
        resampled = rescale(
            image,
            scale_factor,
            channel_axis=2,
            preserve_range=True,
            anti_aliasing=True,
        )
    else:
        resampled = rescale(
            image, scale_factor, preserve_range=True, anti_aliasing=True
        )

    return resampled.astype(image.dtype)


def load_wsi_at_mpp(
    path: Union[str, Path], target_mpp: float, verbose: bool = True
) -> Tuple[np.ndarray, WSIMetadata]:
    path = Path(path)
    metadata = get_wsi_metadata(path)

    if metadata.mpp is None:
        raise ValueError(
            f"Cannot determine MPP for {path}. Please provide MPP explicitly."
        )

    if verbose:
        print(f"WSI native MPP: {metadata.mpp:.3f}")
        print(f"Target MPP: {target_mpp:.3f}")

    if metadata.num_levels and metadata.num_levels > 1:
        level, additional_scale = calculate_target_level(metadata, target_mpp)

        if verbose:
            level_mpp = metadata.mpp * metadata.level_downsamples[level]
            print(f"Using pyramid level {level} (MPP: {level_mpp:.3f})")
            if abs(additional_scale - 1.0) > 0.01:
                print(f"Additional resampling: {additional_scale:.3f}x")

        if path.is_dir():
            image = _load_dicom_level(path, level)
        elif path.suffix.lower() in [".tif", ".tiff"]:
            image = _load_tiff_level(path, level)
        else:
            image = _load_openslide_level(path, level)

        current_mpp = metadata.mpp * metadata.level_downsamples[level]

        if abs(additional_scale - 1.0) > 0.01:
            image = resample_to_mpp(image, current_mpp, target_mpp)
    else:
        if path.is_dir():
            image = _load_dicom_level(path, 0)
        elif path.suffix.lower() in [".tif", ".tiff"]:
            import tifffile

            image = tifffile.imread(path)
        else:
            image = _load_openslide_level(path, 0)

        image = resample_to_mpp(image, metadata.mpp, target_mpp)

    if verbose:
        print(f"Final image size: {image.shape[0]}x{image.shape[1]}")

    return image, metadata


def _load_dicom_level(path: Path, level: int) -> np.ndarray:
    from wsidicom import WsiDicom

    wsi = WsiDicom.open(path)
    level_size = wsi.levels[level].size
    size_tuple = (level_size.width, level_size.height)
    region = wsi.read_region((0, 0), level, size_tuple)
    image_array = np.array(region)
    wsi.close()
    return image_array


def _load_tiff_level(path: Path, level: int) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        if level >= len(tif.pages):
            level = len(tif.pages) - 1
        return tif.pages[level].asarray()


def _load_openslide_level(path: Path, level: int) -> np.ndarray:
    try:
        import openslide
    except ImportError:
        raise ImportError(
            f"OpenSlide is required to read {path.suffix} files. "
            "Install with: pip install openslide-python"
        )

    slide = openslide.OpenSlide(str(path))

    if level >= slide.level_count:
        level = slide.level_count - 1

    level_dimensions = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, level_dimensions)
    image_array = np.array(region.convert("RGB"))
    slide.close()
    return image_array


def _load_pillow(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def estimate_mpp_from_magnification(magnification: float) -> float:
    mpp_map = {
        1: 10.0,
        2: 5.0,
        5: 2.0,
        7: 1.5,
        10: 1.0,
        20: 0.5,
        40: 0.25,
    }

    if magnification in mpp_map:
        return mpp_map[magnification]

    closest_mag = min(mpp_map.keys(), key=lambda x: abs(x - magnification))
    base_mpp = mpp_map[closest_mag]
    return base_mpp * (closest_mag / magnification)


def calculate_scale_factor_for_coordinates(
    source_mpp: float, target_mpp: float
) -> float:
    return source_mpp / target_mpp

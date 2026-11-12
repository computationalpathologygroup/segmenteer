from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
from skimage.transform import resize
from segmenteer.wsi.reader import NativeWSIReader


class WSIMetadata:
    def __init__(
        self,
        width: int,
        height: int,
        mpp: Optional[float] = None,
        num_levels: Optional[int] = None,
        level_dimensions: Optional[list] = None,
        level_downsamples: Optional[list] = None,
        mpp_x: Optional[float] = None,
        mpp_y: Optional[float] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.mpp_x = mpp if mpp_x is None else mpp_x
        self.mpp_y = mpp if mpp_y is None else mpp_y
        self.mpp = self.mpp_x if self.mpp_x == self.mpp_y else None
        self.num_levels = num_levels
        self.level_dimensions = level_dimensions or []
        self.level_downsamples = level_downsamples or []

    def __repr__(self):
        return f"WSIMetadata(width={self.width}, height={self.height}, mpp_x={self.mpp_x}, mpp_y={self.mpp_y}, num_levels={self.num_levels})"


def get_wsi_metadata(path: Union[str, Path]) -> WSIMetadata:
    path = Path(path)
    if path.is_dir():
        return _get_dicom_metadata(path)
    try:
        reader = NativeWSIReader()
        wsi = reader.read(path)
    except Exception:
        return _get_pillow_metadata(path)
    if wsi.backend == "openslide":
        slide = reader._open_openslide(path)
        try:
            level_dimensions = [tuple(dim) for dim in slide.level_dimensions]
        finally:
            slide.close()
    else:
        with reader._open_tiff(path) as tif:
            level_dimensions = [
                reader._level_size(level) for level in reader._tiff_levels(tif)
            ]
    try:
        mpp_x, mpp_y = reader.get_mpp(wsi, 0)
    except ValueError as exc:
        if "Cannot determine level-0 MPP" not in str(exc):
            raise
        mpp_x = mpp_y = None
    return _metadata_from_levels(level_dimensions, mpp_x, mpp_y)


def _metadata_from_levels(level_dimensions, mpp_x=None, mpp_y=None):
    width, height = level_dimensions[0]
    return WSIMetadata(
        width=width,
        height=height,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        num_levels=len(level_dimensions),
        level_dimensions=list(level_dimensions),
        level_downsamples=[width / w for w, _ in level_dimensions],
    )


def _get_dicom_metadata(path: Path) -> WSIMetadata:
    from wsidicom import WsiDicom

    wsi = WsiDicom.open(path)
    try:
        spacing = wsi.mpp
        if hasattr(spacing, "width") and hasattr(spacing, "height"):
            mpp_x, mpp_y = float(spacing.width), float(spacing.height)
        else:
            mpp_x = mpp_y = float(spacing)
        mpp_x, mpp_y = NativeWSIReader._normalise_target_mpp((mpp_x, mpp_y))
        levels = [
            (int(level.size.width), int(level.size.height)) for level in wsi.levels
        ]
        return _metadata_from_levels(levels, mpp_x, mpp_y)
    finally:
        wsi.close()


def _get_tiff_metadata(path: Path) -> WSIMetadata:
    reader = NativeWSIReader(backend="tifffile")
    wsi = reader.read(path)
    with reader._open_tiff(path) as tif:
        levels = [reader._level_size(level) for level in reader._tiff_levels(tif)]
    try:
        mpp_x, mpp_y = reader.get_mpp(wsi)
    except ValueError as exc:
        if "Cannot determine level-0 MPP" not in str(exc):
            raise
        mpp_x = mpp_y = None
    return _metadata_from_levels(levels, mpp_x, mpp_y)


def _get_openslide_metadata(path: Path) -> WSIMetadata:
    reader = NativeWSIReader(backend="openslide")
    wsi = reader.read(path)
    slide = reader._open_openslide(path)
    try:
        levels = [tuple(dim) for dim in slide.level_dimensions]
    finally:
        slide.close()
    try:
        mpp_x, mpp_y = reader.get_mpp(wsi)
    except ValueError as exc:
        if "Cannot determine level-0 MPP" not in str(exc):
            raise
        mpp_x = mpp_y = None
    return _metadata_from_levels(levels, mpp_x, mpp_y)


def _get_pillow_metadata(path: Path) -> WSIMetadata:
    with Image.open(path) as img:
        width, height = img.size
        dpi = img.info.get("dpi")
        mpp_x = mpp_y = None
        if dpi is not None:
            try:
                mpp_x, mpp_y = 25400 / float(dpi[0]), 25400 / float(dpi[1])
                mpp_x, mpp_y = NativeWSIReader._normalise_target_mpp((mpp_x, mpp_y))
            except (ValueError, IndexError, TypeError, ZeroDivisionError):
                mpp_x = mpp_y = None
    return _metadata_from_levels([(width, height)], mpp_x, mpp_y)


def calculate_target_level(
    metadata: WSIMetadata, target_mpp: float | tuple[float, float]
) -> Tuple[int, float | tuple[float, float]]:
    if metadata.mpp_x is None or metadata.mpp_y is None:
        raise ValueError("WSI does not have calibrated X/Y MPP metadata.")
    target_x, target_y = NativeWSIReader._normalise_target_mpp(target_mpp)
    sizes = metadata.level_dimensions or [(metadata.width, metadata.height)]
    size = NativeWSIReader._target_size(
        metadata.width,
        metadata.height,
        metadata.mpp_x,
        metadata.mpp_y,
        target_x,
        target_y,
    )
    level = NativeWSIReader._choose_level(sizes, size)
    width, height = sizes[level]
    ratio_x = target_x / (metadata.mpp_x * metadata.width / width)
    ratio_y = target_y / (metadata.mpp_y * metadata.height / height)
    additional_scale = ratio_x if ratio_x == ratio_y else (ratio_x, ratio_y)
    return (level, additional_scale)


def resample_to_mpp(
    image: np.ndarray,
    current_mpp: float | tuple[float, float],
    target_mpp: float | tuple[float, float],
) -> np.ndarray:
    current_x, current_y = NativeWSIReader._normalise_target_mpp(current_mpp)
    target_x, target_y = NativeWSIReader._normalise_target_mpp(target_mpp)
    width, height = NativeWSIReader._target_size(
        image.shape[1], image.shape[0], current_x, current_y, target_x, target_y
    )
    if (width, height) == (image.shape[1], image.shape[0]):
        return image
    output_shape = (height, width) + image.shape[2:]
    resized = resize(
        image,
        output_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=(width < image.shape[1] or height < image.shape[0]),
    )
    return resized.astype(image.dtype)


def load_wsi_at_mpp(
    path: Union[str, Path],
    target_mpp: float | tuple[float, float],
    verbose: bool = True,
) -> Tuple[np.ndarray, WSIMetadata]:
    path = Path(path)
    metadata = get_wsi_metadata(path)
    if metadata.mpp_x is None or metadata.mpp_y is None:
        raise ValueError(f"Cannot determine calibrated X/Y MPP for {path}.")
    target_x, target_y = NativeWSIReader._normalise_target_mpp(target_mpp)
    if verbose:
        print(f"WSI native MPP: X={metadata.mpp_x:.6g}, Y={metadata.mpp_y:.6g}")
        print(f"Target MPP: X={target_x:.6g}, Y={target_y:.6g}")
    if path.is_dir():
        level, _ = calculate_target_level(metadata, (target_x, target_y))
        image = _load_dicom_level(path, level)
        level_width, level_height = metadata.level_dimensions[level]
        current = (
            metadata.mpp_x * metadata.width / level_width,
            metadata.mpp_y * metadata.height / level_height,
        )
        image = resample_to_mpp(image, current, (target_x, target_y))
    else:
        reader = NativeWSIReader()
        try:
            wsi = reader.read(path)
        except (OSError, ValueError, RuntimeError):
            with Image.open(path) as source:
                image = resample_to_mpp(
                    np.asarray(source),
                    (metadata.mpp_x, metadata.mpp_y),
                    (target_x, target_y),
                )
        else:
            image = reader.get_wsi_at_mpp(wsi, (target_x, target_y))
    if verbose:
        achieved_x = metadata.width * metadata.mpp_x / image.shape[1]
        achieved_y = metadata.height * metadata.mpp_y / image.shape[0]
        print(
            f"Final image size: {image.shape[0]}x{image.shape[1]}, effective MPP: X={achieved_x:.6g}, Y={achieved_y:.6g}"
        )
    return (image, metadata)


def _load_dicom_level(path: Path, level: int) -> np.ndarray:
    from wsidicom import WsiDicom

    wsi = WsiDicom.open(path)
    try:
        level_size = wsi.levels[level].size
        return np.asarray(
            wsi.read_region((0, 0), level, (level_size.width, level_size.height))
        )
    finally:
        wsi.close()


def _load_tiff_level(path: Path, level: int) -> np.ndarray:
    reader = NativeWSIReader(backend="tifffile")
    with reader._open_tiff(path) as tif:
        levels = reader._tiff_levels(tif)
        return reader._normalise_tiff_array(
            levels[level].asarray(), getattr(levels[level], "axes", "")
        )


def _load_openslide_level(path: Path, level: int) -> np.ndarray:
    reader = NativeWSIReader(backend="openslide")
    slide = reader._open_openslide(path)
    try:
        return np.asarray(
            slide.read_region((0, 0), level, slide.level_dimensions[level]).convert(
                "RGB"
            )
        )
    finally:
        slide.close()


def _load_pillow(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def estimate_mpp_from_magnification(magnification: float) -> float:
    magnification = float(magnification)
    NativeWSIReader._positive_mpp(magnification, "Magnification")
    mpp_map = {1: 10.0, 2: 5.0, 5: 2.0, 7: 1.5, 10: 1.0, 20: 0.5, 40: 0.25}
    if magnification in mpp_map:
        return mpp_map[magnification]
    closest_mag = min(mpp_map.keys(), key=lambda x: abs(x - magnification))
    return mpp_map[closest_mag] * closest_mag / magnification


def calculate_scale_factor_for_coordinates(
    source_mpp: float | tuple[float, float], target_mpp: float | tuple[float, float]
) -> float | tuple[float, float]:
    source_x, source_y = NativeWSIReader._normalise_target_mpp(source_mpp)
    target_x, target_y = NativeWSIReader._normalise_target_mpp(target_mpp)
    ratio_x, ratio_y = source_x / target_x, source_y / target_y
    return ratio_x if ratio_x == ratio_y else (ratio_x, ratio_y)

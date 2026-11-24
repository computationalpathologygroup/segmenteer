from __future__ import annotations

import numpy as np
import geojson
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import mapping, shape
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import closing, opening, disk
from skimage.segmentation import watershed

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.utils import mask_to_geojson


class MorphologicalSegmenter(NumpySegmenter):
    def __init__(
        self,
        mpp: float | None = 10,
        disk_size: int = 3,
        *args,
        algorithm_revision: int = 2,
        disk_radius_um: float | None = None,
        **kwargs,
    ):
        if mpp is None:
            mpp = 10
        if algorithm_revision != 2:
            raise ValueError("Unsupported MorphologicalSegmenter algorithm revision.")
        if (
            isinstance(disk_size, (bool, np.bool_))
            or not isinstance(disk_size, (int, np.integer))
            or disk_size < 0
        ):
            raise ValueError("disk_size must be a nonnegative integer.")
        super().__init__(*args, mpp=mpp, **kwargs)
        self.disk_size = int(disk_size)
        if disk_radius_um is not None and (
            not np.isfinite(float(disk_radius_um)) or float(disk_radius_um) < 0
        ):
            raise ValueError("disk_radius_um must be finite and nonnegative.")
        self.disk_radius_um = (
            float(disk_radius_um) if disk_radius_um is not None else None
        )
        self.algorithm_revision = 2

    @property
    def name(self) -> str:
        return (
            f"morphological_radius{self.disk_radius_um:g}um"
            if self.disk_radius_um is not None
            else f"morphological_disk{self.disk_size}"
        )

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError("MorphologicalSegmenter requires a grayscale image.")
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=bool)
        tissue = image <= threshold_otsu(image)
        if self.disk_radius_um is not None:
            mpp_x, mpp_y = getattr(self, "_current_mpp", (self.mpp, self.mpp))
            rx = int(np.ceil(self.disk_radius_um / mpp_x))
            ry = int(np.ceil(self.disk_radius_um / mpp_y))
            yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
            footprint = (xx * mpp_x) ** 2 + (yy * mpp_y) ** 2 <= self.disk_radius_um**2
        else:
            if self.disk_size == 0:
                return tissue
            footprint = disk(self.disk_size)
        return closing(opening(tissue, footprint=footprint), footprint=footprint)


class WatershedTissueSegmenter(NumpySegmenter):
    def __init__(
        self,
        mpp: float | None = 10,
        min_distance_um: float = 100.0,
        *args,
        merge_regions: bool = True,
        algorithm_revision: int = 3,
        **kwargs,
    ):
        if mpp is None:
            mpp = 10
        if algorithm_revision != 3:
            raise ValueError("Unsupported WatershedTissueSegmenter algorithm revision.")
        if not isinstance(merge_regions, (bool, np.bool_)):
            raise ValueError("merge_regions must be a boolean.")
        mpp = float(mpp)
        min_distance_um = float(min_distance_um)
        if not np.isfinite(mpp) or mpp <= 0:
            raise ValueError(f"mpp must be a positive finite number; got {mpp!r}.")
        if not np.isfinite(min_distance_um) or min_distance_um <= 0:
            raise ValueError(
                f"min_distance_um must be a positive finite number; got {min_distance_um!r}."
            )
        super().__init__(*args, mpp=mpp, **kwargs)
        self.min_distance_um = min_distance_um
        self.merge_regions = bool(merge_regions)
        self.algorithm_revision = 3

    @property
    def name(self) -> str:
        return f"watershed_tissue_mindist{self.min_distance_um:g}um"

    @property
    def min_distance_px(self) -> int:
        return max(1, round(self.min_distance_um / self.mpp))

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError("WatershedTissueSegmenter requires a grayscale image.")
        if image.size == 0 or image.min() == image.max():
            return np.zeros(image.shape, dtype=np.int32)
        tissue = image <= threshold_otsu(image)
        if not tissue.any():
            return np.zeros(image.shape, dtype=np.int32)
        distance = ndi.distance_transform_edt(tissue)
        connected, component_count = ndi.label(
            tissue, structure=np.ones((3, 3), dtype=bool)
        )
        peaks = peak_local_max(
            distance,
            min_distance=self.min_distance_px,
            labels=tissue,
            exclude_border=False,
        )
        markers = np.zeros(image.shape, dtype=np.int32)
        marker_id = 0
        for row, col in peaks:
            marker_id += 1
            markers[row, col] = marker_id
        marked_components = np.unique(connected[markers > 0])
        marked_components = set(marked_components[marked_components > 0].tolist())
        missing = [
            component_id
            for component_id in range(1, component_count + 1)
            if component_id not in marked_components
        ]
        if missing:
            positions = ndi.maximum_position(distance, labels=connected, index=missing)
            for position in positions:
                marker_id += 1
                markers[position] = marker_id
        return watershed(-distance, markers, mask=tissue, connectivity=2).astype(
            np.int32, copy=False
        )

    def _convert_to_geojson(self, wsi, labels: np.ndarray) -> geojson.FeatureCollection:
        if self.merge_regions:
            return super()._convert_to_geojson(wsi, labels > 0)
        width, height = self.reader.get_size(wsi, 0)
        scale_x = labels.shape[1] / width
        scale_y = labels.shape[0] / height
        min_area = self._minimum_area_pixels(wsi, labels.shape)
        features = []
        for region in regionprops(labels):
            if region.area < min_area:
                continue
            row, col = region.bbox[:2]
            cropped = mask_to_geojson(
                region.image, min_area=0, scaling_factor=(scale_x, scale_y)
            )
            for feature in cropped.get("features", []):
                geometry = shape(feature["geometry"])
                geometry = affinity.translate(
                    geometry, xoff=col / scale_x, yoff=row / scale_y
                )
                feature["geometry"] = mapping(geometry)
                features.append(feature)
        return geojson.FeatureCollection(features)

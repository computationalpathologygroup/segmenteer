import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_opening, binary_closing
from skimage.measure import label
from segmenteer.core.utils import mask_to_geojson
from segmenteer.io.loader import Image
import geojson


class ODGMMSlideSegmenter:
    def __init__(
        self,
        level: int = 1,
        n_samples: int = 100000,
        n_components: int = 2,
        min_area: int = 10,
        morph_kernel_size: int = 5,
        epsilon: float = 1e-6,
    ):
        self.level = level
        self.n_samples = n_samples
        self.n_components = n_components
        self.min_area = min_area
        self.morph_kernel_size = morph_kernel_size
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return "od_gmm_slide"

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        image_float = image.astype(np.float32)
        od = -np.log((image_float + self.epsilon) / (255.0 + self.epsilon))
        return od

    def _od_to_sum(self, od: np.ndarray) -> np.ndarray:
        return np.sum(od, axis=-1)

    def segment(self, image: Image) -> geojson.FeatureCollection:
        image = image.get_numpy_image(level=self.level)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB (H, W, 3)")

        h, w, _ = image.shape
        total_pixels = h * w

        n_samples = min(self.n_samples, total_pixels)
        indices = np.random.choice(total_pixels, size=n_samples, replace=False)
        row_indices = indices // w
        col_indices = indices % w

        sampled_pixels = image[row_indices, col_indices]
        od_samples = self._rgb_to_od(sampled_pixels)
        od_sum_samples = self._od_to_sum(od_samples).reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=42,
            max_iter=100,
        )
        gmm.fit(od_sum_samples)

        means = gmm.means_.flatten()
        background_cluster = np.argmin(means)

        od_full = self._rgb_to_od(image)
        od_sum_full = self._od_to_sum(od_full).reshape(-1, 1)
        posteriors = gmm.predict_proba(od_sum_full)
        posteriors = posteriors.reshape(h, w, self.n_components)

        tissue_mask = posteriors[:, :, background_cluster] < 0.5

        if self.morph_kernel_size > 0:
            struct_elem = np.ones(
                (self.morph_kernel_size, self.morph_kernel_size), dtype=bool
            )
            tissue_mask = binary_opening(tissue_mask, structure=struct_elem)
            tissue_mask = binary_closing(tissue_mask, structure=struct_elem)

        labeled_mask = label(tissue_mask)
        unique_labels, counts = np.unique(labeled_mask, return_counts=True)

        for lbl, count in zip(unique_labels, counts):
            if lbl == 0:
                continue
            if count < self.min_area:
                tissue_mask[labeled_mask == lbl] = False

        return mask_to_geojson(tissue_mask, self.min_area)

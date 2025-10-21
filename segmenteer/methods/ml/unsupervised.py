import numpy as np
from skimage import color, filters, measure, feature
from skimage.filters import rank
from skimage.morphology import disk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from segmenteer.core.utils import mask_to_geojson


class UnsupervisedClusteringSegmenter:
    def __init__(self, patch_size: int = 64, n_clusters: int = 3, min_area: int = 10):
        self.patch_size = patch_size
        self.n_clusters = n_clusters
        self.min_area = min_area
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    @property
    def name(self) -> str:
        return f"unsupervised_kmeans_p{self.patch_size}_k{self.n_clusters}"

    def extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0 or np.std(patch) < 1e-6:
            return np.zeros(35)

        if patch.dtype == np.uint8:
            patch_float = patch.astype(np.float32) / 255.0
        else:
            patch_float = np.clip(patch, 0, 1)

        features = []

        if len(patch.shape) == 3 and patch.shape[2] == 3:
            rgb_mean = np.mean(patch_float, axis=(0, 1))
            rgb_std = np.std(patch_float, axis=(0, 1))
            features.extend(rgb_mean)
            features.extend(rgb_std)

            try:
                hsv = color.rgb2hsv(patch_float)
                hsv_mean = np.mean(hsv, axis=(0, 1))
                hsv_std = np.std(hsv, axis=(0, 1))
                features.extend(hsv_mean)
                features.extend(hsv_std)
            except:
                features.extend([0.0] * 6)

            try:
                lab = color.rgb2lab(patch_float)
                lab_mean = np.mean(lab, axis=(0, 1))
                features.extend(lab_mean)
            except:
                features.extend([0.0] * 3)

            gray = color.rgb2gray(patch_float)
        else:
            features.extend([0.0] * 15)
            gray = (
                patch_float if len(patch_float.shape) == 2 else patch_float.mean(axis=2)
            )

        gray = np.clip(gray, 0, 1)
        gray_uint8 = (gray * 255).astype(np.uint8)

        intensity_mean = np.mean(gray)
        intensity_std = np.std(gray)
        intensity_skew = stats.skew(gray.flatten()) if len(gray.flatten()) > 1 else 0.0
        features.extend([intensity_mean, intensity_std, intensity_skew])

        if np.std(gray_uint8) > 1:
            try:
                lbp = feature.local_binary_pattern(
                    gray_uint8, P=8, R=1, method="uniform"
                )
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
                lbp_hist = lbp_hist.astype(float)
                if lbp_hist.sum() > 0:
                    lbp_hist /= lbp_hist.sum()
                features.extend(lbp_hist[:5])
            except:
                features.extend([0.0] * 5)

            try:
                contrast = rank.enhance_contrast(gray_uint8, disk(3))
                features.append(np.mean(contrast))
            except:
                features.append(0.0)
        else:
            features.extend([0.0] * 6)

        try:
            edges = filters.sobel(gray)
            features.append(np.mean(edges))
            features.append(np.std(edges))
        except:
            features.extend([0.0, 0.0])

        try:
            threshold = filters.threshold_otsu(gray)
            binary = gray < threshold
            tissue_fraction = np.sum(binary) / binary.size

            labels = measure.label(binary)
            num_regions = len(np.unique(labels)) - 1

            features.extend([tissue_fraction, num_regions])
        except:
            features.extend([0.5, 0])

        features_array = np.array(features[:35])
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)

        if len(features_array) < 35:
            features_array = np.pad(features_array, (0, 35 - len(features_array)))

        return features_array[:35]

    def extract_patches(self, image: np.ndarray):
        h, w = image.shape[:2]
        patches = []
        coordinates = []

        step = self.patch_size

        for i in range(0, h - self.patch_size + 1, step):
            for j in range(0, w - self.patch_size + 1, step):
                patch = image[i : i + self.patch_size, j : j + self.patch_size]
                if patch.shape[:2] == (self.patch_size, self.patch_size):
                    patches.append(patch)
                    coordinates.append((i, j))

        return patches, coordinates

    def segment(self, image: np.ndarray) -> dict:
        patches, coords = self.extract_patches(image)

        if len(patches) == 0:
            return {"type": "FeatureCollection", "features": []}

        features = np.array([self.extract_patch_features(p) for p in patches])

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        feature_vars = np.var(features, axis=0)
        non_constant = feature_vars > 1e-8

        if np.any(non_constant):
            features = features[:, non_constant]
        else:
            features = np.random.randn(*features.shape) * 0.1

        features_scaled = self.scaler.fit_transform(features)

        labels = self.clusterer.fit_predict(features_scaled)

        cluster_mapping = {}
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]

            intensity_idx = 15 if features.shape[1] > 15 else 0
            saturation_idx = 4 if features.shape[1] > 4 else 0

            mean_intensity = (
                np.mean(cluster_features[:, intensity_idx])
                if cluster_features.shape[1] > intensity_idx
                else 0.5
            )
            mean_saturation = (
                np.mean(cluster_features[:, saturation_idx])
                if cluster_features.shape[1] > saturation_idx
                else 0.5
            )

            if mean_intensity > 0.75 and mean_saturation < 0.15:
                cluster_mapping[cluster_id] = "background"
            else:
                cluster_mapping[cluster_id] = "tissue"

        tissue_mask = np.zeros(image.shape[:2], dtype=bool)

        for label, (i, j) in zip(labels, coords):
            if cluster_mapping.get(label, "tissue") == "tissue":
                tissue_mask[i : i + self.patch_size, j : j + self.patch_size] = True

        return mask_to_geojson(tissue_mask, self.min_area)

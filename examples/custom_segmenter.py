import segmenteer as seg
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel


class EdgeBasedSegmenter:
    def __init__(self, threshold: float = 0.1, min_area: int = 10):
        self.threshold = threshold
        self.min_area = min_area

    @property
    def name(self) -> str:
        return f"edge_threshold{self.threshold}"

    def segment(self, image: np.ndarray) -> dict:
        if image.ndim == 3:
            gray = rgb2gray(image)
        else:
            gray = image

        edges = sobel(gray)
        mask = edges > self.threshold
        return seg.mask_to_geojson(mask, self.min_area)


def main():
    image = seg.load_image("image.tiff")

    segmenters = [
        seg.OtsuSegmenter(),
        EdgeBasedSegmenter(threshold=0.05),
        EdgeBasedSegmenter(threshold=0.10),
        EdgeBasedSegmenter(threshold=0.15),
    ]

    runner = seg.BenchmarkRunner()
    results = runner.run_multiple(segmenters, image)

    for result in results:
        u = result.unsupervised_metrics
        print(
            f"{result.method_name}: {result.execution_time:.6f}s, {u.num_objects} objects"
        )
        seg.save_geojson(result.geojson, f"output_{result.method_name}.geojson")


if __name__ == "__main__":
    main()

from pathlib import Path

from segmenteer.core.base import NumpySegmenter


def get_model_cache_dir() -> Path:
    cache_dir = Path(__file__).parent.parent.parent.parent / "models" / "rtlucassen"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class RTLucassenSlideSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 7.04,  # 7.04 mpp corresponds to 1.25x magnification described in paper.
        device: str = "cuda",
        *args,
        **kwargs,
    ):
        try:
            from slidesegmenter import SlideSegmenter
        except ImportError:
            raise ImportError(
                "slidesegmenter is required for RTLucassenSlideSegmenter.\n"
                "Install with: pip install 'segmenteer[rtlucassen]'"
            ) from None
        super().__init__(*args, **kwargs)
        self.mpp = mpp
        self.segmenter = SlideSegmenter(
            device=device,
            alternative_directory=get_model_cache_dir(),
            pen_marking_segmentation=False,
        )

    @property
    def name(self) -> str:
        return "rtlucassen_slidesegmenter"

    def _segment_numpy(self, image):
        normalized = image / 255
        segmentation = self.segmenter.segment(normalized)["tissue"].astype(bool)[..., 0]
        return segmentation

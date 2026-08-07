from pathlib import Path

from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import get_method_model_dir


def get_model_cache_dir() -> Path:
    """Return the shared local SlideSegmenter weights directory."""
    return get_method_model_dir("rtlucassen")


class RTLucassenSlideSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 7.04,  # 7.04 mpp corresponds to 1.25x magnification described in paper.
        device: str | None = None,
        *args,
        **kwargs,
    ):
        try:
            from slidesegmenter import SlideSegmenter
        except ImportError:
            raise ImportError(
                "slidesegmenter is required for RTLucassenSlideSegmenter.\n"
                "Install with: uv sync --extra rtlucassen"
            ) from None
        # SlideSegmenter does not support MPS, so retain CUDA-or-CPU selection.
        device = resolve_torch_device(device, allow_mps=False)
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

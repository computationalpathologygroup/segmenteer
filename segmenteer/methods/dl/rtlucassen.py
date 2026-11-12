from segmenteer.benchmark.profiling import measure_phase
from pathlib import Path
from segmenteer.core.base import NumpySegmenter
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.model_cache import get_method_model_dir
from segmenteer.external_assets import require_network


def get_model_cache_dir() -> Path:
    return get_method_model_dir("rtlucassen")


class RTLucassenSlideSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self, mpp: float | None = 7.04, device: str | None = None, *args, **kwargs
    ):
        device = resolve_torch_device(device, allow_mps=False)
        super().__init__(*args, **kwargs)
        if mpp is None:
            mpp = 7.04
        self.mpp = mpp
        self.device = device
        self.segmenter = None

    @property
    def name(self) -> str:
        return "rtlucassen_slidesegmenter"

    def _segment_numpy(self, image):
        if self.segmenter is None:
            try:
                from slidesegmenter import SlideSegmenter
            except ImportError as exc:
                raise ImportError(
                    "slidesegmenter is required for RTLucassenSlideSegmenter. Install with: uv sync --extra rtlucassen"
                ) from exc
            # Allow SlideSegmenter to use its existing local weight cache.
            cache = get_model_cache_dir()
            has_local_weights = any(
                path.is_file()
                for suffix in ("*.pth", "*.pt", "*.ckpt")
                for path in cache.rglob(suffix)
            )
            if not has_local_weights:
                require_network(
                    "SlideSegmenter-managed model loading (potential external fetch)",
                    pinned=False,
                )
            with measure_phase("model_initialization"):
                self.segmenter = SlideSegmenter(
                    device=self.device,
                    alternative_directory=get_model_cache_dir(),
                    pen_marking_segmentation=False,
                )
        normalized = image / 255
        segmentation = self.segmenter.segment(normalized)["tissue"].astype(bool)[..., 0]
        return segmentation


def __getattr__(name):
    if name == "LIBTRIDENTRTLucassenSlideSegmenterSegmenter":
        from segmenteer.methods.dl.rtl_trident import (
            LIBTRIDENTRTLucassenSlideSegmenterSegmenter,
        )

        return LIBTRIDENTRTLucassenSlideSegmenterSegmenter
    raise AttributeError(name)

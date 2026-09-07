from pathlib import Path

from segmenteer.core.runtime import resolve_torch_device
from segmenteer.core.utils import mask_to_geojson
from segmenteer.model_cache import find_local_model, get_method_model_dir


_MODEL_NAMESPACE = "atlaspatch-sam2"
_MODEL_FILE = "model.pth"


def get_model_cache_dir() -> Path:
    """Return the shared local AtlasPatch/SAM2 weights directory."""
    return get_method_model_dir(_MODEL_NAMESPACE)


class AtlasPatchSAM2Segmenter:
    def __init__(
        self,
        device: str | None = None,
        mask_threshold: float = 0.5,
        min_area: int = 0,
        checkpoint_path: str | Path | None = None,
    ):
        try:
            import atlas_patch  # noqa: F401
            import sam2  # noqa: F401
        except ImportError:
            raise ImportError(
                "AtlasPatchSAM2Segmenter requires the project-managed "
                "AtlasPatch + SAM 2 extra.\n"
                "Install with: uv sync --extra atlaspatch-sam2\n"
                "or, for every supported method: uv sync --extra all"
            ) from None
        self.device = resolve_torch_device(device)
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"AtlasPatch checkpoint_path does not exist: {self.checkpoint_path}"
            )
        self._service = None

    @property
    def name(self) -> str:
        return "atlaspatch_sam2"

    def _local_checkpoint_path(self) -> Path | None:
        """Prefer a project-managed model file over AtlasPatch auto-download."""
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        return find_local_model(_MODEL_NAMESPACE, _MODEL_FILE)

    def _get_service(self):
        if self._service is None:
            import atlas_patch
            from atlas_patch.core.config import SegmentationConfig
            from atlas_patch.services.segmentation import \
                SAM2SegmentationService

            pkg_dir = Path(atlas_patch.__file__).parent
            config_path = pkg_dir / "configs" / "sam2.1_hiera_t.yaml"
            checkpoint_path = self._local_checkpoint_path()
            if checkpoint_path is not None:
                print(f"Loading AtlasPatch weights from {checkpoint_path}")
            else:
                print(
                    "AtlasPatch weights were not found under "
                    f"{get_model_cache_dir()}; AtlasPatch will use its managed cache/download."
                )
            cfg = SegmentationConfig(
                checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
                config_path=config_path,
                device=self.device,
            ).validated()
            self._service = SAM2SegmentationService(cfg)
        return self._service

    def segment(self, path: Path):
        from atlas_patch.core.wsi import WSIFactory

        wsi = WSIFactory.load(path)
        try:
            wsi_width, wsi_height = wsi.get_size(lv=0)
            service = self._get_service()
            mask_result = service.segment_thumbnail(wsi)
        finally:
            wsi.cleanup()

        # mask_result.data: float32 (H, W), aspect ratio preserved (no squaring)
        mask = mask_result.data >= self.mask_threshold

        # scaling_factor < 1: maps mask pixel coords → level-0 WSI coords
        scaling_factor = mask.shape[1] / wsi_width  # mask_width / wsi_width
        return mask_to_geojson(mask, self.min_area, scaling_factor=scaling_factor)

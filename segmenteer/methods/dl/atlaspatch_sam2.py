from pathlib import Path

from segmenteer.core.utils import mask_to_geojson


class AtlasPatchSAM2Segmenter:
    def __init__(
        self,
        device: str | None = None,
        mask_threshold: float = 0.5,
        min_area: int = 10,
    ):
        try:
            import atlas_patch  # noqa: F401
        except ImportError:
            raise ImportError(
                "atlas-patch and sam2 are required for AtlasPatchSAM2Segmenter.\n"
                "Install with: uv pip install atlas-patch 'git+https://github.com/facebookresearch/sam2.git'"
            ) from None
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self._service = None

    @property
    def name(self) -> str:
        return "atlaspatch_sam2"

    def _get_service(self):
        if self._service is None:
            import atlas_patch
            from atlas_patch.core.config import SegmentationConfig
            from atlas_patch.services.segmentation import \
                SAM2SegmentationService

            pkg_dir = Path(atlas_patch.__file__).parent
            config_path = pkg_dir / "configs" / "sam2.1_hiera_t.yaml"
            cfg = SegmentationConfig(
                checkpoint_path=None,
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

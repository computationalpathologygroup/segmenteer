import time
from segmenteer.benchmark.profiling import measure_phase
from pathlib import Path
from segmenteer.core.runtime import resolve_torch_device
from segmenteer.core.utils import mask_to_geojson, MASK_POLYGONIZATION_PROTOCOL
from segmenteer.model_cache import find_local_model, get_method_model_dir
from segmenteer.external_assets import require_network

_MODEL_NAMESPACE = "atlaspatch-sam2"
_MODEL_FILE = "model.pth"


def get_model_cache_dir() -> Path:
    return get_method_model_dir(_MODEL_NAMESPACE)


class AtlasPatchSAM2Segmenter:
    def __init__(
        self,
        device: str | None = None,
        mask_threshold: float = 0.5,
        min_area: int = 0,
        checkpoint_path: str | Path | None = None,
    ):
        start = time.perf_counter()
        try:
            import atlas_patch
            import sam2
        except ImportError:
            raise ImportError(
                "AtlasPatchSAM2Segmenter requires the project-managed AtlasPatch + SAM 2 extra.\nInstall with: uv sync --extra atlaspatch-sam2\nor, for every supported method: uv sync --extra all"
            ) from None
        self.device = resolve_torch_device(device)
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        if self.checkpoint_path is not None and (not self.checkpoint_path.is_file()):
            raise FileNotFoundError(
                f"AtlasPatch checkpoint_path does not exist: {self.checkpoint_path}"
            )
        self._service = None
        self._constructor_time_s = time.perf_counter() - start

    @property
    def prediction_protocol(self) -> str:
        return MASK_POLYGONIZATION_PROTOCOL

    @property
    def name(self) -> str:
        return "atlaspatch_sam2"

    def _local_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        return find_local_model(_MODEL_NAMESPACE, _MODEL_FILE)

    def _get_service(self):
        if self._service is None:
            import atlas_patch
            from atlas_patch.core.config import SegmentationConfig
            from atlas_patch.services.segmentation import SAM2SegmentationService

            pkg_dir = Path(atlas_patch.__file__).parent
            config_path = pkg_dir / "configs" / "sam2.1_hiera_t.yaml"
            checkpoint_path = self._local_checkpoint_path()
            if checkpoint_path is not None:
                print(f"Loading AtlasPatch weights from {checkpoint_path}")
            else:
                require_network(
                    "AtlasPatch-managed model download (unversioned)", pinned=False
                )
                print(
                    f"AtlasPatch weights were not found under {get_model_cache_dir()}; AtlasPatch will use its managed cache/download."
                )
            cfg = SegmentationConfig(
                checkpoint_path=str(checkpoint_path)
                if checkpoint_path is not None
                else None,
                config_path=config_path,
                device=self.device,
            ).validated()
            self._service = SAM2SegmentationService(cfg)
        return self._service

    def segment(self, path: Path):
        from atlas_patch.core.wsi import WSIFactory

        with measure_phase("image_loading"):
            wsi = WSIFactory.load(path)
        try:
            with measure_phase("image_loading"):
                wsi_width, wsi_height = wsi.get_size(lv=0)
            if self._service is None:
                with measure_phase("model_initialization"):
                    service = self._get_service()
            else:
                service = self._get_service()
            with measure_phase("algorithm"):
                mask_result = service.segment_thumbnail(wsi)
        finally:
            wsi.cleanup()
        with measure_phase("polygon_conversion"):
            mask = mask_result.data >= self.mask_threshold
            scaling_factor = (mask.shape[1] / wsi_width, mask.shape[0] / wsi_height)
            return mask_to_geojson(mask, self.min_area, scaling_factor=scaling_factor)

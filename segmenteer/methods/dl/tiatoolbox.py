    # >>> from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
    # >>> segmentor = SemanticSegmentor(model="efficientunet-tissue_mask")
    # >>> results = segmentor.run(
    # ...     ["/example_wsi.svs"],
    # ...     masks=None,
    # ...     auto_get_mask=False,
    # ...     patch_mode=False,
    # ...     save_dir=Path("/tissue_mask/"),
    # ...     output_type="annotationstore",
    # ... )

from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor

from segmenteer.core.base import PathSegmenter

class TIAToolboxSegmenter(PathSegmenter):
    def name() -> str:
        return "tiatoolbox"

    def _segment_path(self, image_path, output_path):
        segmentor = SemanticSegmentor(model="efficientunet-tissue_mask")
        save_dir = output_path.parent / "output"
        results = segmentor.run(
            [image_path],
            masks=None,
            auto_get_mask=False,
            patch_mode=False,
            save_dir=save_dir,
            output_type="dict",
            device="cuda",
        )
        print(results)
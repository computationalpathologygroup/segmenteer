from segmenteer.core.base import NumpySegmenter


class BigPictureSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 8,  # 8 mpp is recommended https://github.com/imi-bigpicture/tissue-segmentation/tree/6d97a25a8255f591eb2c705611a32a5f56101a25/tissue_segmentation.
        dilation_disk_size: int = 32,
        confidence_threshold: float = 0.8,
        dilate_mask: bool = True,
        apply_hole_filling: bool = True,
        select_largest_tissue_objects: bool = False,  # This is different from the default because we'd like to not automatically discard small tissue objects.
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        from tissue_segmentation import create_tissue_mask

        self.mpp = mpp
        self.segmenter_func = create_tissue_mask
        self.dilation_disk_size = dilation_disk_size
        self.confidence_threshold = confidence_threshold
        self.dilate_mask = dilate_mask
        self.apply_hole_filling = apply_hole_filling
        self.select_largest_tissue_objects = select_largest_tissue_objects

    @property
    def name(self) -> str:
        return "bigpicture"

    def _segment_numpy(self, image):
        return self.segmenter_func(
            image,
            dilation_disk_size=self.dilation_disk_size,
            confidence_threshold=self.confidence_threshold,
            dilate_mask=self.dilate_mask,
            apply_hole_filling=self.apply_hole_filling,
            select_largest_tissue_objects=self.select_largest_tissue_objects,
        )

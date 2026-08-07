from __future__ import annotations

import warnings

import numpy as np

from segmenteer.core.base import NumpySegmenter


class BigPictureSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 8,
        dilation_disk_size: int = 32,
        confidence_threshold: float = 0.8,
        dilate_mask: bool = True,
        apply_hole_filling: bool = True,
        select_largest_tissue_objects: bool = False,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Segment tissue using the BIGPICTURE tissue-segmentation model.

        An MPP of 8 µm/pixel is recommended by the upstream implementation:
        https://github.com/imi-bigpicture/tissue-segmentation/tree/6d97a25a8255f591eb2c705611a32a5f56101a25/tissue_segmentation

        ``select_largest_tissue_objects`` defaults to False here because
        Segmenteer should not automatically discard disconnected tissue
        fragments.
        """
        super().__init__(*args, **kwargs)
        self.device = device

        # TensorFlow/Metal can otherwise select an Apple GPU implicitly.
        # The benchmark entry point passes device="cpu" so BigPicture follows
        # the same timing policy as the other wrappers. Leaving device unset
        # keeps the upstream TensorFlow default for library users.
        if device is not None and device.casefold() == "cpu":
            try:
                import tensorflow as tf
            except ModuleNotFoundError:
                tf = None

            if tf is not None:
                try:
                    tf.config.set_visible_devices([], "GPU")
                except RuntimeError as exc:
                    raise RuntimeError(
                        "TensorFlow devices were initialised before BigPicture "
                        "could enforce CPU-only execution. Start a fresh Python "
                        "process."
                    ) from exc

        try:
            from tissue_segmentation import create_tissue_mask
        except ModuleNotFoundError as exc:
            if exc.name != "tissue_segmentation":
                raise

            raise ModuleNotFoundError(
                "BigPictureSegmenter requires the optional BigPicture "
                "dependencies. From the project root, run "
                "`uv sync --extra bigpicture` (or `uv sync --extra all`) "
                "and then rerun with `uv run python run.py`."
            ) from exc

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

    @staticmethod
    def _match_mask_to_input_shape(
        mask: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Make the BigPicture mask match the image passed to the model.

        The upstream implementation can occasionally return a mask whose
        dimensions differ slightly because of integer rounding during model
        output rescaling and padding.

        Extra pixels are removed from the bottom/right edges. Missing pixels
        are padded with background at the bottom/right edges. This preserves
        the top-left slide-coordinate origin.
        """
        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                "BigPicture returned an invalid mask shape. "
                f"Expected a two-dimensional mask, received {mask.shape}."
            )

        target_height, target_width = image.shape[:2]
        mask_height, mask_width = mask.shape

        if (mask_height, mask_width) == (target_height, target_width):
            return mask.astype(bool, copy=False)

        warnings.warn(
            "BigPicture returned a mask whose dimensions did not match its "
            f"input image: mask={mask.shape}, "
            f"input={(target_height, target_width)}. "
            "The mask was cropped or padded at its bottom and right edges.",
            RuntimeWarning,
            stacklevel=2,
        )

        normalized_mask = np.zeros(
            (target_height, target_width),
            dtype=bool,
        )

        copy_height = min(mask_height, target_height)
        copy_width = min(mask_width, target_width)

        normalized_mask[:copy_height, :copy_width] = mask[
            :copy_height,
            :copy_width,
        ].astype(bool, copy=False)

        return normalized_mask

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        mask = self.segmenter_func(
            image,
            dilation_disk_size=self.dilation_disk_size,
            confidence_threshold=self.confidence_threshold,
            dilate_mask=self.dilate_mask,
            apply_hole_filling=self.apply_hole_filling,
            select_largest_tissue_objects=(
                self.select_largest_tissue_objects
            ),
        )

        return self._match_mask_to_input_shape(mask, image)
from __future__ import annotations
from segmenteer.benchmark.profiling import measure_phase
import warnings
import numpy as np
from segmenteer.core.base import NumpySegmenter


class BigPictureSegmenter(NumpySegmenter):
    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float | None = 8,
        dilation_disk_size: int = 32,
        confidence_threshold: float = 0.8,
        dilate_mask: bool = True,
        apply_hole_filling: bool = True,
        select_largest_tissue_objects: bool = False,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = device
        if mpp is None:
            mpp = 8
        self.mpp = mpp
        self.segmenter_func = None
        self.dilation_disk_size = dilation_disk_size
        self.confidence_threshold = confidence_threshold
        self.dilate_mask = dilate_mask
        self.apply_hole_filling = apply_hole_filling
        self.select_largest_tissue_objects = select_largest_tissue_objects

    def _get_segmenter_func(self):
        if self.segmenter_func is None:
            if self.device is not None and self.device.casefold() == "cpu":
                try:
                    import tensorflow as tf
                except ModuleNotFoundError:
                    tf = None
                if tf is not None:
                    try:
                        tf.config.set_visible_devices([], "GPU")
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "TensorFlow devices were initialised before BigPicture could enforce CPU-only execution. Start a fresh Python process."
                        ) from exc
            try:
                from tissue_segmentation import create_tissue_mask
            except ModuleNotFoundError as exc:
                if exc.name != "tissue_segmentation":
                    raise
                raise ModuleNotFoundError(
                    "BigPictureSegmenter requires the optional BigPicture dependencies. From the project root, run `uv sync --extra bigpicture` (or `uv sync --extra all`) and then rerun with `uv run python run.py`."
                ) from exc
            self.segmenter_func = create_tissue_mask
        return self.segmenter_func

    @property
    def name(self) -> str:
        return "bigpicture"

    @staticmethod
    def _match_mask_to_input_shape(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(
                f"BigPicture returned an invalid mask shape. Expected a two-dimensional mask, received {mask.shape}."
            )
        target_height, target_width = image.shape[:2]
        mask_height, mask_width = mask.shape
        if (mask_height, mask_width) == (target_height, target_width):
            return mask.astype(bool, copy=False)
        warnings.warn(
            f"BigPicture returned a mask whose dimensions did not match its input image: mask={mask.shape}, input={(target_height, target_width)}. The mask was cropped or padded at its bottom and right edges.",
            RuntimeWarning,
            stacklevel=2,
        )
        normalized_mask = np.zeros((target_height, target_width), dtype=bool)
        copy_height = min(mask_height, target_height)
        copy_width = min(mask_width, target_width)
        normalized_mask[:copy_height, :copy_width] = mask[
            :copy_height, :copy_width
        ].astype(bool, copy=False)
        return normalized_mask

    @staticmethod
    def _empty_upstream_selection_mask(exc: ValueError) -> np.ndarray | None:
        """Recover only the known empty-component failure in upstream BigPicture.

        In the upstream ``create_tissue_mask``, ``ndimage.label(mask)`` returns
        ``num_objects == 0`` for an empty thresholded mask. The subsequent
        ``np.argmax(np.diff(...))`` then raises. Inspect that exact traceback
        frame rather than swallowing unrelated ``argmax`` failures from the
        network or other libraries. No tissue has been predicted in this case.
        """
        if str(exc) != "attempt to get argmax of an empty sequence":
            return None
        tb = exc.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            if (
                frame.f_code.co_name == "create_tissue_mask"
                and frame.f_locals.get("num_objects") == 0
            ):
                mask = frame.f_locals.get("mask")
                if isinstance(mask, np.ndarray) and mask.ndim == 2 and not np.any(mask):
                    return np.zeros(mask.shape, dtype=bool)
            tb = tb.tb_next
        return None

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        if self.segmenter_func is None:
            with measure_phase("model_initialization"):
                func = self._get_segmenter_func()
        else:
            func = self._get_segmenter_func()
        try:
            mask = func(
                image,
                dilation_disk_size=self.dilation_disk_size,
                confidence_threshold=self.confidence_threshold,
                dilate_mask=self.dilate_mask,
                apply_hole_filling=self.apply_hole_filling,
                select_largest_tissue_objects=self.select_largest_tissue_objects,
            )
        except ValueError as exc:
            recovered = (
                self._empty_upstream_selection_mask(exc)
                if self.select_largest_tissue_objects
                else None
            )
            if recovered is None:
                raise
            warnings.warn(
                "BigPicture detected no tissue components; its upstream "
                "largest-object selection fails on an empty mask. "
                "Returning the empty prediction without rerunning the network.",
                RuntimeWarning,
                stacklevel=2,
            )
            mask = recovered
        return self._match_mask_to_input_shape(mask, image)

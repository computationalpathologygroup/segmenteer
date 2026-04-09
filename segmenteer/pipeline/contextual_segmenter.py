"""ContextualSegmenter — bridges context definers and prompt-based segmenters.

A :class:`ContextualSegmenter` behaves like any other segmenter in the
benchmark framework (it satisfies the :class:`~segmenteer.core.Segmenter`
protocol) but, before running inference on each image, it asks a
:class:`~segmenteer.pipeline.context.ContextDefiner` to produce a
natural-language prompt and injects that prompt into the wrapped segmenter.

Because the wrapped segmenter is pre-instantiated (model weights already
loaded), this approach has zero per-image model-loading overhead.  Only the
prompt string is updated between images.

This module deliberately contains no deep-learning imports.  It only relies
on the public ``segment`` method defined on every segmenter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geojson

if TYPE_CHECKING:
    from segmenteer.pipeline.context import ContextDefiner

logger = logging.getLogger(__name__)

# Maps the attribute names used by each prompt-aware segmenter class to the
# canonical slot we will write.  Extend this dict if new segmenter types are
# added that use a different attribute.
_PROMPT_ATTR_CANDIDATES = ("text_prompt", "prompt")


def _set_prompt(segmenter: object, prompt: str) -> None:
    """Write *prompt* into the first recognised prompt attribute of *segmenter*."""
    for attr in _PROMPT_ATTR_CANDIDATES:
        if hasattr(segmenter, attr):
            setattr(segmenter, attr, prompt)
            return


class ContextualSegmenter:
    """Pairs a context definer with a prompt-based segmenter.

    On each call to :meth:`segment`:

    1. The :class:`~segmenteer.pipeline.context.ContextDefiner` examines the
       WSI and returns a natural-language prompt string.
    2. That prompt is injected into the inner segmenter's prompt attribute.
    3. The inner segmenter's :meth:`segment` method is called and its result
       returned unchanged.

    The inner segmenter's model weights are loaded exactly once (at
    construction time) and reused across all images.

    Parameters
    ----------
    segmenter:
        A pre-instantiated prompt-capable segmenter, such as
        :class:`~segmenteer.methods.dl.CONCHGradCAMSegmenter`,
        :class:`~segmenteer.methods.dl.FastSAMSegmenter`, or
        :class:`~segmenteer.methods.dl.SAM3Segmenter`.
    context_definer:
        Any object implementing the
        :class:`~segmenteer.pipeline.context.ContextDefiner` protocol.

    Examples
    --------
    >>> definer  = OllamaContextDefiner(model="llava", image_mpp=20)
    >>> inner    = seg.CONCHGradCAMSegmenter(mpp=10)
    >>> combined = ContextualSegmenter(inner, definer)
    >>> seg.run_dataset([combined], images)
    """

    def __init__(self, segmenter: object, context_definer: "ContextDefiner") -> None:
        # Validate at construction time that the segmenter is prompt-capable,
        # so the error surfaces immediately rather than on the first image.
        if not any(hasattr(segmenter, attr) for attr in _PROMPT_ATTR_CANDIDATES):
            raise TypeError(
                f"{type(segmenter).__name__!r} does not expose a prompt attribute "
                f"({' or '.join(_PROMPT_ATTR_CANDIDATES)}).  "
                "Only prompt-based segmenters (e.g. CONCHGradCAMSegmenter, "
                "FastSAMSegmenter, SAM3Segmenter) can be used inside a "
                "ContextualSegmenter."
            )
        self._segmenter = segmenter
        self._context_definer = context_definer
        self._last_prompt: str | None = None

    # ---------------------------------------------------------------------- #
    # Segmenter protocol                                                       #
    # ---------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Compound name used for output directory naming.

        Format: ``<segmenter_name>[<definer_name>]``
        """
        inner_name = getattr(self._segmenter, "name", type(self._segmenter).__name__)
        return f"{inner_name}[{self._context_definer.name}]"

    def segment(self, path: Path) -> geojson.FeatureCollection:
        """Segment *path* using a context-definer-generated prompt.

        Parameters
        ----------
        path:
            Path to the WSI file.

        Returns
        -------
        geojson.FeatureCollection
            Segmentation result from the inner segmenter.
        """
        path = Path(path)
        prompt = self._context_definer.define(path)
        logger.debug("%s: prompt from definer = %r", path.name, prompt)
        self._last_prompt = prompt
        _set_prompt(self._segmenter, prompt)
        return self._segmenter.segment(path)

    # ---------------------------------------------------------------------- #
    # Convenience                                                              #
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"ContextualSegmenter("
            f"segmenter={self._segmenter!r}, "
            f"context_definer={self._context_definer!r})"
        )

"""Context definers for two-stage segmentation pipelines.

A :class:`ContextDefiner` examines a WSI and produces a natural-language
prompt that downstream prompt-based segmenters (CONCH, FastSAM, SAM3) will
use as their text prompt.

The canonical implementation is :class:`OllamaContextDefiner`, which reads a
downsampled thumbnail from the WSI, attaches it to an Ollama vision-model
request, and returns the model's response as the prompt string.  A vision-
capable model (e.g. *llava*, *llava-phi3*, *moondream*) must be used.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocol                                                                     #
# --------------------------------------------------------------------------- #


@runtime_checkable
class ContextDefiner(Protocol):
    """Minimal interface for context-definer objects.

    A context definer ingests a WSI path and returns a natural-language
    string that describes the slide's content.  That string is then injected
    as the ``text_prompt`` / ``prompt`` argument of a prompt-based segmenter.
    """

    @property
    def name(self) -> str:
        """Short, filesystem-safe identifier used in output directory names."""
        ...

    def define(self, path: Path) -> str:
        """Return a text description / prompt for *path*.

        Parameters
        ----------
        path:
            Absolute path to a WSI file.

        Returns
        -------
        str
            Natural-language prompt for the downstream segmenter.
        """
        ...


# --------------------------------------------------------------------------- #
# OllamaContextDefiner                                                         #
# --------------------------------------------------------------------------- #


class OllamaContextDefiner:
    """Query a locally-running Ollama server to generate a segmentation prompt.

    The definer can operate in two modes:

    * **Text-only** (``image_mpp=None``): sends only slide metadata (filename,
      MPP) and the user prompt to the model.
    * **Vision** (``image_mpp`` is a float): reads a downsampled thumbnail
      from the WSI at the requested MPP, encodes it as a JPEG / PNG, and
      attaches it to the Ollama request.  Use a vision-capable model such as
      *llava*, *llava-phi3*, or *moondream* in this mode.

    Parameters
    ----------
    model:
        Ollama model tag for a vision-capable model, e.g. ``"llava"``,
        ``"llava-phi3"``, ``"moondream"``.
    system_prompt:
        System message that sets the model's role and output constraints.
        Defaults to a concise histopathology assistant persona instructing the
        model to output a single, short tissue-description phrase.
    user_prompt_template:
        User message template.  The following placeholders are substituted at
        runtime:

        * ``{filename}``  — basename of the WSI file
        * ``{stem}``      — basename without extension
        * ``{mpp}``       — the MPP used for the thumbnail

    host:
        Base URL of the Ollama REST API.  Defaults to ``http://localhost:11434``.
    image_mpp:
        MPP at which to downsample the WSI before sending to the model.
        Lower values → higher resolution thumbnail; 20 is a sensible default
        (roughly 0.5× objective magnification for a 40× scanner).
    options:
        Additional Ollama model options forwarded verbatim (e.g.
        ``{"temperature": 0.1, "num_predict": 32}``).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llava",
        system_prompt: str = (
            "You are a concise histopathology assistant. "
            "Given a whole-slide image thumbnail, output a short phrase "
            "(3-8 words) that describes the primary tissue type visible, "
            "suitable for use as a segmentation prompt.  "
            "Output ONLY the phrase — no punctuation, no explanation."
        ),
        user_prompt_template: str = (
            "Describe the main tissue visible in this slide ({filename})."
        ),
        host: str = "http://localhost:11434",
        image_mpp: float = 20.0,
        options: dict | None = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.host = host.rstrip("/")
        self.image_mpp = image_mpp
        self.options: dict = options or {}
        self.timeout = timeout

        self._validate_ollama_available()

    # ---------------------------------------------------------------------- #
    # Protocol implementation                                                  #
    # ---------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        return f"ollama_{self.model}_mpp{self.image_mpp}"

    def define(self, path: Path) -> str:
        """Query Ollama and return the generated prompt string.

        Parameters
        ----------
        path:
            Path to the WSI file.

        Returns
        -------
        str
            Stripped response from the Ollama model.
        """
        path = Path(path)
        user_message = self.user_prompt_template.format(
            filename=path.name,
            stem=path.stem,
            mpp=str(self.image_mpp),
        )

        thumbnail_b64 = self._read_thumbnail_b64(path)
        if thumbnail_b64 is None:
            raise RuntimeError(
                f"Could not read a thumbnail from {path} at mpp={self.image_mpp}. "
                "A visual thumbnail is required — OllamaContextDefiner cannot run "
                "without image input."
            )
        prompt = self._call_ollama(user_message, thumbnail_b64)
        logger.info("OllamaContextDefiner [%s] → '%s'", path.name, prompt)
        return prompt

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _validate_ollama_available(self) -> None:
        """Raise an informative error if the ``ollama`` package is absent."""
        try:
            import ollama  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required for OllamaContextDefiner.\n"
                "Install with:  pip install ollama"
            ) from None

    def _read_thumbnail_b64(self, path: Path) -> str | None:
        """Return a base-64-encoded JPEG thumbnail of the WSI at ``self.image_mpp``."""
        from segmenteer.wsi import load_wsi_at_mpp
        from PIL import Image as PILImage

        image_np, _ = load_wsi_at_mpp(path, target_mpp=self.image_mpp)
        pil_img = PILImage.fromarray(image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _call_ollama(self, user_message: str, image_b64: str) -> str:
        """Send a vision chat request to Ollama and return the assistant's reply."""
        import ollama

        messages: list[dict] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": user_message, "images": [image_b64]})

        client = ollama.Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=messages,
            options=self.options if self.options else None,
        )

        return response.message.content.strip()

    def __repr__(self) -> str:
        return (
            f"OllamaContextDefiner(model={self.model!r}, "
            f"image_mpp={self.image_mpp}, host={self.host!r})"
        )

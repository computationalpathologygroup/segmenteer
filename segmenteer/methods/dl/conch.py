"""CONCH VLM-based tissue segmenter using patch-token cosine similarity.

CONCH (CONtrastive learning from Captions for Histopathology) is a
Vision-Language Model trained on histopathology image-caption pairs
(Lu et al., Nature Medicine 2024).

Approach
--------
1. The downsampled WSI is letterbox-resized to CONCH's 448×448 input
   without any cropping — the entire slide is visible to the model.
2. Patch token features from the last transformer block are projected into
   the shared CLIP embedding space and compared with the text prompt via
   cosine similarity.  This gives a 28×28 spatial similarity map directly
   measuring "how tissue-like is each region" — more principled than GradCAM
   for dense prediction.
3. The map is upsampled back to the original image resolution accounting for
   letterbox padding.
4. Thresholding uses Otsu's method by default (``activation_threshold=None``)
   so no manual tuning is needed across different slides and prompts.

Install the CONCH package before use:
    uv sync --extra conch

Weights must be downloaded manually from https://huggingface.co/MahmoodLab/conch
and placed at models/conch/pytorch_model.bin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from segmenteer.core.base import NumpySegmenter

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _best_device() -> str:
    if not _TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_WEIGHTS_PATH = (
    Path(__file__).parent.parent.parent.parent / "models" / "conch" / "pytorch_model.bin"
)


def _load_conch(weights_path: str | None, device: str):
    """Load CONCH model and return (model, tokenizer, preprocess_transform)."""
    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    except ImportError:
        raise ImportError(
            "The 'conch' package is required for CONCHGradCAMSegmenter.\n"
            "Install with: uv sync --extra conch"
        ) from None

    checkpoint = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"CONCH weights not found at {checkpoint}.\n"
            "Download pytorch_model.bin from https://huggingface.co/MahmoodLab/conch\n"
            "(accept the licence on HuggingFace first, then download manually)\n"
            f"and place it at:  {checkpoint}"
        )

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", str(checkpoint), device=device
    )
    model.eval()

    raw_tokenizer = get_tokenizer()
    # Probe to detect tokenizer type: open_clip tokenizers return a plain
    # torch.Tensor; the CONCH HF-wrapped tokenizer returns a BatchEncoding.
    _probe = raw_tokenizer(["test"])
    tok_is_hf = not isinstance(_probe, torch.Tensor)

    return model, raw_tokenizer, tok_is_hf, preprocess


def _embed_text(model, tokenizer, tok_is_hf: bool, prompts: list[str], device: str):
    """Return L2-normalised text embeddings, shape (N, D)."""
    with torch.no_grad():
        if tok_is_hf:
            out = tokenizer(
                prompts, padding=True, truncation=True,
                max_length=77, return_tensors="pt",
            )
            tokens = out["input_ids"].to(device)
            mask = out.get("attention_mask")
            mask = mask.to(device) if mask is not None else None
        else:
            tokens = tokenizer(prompts).to(device)
            mask = None

        kwargs = {"attention_mask": mask} if mask is not None else {}
        try:
            embs = model.encode_text(tokens, **kwargs)
        except TypeError:
            embs = model.encode_text(tokens)
        embs = F.normalize(embs, dim=-1)
    return embs  # (N, D)


def _find_last_visual_block(model):
    """Return the last transformer block of CONCH's visual encoder.

    Uses the same exhaustive named-module search as prompters.py so it works
    for both timm-backed (visual.trunk.blocks) and open_clip
    (visual.transformer.resblocks) layouts.
    """
    candidates = []
    for name, mod in model.named_modules():
        if "visual" not in name and "vision" not in name:
            continue
        if not isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        if len(mod) < 2:
            continue
        last = mod[-1]
        if hasattr(last, "attn") or hasattr(last, "self_attn") or hasattr(last, "attention"):
            candidates.append((name.count("."), mod))

    if not candidates:
        raise RuntimeError(
            "Could not locate the last transformer block in CONCH's visual encoder."
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return list(candidates[0][1])[-1]


def _patch_text_similarity(model, img_tensor, text_emb, block) -> np.ndarray:
    """Compute per-patch cosine similarity between patch tokens and text embedding.

    CONCH's visual encoder uses AttentionalPooler (attn_pool_contrast) to
    aggregate patch tokens into a single 512-dim CLIP embedding.  We replicate
    that projection per-patch (treating each patch as if it were the only
    attended value) to obtain a spatial similarity map:

        patch_feats  (N, 768)
          → ln_k normalisation
          → V-projection from MHA  (N, 512)
          → out_proj               (N, 512)
          → @ proj_contrast        (N, 512)   [same op as in encode_image]
          → L2-normalise
          → cosine sim with text_emb
    """
    captured: list = [None]

    def _fwd(module, inp, out):
        raw = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(raw, torch.Tensor):
            captured[0] = raw.detach()

    h = block.register_forward_hook(_fwd)
    try:
        with torch.no_grad():
            model.encode_image(img_tensor, proj_contrast=True, normalize=True)
    finally:
        h.remove()

    if captured[0] is None:
        return np.zeros((1, 1), dtype=np.float32)

    # Drop CLS token → (N_patches, D_model=768)
    patch_feats = captured[0][0, 1:, :].float()

    visual = model.visual
    apc = visual.attn_pool_contrast        # AttentionalPooler
    mha = apc.attn                         # nn.MultiheadAttention

    with torch.no_grad():
        # Step 1: layer-norm (same as attn_pool_contrast applies before attention)
        patch_norm = apc.ln_k(patch_feats)                                  # (N, 768)

        # Step 2: V projection  – weight shape (512, 768), F.linear does x @ w.T
        patch_v = F.linear(patch_norm, mha.v_proj_weight)                   # (N, 512)

        # Step 3: out_proj of MHA  – weight (512, 512)
        patch_out = F.linear(patch_v, mha.out_proj.weight, mha.out_proj.bias)  # (N, 512)

        # Step 4: proj_contrast – stored as nn.Parameter (512, 512), applied as x @ W
        patch_clip = patch_out @ visual.proj_contrast                       # (N, 512)

        patch_clip = F.normalize(patch_clip, dim=-1)
        sim = (patch_clip @ text_emb.squeeze(0)).cpu().numpy()              # (N,)

    G = int(round(sim.shape[0] ** 0.5))
    return sim[: G * G].reshape(G, G).astype(np.float32)  # (G, G)


# --------------------------------------------------------------------------- #
# Segmenter                                                                    #
# --------------------------------------------------------------------------- #

MODEL_SIZE = 448   # CONCH native input resolution
PATCH_GRID = 28    # = MODEL_SIZE / patch_size(16)


class CONCHGradCAMSegmenter(NumpySegmenter):
    """Segment tissue in histopathology WSIs using CONCH patch-token similarity.

    The downsampled WSI is **letterbox**-resized to 448×448 (no cropping, so
    the full slide is visible to the model), then per-patch cosine similarities
    between the CONCH patch token embeddings and the text prompt are computed.
    The resulting 28×28 similarity map is upsampled back to the original
    resolution and thresholded.  By default Otsu's method is used to find the
    threshold automatically (``activation_threshold=None``).

    Parameters
    ----------
    mpp:
        Resolution at which the WSI is read (microns per pixel).
    prompt:
        Text description of the tissue class you want to segment.
    activation_threshold:
        Cosine-similarity threshold above which a pixel is classified as
        tissue.  Pass ``None`` (default) to use Otsu's method automatically.
    weights_path:
        Path to a local ``pytorch_model.bin`` checkpoint.
    device:
        PyTorch device string.  Defaults to the best available device.
    """

    APPLY_TO_GRAYSCALE = False

    def __init__(
        self,
        mpp: float = 10,
        prompt: str = "tissue on hematoxylin and eosin stained slide",
        activation_threshold: float | None = None,
        weights_path: str | None = None,
        device: str | None = None,
        *args,
        **kwargs,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchvision are required for CONCHGradCAMSegmenter.\n"
                "Install with: uv sync --extra conch"
            )
        super().__init__(mpp=mpp, *args, **kwargs)

        self.prompt = prompt
        self.activation_threshold = activation_threshold
        self.weights_path = weights_path
        self.device = device if device is not None else _best_device()

        print("Initializing CONCHGradCAMSegmenter")
        print(f"  Prompt:    '{self.prompt}'")
        print(f"  Threshold: {'auto (Otsu)' if activation_threshold is None else activation_threshold}")
        print(f"  Device:    {self.device}")

        self._model = None
        self._tokenizer = None
        self._tok_is_hf = False
        self._to_tensor_norm: T.Compose | None = None
        self._text_emb: torch.Tensor | None = None
        self._block = None
        self._load_model()

    def _load_model(self):
        self._model, self._tokenizer, self._tok_is_hf, full_preprocess = _load_conch(
            self.weights_path, self.device
        )
        # Keep only ToTensor + Normalize — letterboxing is handled manually.
        self._to_tensor_norm = T.Compose(
            [t for t in full_preprocess.transforms if isinstance(t, (T.ToTensor, T.Normalize))]
        )
        if not self._to_tensor_norm.transforms:
            self._to_tensor_norm = T.Compose([
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])
        self._block = _find_last_visual_block(self._model)
        self._text_emb = _embed_text(
            self._model, self._tokenizer, self._tok_is_hf, [self.prompt], self.device
        )  # (1, D)
        print("CONCH model ready.")

    @property
    def name(self) -> str:
        safe_prompt = self.prompt[:30].replace(" ", "-").replace("&", "and")
        thresh = "auto" if self.activation_threshold is None else self.activation_threshold
        return f"conch_gradcam__prompt={safe_prompt}_threshold={thresh}_mpp={self.mpp}"

    def _letterbox(self, image_rgb: np.ndarray):
        """Resize to MODEL_SIZE×MODEL_SIZE without cropping.

        Returns ``(tensor, pad_top, pad_left, new_h, new_w)`` so the
        similarity map can be un-padded and mapped back to original coords.
        """
        H, W = image_rgb.shape[:2]
        scale = MODEL_SIZE / max(H, W)
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.LANCZOS)
        pad_top = (MODEL_SIZE - new_h) // 2
        pad_left = (MODEL_SIZE - new_w) // 2
        canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (255, 255, 255))
        canvas.paste(pil, (pad_left, pad_top))
        tensor = self._to_tensor_norm(canvas).unsqueeze(0).to(self.device)
        return tensor, pad_top, pad_left, new_h, new_w

    def _segment_numpy(self, image: np.ndarray) -> np.ndarray:
        """Return a boolean tissue mask the same size as *image*.

        Parameters
        ----------
        image:
            HxWx3 uint8 RGB numpy array at the requested MPP.
        """
        from skimage.transform import resize as sk_resize

        H, W = image.shape[:2]

        img_tensor, pad_top, pad_left, new_h, new_w = self._letterbox(image)

        # (G, G) cosine-similarity map in the padded 448×448 space
        sim_map = _patch_text_similarity(
            self._model, img_tensor, self._text_emb, self._block
        )

        # Upsample to 448×448 then crop out the letterbox padding
        sim_full = sk_resize(
            sim_map, (MODEL_SIZE, MODEL_SIZE),
            order=1, mode="edge", anti_aliasing=False, preserve_range=True,
        ).astype(np.float32)
        sim_resized = sim_full[pad_top: pad_top + new_h, pad_left: pad_left + new_w]

        # Upsample to original image resolution
        heatmap = sk_resize(
            sim_resized, (H, W),
            order=1, mode="edge", anti_aliasing=False, preserve_range=True,
        ).astype(np.float32)

        if self.activation_threshold is None:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(heatmap)
        else:
            thresh = float(self.activation_threshold)

        return heatmap >= thresh

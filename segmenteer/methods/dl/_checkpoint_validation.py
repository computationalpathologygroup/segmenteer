"""Validate complete model state dictionaries before inference.

Checkpoint deserialization is performed separately by each adapter.
"""

from collections.abc import Mapping
from pathlib import Path


def load_complete_state_dict(
    model,
    state_dict,
    *,
    model_name: str,
    checkpoint_path: str | Path,
    lightning_model_prefix: bool = False,
    drop_hest_aux_classifier: bool = False,
) -> None:
    """Load all expected weights, excluding HEST's unused auxiliary head."""
    source = str(checkpoint_path)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(
            f"{model_name} checkpoint {source} has no nonempty state dictionary"
        )

    converted = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{model_name} checkpoint {source} contains a non-string weight key"
            )
        name = key.removeprefix("model.") if lightning_model_prefix else key
        if drop_hest_aux_classifier and name.startswith("aux_classifier."):
            # Exclude the training-only auxiliary head used by HEST.
            continue
        if name in converted:
            raise ValueError(
                f"{model_name} checkpoint {source} has a duplicate normalized weight key: {name}"
            )
        converted[name] = value

    if not converted:
        raise ValueError(
            f"{model_name} checkpoint {source} contains no usable model weights"
        )

    expected = set(model.state_dict())
    provided = set(converted)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:

        def sample(keys):
            snippet = ", ".join(keys[:8])
            return snippet + (f", ... (+{len(keys) - 8} more)" if len(keys) > 8 else "")

        raise RuntimeError(
            f"{model_name} checkpoint is incompatible with its architecture: {source}. "
            f"Missing keys ({len(missing)}): [{sample(missing)}]. "
            f"Unexpected keys ({len(unexpected)}): [{sample(unexpected)}]. "
            "Refusing to run inference with partially initialized weights. "
            "Check the checkpoint source and the installed model-library versions."
        )

    # Strict loading also validates the shapes of matching parameters.
    model.load_state_dict(converted, strict=True)

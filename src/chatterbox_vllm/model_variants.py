"""Stable identifiers and checkpoint details for supported T3 variants."""

from __future__ import annotations


ENGLISH_V1_MODEL_ID = "english-v1"
MULTILINGUAL_V2_MODEL_ID = "multilingual-v2"
MULTILINGUAL_V3_MODEL_ID = "multilingual-v3"

DEFAULT_MODEL_ID = MULTILINGUAL_V3_MODEL_ID
LEGACY_PROJECT_MODEL_ID = ENGLISH_V1_MODEL_ID

MODEL_LABELS = {
    ENGLISH_V1_MODEL_ID: "Original English",
    MULTILINGUAL_V2_MODEL_ID: "Multilingual V2 (English mode)",
    MULTILINGUAL_V3_MODEL_ID: "Multilingual V3 (English mode)",
}

MODEL_ALIASES = {
    "english": ENGLISH_V1_MODEL_ID,
    "v1": ENGLISH_V1_MODEL_ID,
    "multilingual": MULTILINGUAL_V3_MODEL_ID,
    "v2": MULTILINGUAL_V2_MODEL_ID,
    "v3": MULTILINGUAL_V3_MODEL_ID,
    **{model_id: model_id for model_id in MODEL_LABELS},
}

MULTILINGUAL_CHECKPOINTS = {
    MULTILINGUAL_V2_MODEL_ID: (
        "t3_mtl23ls_v2.safetensors",
        "05e904af2b5c7f8e482687a9d7336c5c824467d9",
    ),
    MULTILINGUAL_V3_MODEL_ID: (
        "t3_mtl23ls_v3.safetensors",
        "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    ),
}


def resolve_model_id(value: str | None) -> str:
    """Resolve a user-facing alias to a stable model identifier."""

    normalized = (value or DEFAULT_MODEL_ID).strip().lower()
    try:
        return MODEL_ALIASES[normalized]
    except KeyError as error:
        choices = ", ".join(MODEL_LABELS)
        raise ValueError(
            f"Unknown Chatterbox model variant {value!r}; choose one of {choices}"
        ) from error


def model_label(model_id: str) -> str:
    return MODEL_LABELS[resolve_model_id(model_id)]

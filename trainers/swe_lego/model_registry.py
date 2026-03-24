"""Model family registry for SWE-Lego pipeline.

Maps known model families to their inference, serving, and template
configurations so that pipeline scripts are generated correctly for
each model variant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """Inference and training profile for a model family."""

    # LLaMA-Factory chat template name
    template: str
    # OpenHands MODEL_CONFIG identifier (e.g. "llm.eval_qwen3_8b")
    openhands_model_config: str
    # vLLM max_model_len default
    max_model_len: int = 163840
    # vLLM extra flags (e.g. --reasoning-parser)
    vllm_extra_flags: list[str] = field(default_factory=list)
    # Minimum LLaMA-Factory version required
    min_llamafactory_version: str = "0.9.4"
    # Minimum vLLM version required
    min_vllm_version: str = "0.16.0"
    # Recommended sampling parameters
    sampling_temperature: float = 0.6
    sampling_top_p: float = 0.95


# ---------------------------------------------------------------------------
# Known model profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ModelProfile] = {
    "qwen3": ModelProfile(
        template="qwen3_nothink",
        openhands_model_config="llm.eval_qwen3_8b",
        max_model_len=163840,
    ),
    "qwen3.5": ModelProfile(
        template="qwen3",
        openhands_model_config="llm.eval_qwen3_5",
        max_model_len=262144,
        vllm_extra_flags=["--reasoning-parser", "qwen3"],
        min_llamafactory_version="0.9.5",
        min_vllm_version="0.17.0",
    ),
    "qwen2.5": ModelProfile(
        template="qwen2.5",
        openhands_model_config="llm.eval_qwen2_5",
        max_model_len=131072,
    ),
}

# Pattern: "Qwen/Qwen{family}-{size}" -> family key
_MODEL_ID_RE = re.compile(
    r"^(?:Qwen/)?Qwen([\d.]+)(?:-Coder)?-(.+)$", re.IGNORECASE
)


def resolve_model_profile(
    model_id: str,
    overrides: dict[str, Any] | None = None,
) -> ModelProfile:
    """Resolve the model profile for a HuggingFace model ID.

    Looks up the model family from the ID (e.g. ``Qwen/Qwen3.5-9B`` ->
    ``qwen3.5``) and returns the corresponding :class:`ModelProfile`.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier, e.g. ``"Qwen/Qwen3.5-9B"``.
    overrides:
        Optional dict of field overrides applied on top of the base profile.

    Returns
    -------
    ModelProfile
    """
    family = _extract_family(model_id)
    profile = _PROFILES.get(family)
    if profile is None:
        # Fall back to qwen3 profile for unknown families
        profile = _PROFILES["qwen3"]

    if overrides:
        kwargs = {f.name: getattr(profile, f.name) for f in profile.__dataclass_fields__.values()}
        kwargs.update({k: v for k, v in overrides.items() if k in kwargs})
        profile = ModelProfile(**kwargs)

    return profile


def _extract_family(model_id: str) -> str:
    """Extract the Qwen family key from a model ID.

    Examples::

        >>> _extract_family("Qwen/Qwen3.5-9B")
        'qwen3.5'
        >>> _extract_family("Qwen/Qwen3-8B")
        'qwen3'
        >>> _extract_family("Qwen/Qwen2.5-Coder-32B-Instruct")
        'qwen2.5'
    """
    m = _MODEL_ID_RE.match(model_id)
    if m:
        return f"qwen{m.group(1)}"
    return "qwen3"  # default


def get_known_families() -> list[str]:
    """Return list of registered model family keys."""
    return list(_PROFILES.keys())

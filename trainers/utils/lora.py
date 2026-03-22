"""Shared LoRA / QLoRA application utility."""

from __future__ import annotations

import logging
from typing import Any


def apply_lora(model: Any, adapter: str, training_params: dict[str, Any], logger: logging.Logger) -> Any:
    """Apply a LoRA or QLoRA adapter to a causal-LM model.

    Parameters
    ----------
    model:
        A ``transformers.PreTrainedModel`` (or compatible) to wrap.
    adapter:
        ``"lora"`` or ``"qlora"``.
    training_params:
        Recipe ``training_params`` dict – used to read ``lora_r``,
        ``lora_alpha``, ``lora_dropout``, and ``lora_target_modules``.
    logger:
        Logger instance for info/warning messages.

    Returns
    -------
    The peft-wrapped model.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("LoRA/QLoRA requires peft. Install it with: pip install peft") from exc

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(training_params.get("lora_r", 16)),
        lora_alpha=int(training_params.get("lora_alpha", 32)),
        lora_dropout=float(training_params.get("lora_dropout", 0.05)),
        target_modules=training_params.get("lora_target_modules", ["q_proj", "v_proj"]),
    )
    if adapter == "qlora":
        logger.warning("QLoRA requested without explicit quantization config; using LoRA-compatible path")
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: %.2fM / %.2fM trainable params (%.2f%%)",
        trainable / 1e6, total / 1e6, 100 * trainable / total,
    )
    return model

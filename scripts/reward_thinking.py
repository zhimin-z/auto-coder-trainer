"""Thinking-style reward shaping for GSM8K (exp21).

Single compute_score function, four behaviour modes, switched at run-time via
the `ACT_THINKING_REWARD_STYLE` environment variable. The TinyZero launcher
exports any string value under `trainer.params.*` as an `ACT_PARAM_<KEY>`
env var, and we additionally honour the unprefixed
`ACT_THINKING_REWARD_STYLE` for backward compatibility / direct invocation.

Modes:
  - `baseline`            — verl's built-in binary score (1.0 / 0.0).
  - `step_count_bonus`    — encourages explicit reasoning. Adds 0.05 per
                            arithmetic step / numeric sentence, capped at +0.3.
  - `step_count_penalty`  — discourages chain-of-thought padding. Subtracts
                            0.02 per step beyond the 6th, capped at -0.2.
  - `cleverness_bonus`    — rewards short-but-correct answers. Adds
                            0.2 only when the answer is correct AND the
                            response is below 60 tokens (whitespace split).

All modes return 0.0 when the answer is unparseable. The signature follows
verl's `verl.workers.reward_manager.naive`:
    compute_score(data_source, solution_str, ground_truth, extra_info)
"""

from __future__ import annotations

import os
import re
from typing import Any

_FINAL_RE = re.compile(r"#### (-?[0-9.,]+)")
_STEP_LINE_RE = re.compile(r"^[^=\n]*[0-9][^=\n]*[=+\-*/][^=\n]*$", re.MULTILINE)
_NUMERIC_SENTENCE_RE = re.compile(r"[A-Za-z][^.\n]{5,}\d[^.\n]*\.")


def _resolve_style() -> str:
    raw = (
        os.environ.get("ACT_THINKING_REWARD_STYLE")
        or os.environ.get("ACT_PARAM_THINKING_REWARD_STYLE")
        or "baseline"
    )
    return raw.strip().lower() or "baseline"


def _extract(solution_str: str) -> str | None:
    head_or_tail = solution_str[-300:] if len(solution_str) > 300 else solution_str
    matches = _FINAL_RE.findall(head_or_tail)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")


def _count_steps(solution_str: str) -> int:
    return len(_STEP_LINE_RE.findall(solution_str)) + len(_NUMERIC_SENTENCE_RE.findall(solution_str))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    answer = _extract(solution_str)
    if answer is None:
        return 0.0
    correct = str(answer) == str(ground_truth)
    base = 1.0 if correct else 0.0

    style = _resolve_style()

    if style == "baseline":
        return base

    if style == "step_count_bonus":
        return base + min(0.3, 0.05 * _count_steps(solution_str))

    if style == "step_count_penalty":
        steps = _count_steps(solution_str)
        if steps <= 6:
            return base
        return base - min(0.2, 0.02 * (steps - 6))

    if style == "cleverness_bonus":
        if not correct:
            return base
        token_count = len(solution_str.split())
        return base + (0.2 if token_count < 60 else 0.0)

    # Unknown style → fail loud so the smoke catches typos in the recipe.
    raise ValueError(
        f"Unknown ACT_THINKING_REWARD_STYLE={style!r}; "
        "expected one of baseline / step_count_bonus / step_count_penalty / cleverness_bonus"
    )

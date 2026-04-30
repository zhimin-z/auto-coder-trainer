"""Process reward for GSM8K — per-step bonus on top of the binary outcome.

Comparison baseline for exp05's reward-design ablation. Adds a small
length-aware shaping term so the gradient picks up signal from how the model
*reasons*, not just whether the final number matches.

Score = answer_score + shaping, where:

  - answer_score: 1.0 if the final `#### <num>` matches the ground truth,
    else 0.0 (same as verl's built-in binary GSM8K reward).
  - shaping: 0.05 per identified reasoning step in the trace, capped at 0.3.
    A "reasoning step" is a numeric line like `5 + 3 = 8` or a chain-of-thought
    sentence ending in `.` or `\n` containing a numeric token. This is a
    cheap proxy for "did the model show its work" — not a real process
    reward (no per-step verifier), but the right shape for ablation against
    binary and partial.

Total caps at 1.3, so cells with `algorithm.adv_estimator=grpo` still see
mostly outcome signal once the model converges to correct answers.

Signature: compute_score(data_source, solution_str, ground_truth, extra_info)
"""

from __future__ import annotations

import re
from typing import Any

_FINAL_RE = re.compile(r"#### (-?[0-9.,]+)")
# Lines that look like an arithmetic step, e.g. "5 + 3 = 8" or "we get 12".
_STEP_LINE_RE = re.compile(r"^[^=\n]*[0-9][^=\n]*[=+\-*/][^=\n]*$", re.MULTILINE)
_NUMERIC_SENTENCE_RE = re.compile(r"[A-Za-z][^.\n]{5,}\d[^.\n]*\.")
_PER_STEP_BONUS = 0.05
_MAX_SHAPING = 0.3


def _extract(solution_str: str) -> str | None:
    head_or_tail = solution_str[-300:] if len(solution_str) > 300 else solution_str
    matches = _FINAL_RE.findall(head_or_tail)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")


def _count_steps(solution_str: str) -> int:
    arithmetic = len(_STEP_LINE_RE.findall(solution_str))
    sentences = len(_NUMERIC_SENTENCE_RE.findall(solution_str))
    return arithmetic + sentences


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    answer = _extract(solution_str)
    answer_score = 1.0 if answer is not None and str(answer) == str(ground_truth) else 0.0
    shaping = min(_MAX_SHAPING, _PER_STEP_BONUS * _count_steps(solution_str))
    return answer_score + shaping

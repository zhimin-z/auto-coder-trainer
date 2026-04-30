"""Partial reward for GSM8K — splits credit into format and answer.

Comparison baseline for exp05's reward-design ablation. Where verl's built-in
`gsm8k.compute_score` is binary (1.0 / 0.0), this variant gives partial credit:

  - 0.0 if no `#### <number>` block can be found at all
  - 0.3 if the `#### <number>` format is present but the number is wrong
  - 1.0 if the number matches the ground truth

The expectation in the literature is that partial reward smooths the gradient
landscape early in training when the model rarely reaches the right answer
but is starting to learn the answer-suffix format.

Signature follows verl's `verl.workers.reward_manager.naive`:
    compute_score(data_source, solution_str, ground_truth, extra_info)
"""

from __future__ import annotations

import re
from typing import Any

_FINAL_RE = re.compile(r"#### (-?[0-9.,]+)")


def _extract(solution_str: str) -> str | None:
    # Match the last '#### <number>' suffix, mirroring verl's GSM8K extractor.
    if len(solution_str) > 300:
        solution_str = solution_str[-300:]
    matches = _FINAL_RE.findall(solution_str)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    answer = _extract(solution_str)
    if answer is None:
        return 0.0
    if str(answer) == str(ground_truth):
        return 1.0
    return 0.3

"""Custom reward for the TinyZero Countdown task.

verl's reward manager calls `compute_score(data_source, solution_str,
ground_truth, extra_info)` per rollout. We:

  1. Pull the model's expression from the last `<answer>...</answer>` block.
  2. Verify it uses **exactly** the input `nums` (each once, multiset).
  3. Safely eval the expression and check it equals `target`.

Returns 1.0 on success, 0.0 otherwise. Format errors and parse failures all
score 0 — same convention as TinyZero's original `tinyzero/verl/rewards/countdown.py`.
"""

from __future__ import annotations

import ast
import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Match unsigned integer literals only — leading `-` belongs to a binary op
# or a unary-minus we don't want to swallow into the number itself.
_NUMBER_RE = re.compile(r"\d+")
# Only allow a tiny arithmetic-expression AST; any other node = invalid.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Num,  # py<3.8 alias kept by ast for back-compat
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Load,
)


def _extract_expression(solution_str: str) -> str | None:
    matches = _ANSWER_RE.findall(solution_str)
    if not matches:
        return None
    return matches[-1].strip()


def _safe_eval(expr: str) -> float | None:
    """Parse & eval a numeric arithmetic expression. Return None on any error."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        # nosec - AST is restricted to numeric literals + arithmetic above.
        return float(eval(compile(tree, "<expr>", "eval")))  # noqa: S307
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def _expression_uses_exact_nums(expr: str, allowed: list[int]) -> bool:
    """Multiset check: every literal in expr appears in `allowed` exactly once each."""
    used = [int(t) for t in _NUMBER_RE.findall(expr)]
    if len(used) != len(allowed):
        return False
    return sorted(used) == sorted(allowed)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    if not isinstance(ground_truth, dict):
        return 0.0
    target = ground_truth.get("target")
    nums = ground_truth.get("nums")
    if target is None or not isinstance(nums, (list, tuple)):
        return 0.0

    expr = _extract_expression(solution_str)
    if expr is None or not expr:
        return 0.0
    if not _expression_uses_exact_nums(expr, [int(n) for n in nums]):
        return 0.0
    value = _safe_eval(expr)
    if value is None:
        return 0.0
    return 1.0 if abs(value - float(target)) < 1e-6 else 0.0

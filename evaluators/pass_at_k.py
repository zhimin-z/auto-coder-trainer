"""pass@k metric computation."""

import math
from typing import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct samples per problem.
        k: k in pass@k.

    Returns:
        pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_pass_at_k(results: Sequence[dict], k_values: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Compute pass@k for multiple k values across all problems.

    Args:
        results: List of {problem_id, n_samples, n_correct} dicts.
        k_values: k values to compute.

    Returns:
        Dict mapping "pass@k" to the average pass@k across problems.

    TODO: Implement aggregation logic.
    """
    raise NotImplementedError("pass@k aggregation not yet implemented")

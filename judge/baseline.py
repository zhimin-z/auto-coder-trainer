"""Baseline alignment utilities for experiment judge."""

from typing import Any


def find_baseline(recipe: dict[str, Any], result_db: Any) -> dict[str, Any] | None:
    """Find the matching baseline experiment for a given recipe.

    TODO: Query result DB for baseline with same model + dataset + default trainer params.
    """
    raise NotImplementedError


def compute_delta(current_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    """Compute metric deltas between current and baseline.

    TODO: Implement delta computation with significance testing.
    """
    raise NotImplementedError

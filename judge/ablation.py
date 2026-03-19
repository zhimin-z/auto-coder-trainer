"""Ablation registry and validation for experiment judge."""

from typing import Any


def validate_ablation_coverage(recipe: dict[str, Any], result_db: Any) -> list[str]:
    """Check which ablation experiments are missing.

    Returns list of missing ablation names.

    TODO: Cross-reference recipe.ablation spec with result DB.
    """
    raise NotImplementedError


def register_ablation(experiment_id: str, variable: str, value: Any, result_db: Any) -> None:
    """Register an ablation result in the result DB.

    TODO: Implement ablation registration.
    """
    raise NotImplementedError

"""Result deduplication for experiment judge."""

from typing import Any


def compute_config_hash(recipe: dict[str, Any]) -> str:
    """Compute a deterministic hash of the experiment configuration.

    TODO: Hash relevant recipe fields (model, dataset, trainer, params).
    """
    raise NotImplementedError


def find_duplicates(config_hash: str, result_db: Any) -> list[dict[str, Any]]:
    """Find duplicate experiments in the result DB.

    TODO: Query result DB by config hash.
    """
    raise NotImplementedError

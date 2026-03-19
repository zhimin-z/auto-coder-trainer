"""Seed management utilities for reproducible experiments."""


def set_all_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, PyTorch, and CUDA.

    TODO: Implement seed setting.
    """
    raise NotImplementedError("Seed setting not yet implemented")


def get_seed_list(config: dict) -> list[int]:
    """Extract seed list from recipe eval config. Defaults to [42, 123, 456]."""
    return config.get("eval", {}).get("seeds", [42, 123, 456])

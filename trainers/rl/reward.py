"""Reward function registry for RL training.

Supports composable reward functions that can be specified in Recipe IR.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def compute(self, trajectory: dict[str, Any]) -> float:
        """Compute reward for a single trajectory."""
        ...


class BinaryPassReward(BaseReward):
    """Binary reward: 1.0 if tests pass, 0.0 otherwise."""

    def compute(self, trajectory: dict[str, Any]) -> float:
        raise NotImplementedError


class EntropyAwareReward(BaseReward):
    """Entropy-aware reward that penalizes low-entropy (degenerate) solutions."""

    def __init__(self, entropy_coeff: float = 0.01):
        self.entropy_coeff = entropy_coeff

    def compute(self, trajectory: dict[str, Any]) -> float:
        raise NotImplementedError


class CompositeReward(BaseReward):
    """Weighted combination of multiple reward components."""

    def __init__(self, components: list[tuple[BaseReward, float]]):
        self.components = components

    def compute(self, trajectory: dict[str, Any]) -> float:
        raise NotImplementedError


# Registry
REWARD_REGISTRY: dict[str, type[BaseReward]] = {
    "binary_pass": BinaryPassReward,
    "entropy_aware": EntropyAwareReward,
    "composite": CompositeReward,
}


def build_reward(reward_config: dict[str, Any]) -> BaseReward:
    """Build a reward function from recipe config.

    TODO: Implement factory logic.
    """
    raise NotImplementedError("Reward building not yet implemented")

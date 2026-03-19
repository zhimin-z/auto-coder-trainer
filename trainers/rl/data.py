"""RL data loading and environment setup utilities."""

from typing import Any


def load_rl_prompts(sources: list[dict[str, Any]], filters: list[dict[str, Any]] | None = None) -> Any:
    """Load coding task prompts for RL rollouts.

    TODO: Implement prompt loading from trajectory datasets.
    """
    raise NotImplementedError("RL prompt loading not yet implemented")


def setup_rollout_env(env_config: dict[str, Any]) -> Any:
    """Set up the rollout environment for RL training.

    TODO: Implement sandboxed code execution environment.
    """
    raise NotImplementedError("Rollout environment setup not yet implemented")

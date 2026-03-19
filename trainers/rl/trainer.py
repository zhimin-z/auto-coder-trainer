"""RL Trainer skeleton — wraps veRL for GRPO/PPO training.

Default backend: veRL (https://github.com/volcengine/verl)
Expects a compiled recipe config with trainer.type in ("rl", "grpo") and trainer.backend == "verl".
"""

from typing import Any

from trainers.base import BaseTrainer, TrainResult, EvalResult


class RLTrainer(BaseTrainer):
    """Reinforcement Learning trainer using veRL.

    Config expectations:
        model.base: HuggingFace model ID
        trainer.type: "rl" | "grpo"
        trainer.backend: "verl" (default)
        trainer.params: {lr, ppo_epochs, rollout_batch_size, kl_coeff, entropy_coeff}
        trainer.reward: {type, components}
    """

    def prepare_data(self) -> Any:
        """Load and prepare RL training data (prompts + environments).

        TODO: Implement with veRL data utilities.
        - Load coding task prompts
        - Set up rollout environments
        - Configure reward functions
        """
        raise NotImplementedError("RL data preparation not yet implemented")

    def train(self) -> TrainResult:
        """Run RL training with veRL.

        TODO: Implement with verl.trainer.main_ppo or GRPO equivalent.
        - Initialize actor/critic models
        - Configure PPO/GRPO training loop
        - Set up reward model from recipe
        - Train with rollouts and save checkpoint

        veRL integration point:
            from verl.trainer.main_ppo import main as verl_main
            verl_main(config=self._build_verl_config())
        """
        raise NotImplementedError("RL training not yet implemented")

    def evaluate(self, checkpoint_path: str, benchmark: str, seed: int = 42) -> EvalResult:
        """Evaluate RL-trained checkpoint on a benchmark.

        TODO: Delegate to evaluators/ module.
        """
        raise NotImplementedError("RL evaluation not yet implemented")

    def _build_verl_config(self) -> dict[str, Any]:
        """Convert recipe config to veRL-native configuration.

        TODO: Map recipe fields to veRL config format.
        """
        raise NotImplementedError("veRL config builder not yet implemented")

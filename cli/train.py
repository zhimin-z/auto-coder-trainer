"""Train command — execute a training experiment from a recipe.

Compiles a Recipe IR JSON into a training configuration, runs the
experiment, evaluates results, and submits to the experiment judge.
"""

import argparse


def run_train(args: argparse.Namespace) -> None:
    """Execute the training pipeline.

    Pipeline:
        1. Load and validate recipe JSON
        2. Compile recipe to training config
        3. Select trainer backend (TRL for SFT, veRL for RL)
        4. Run training
        5. Evaluate on specified benchmarks
        6. Submit to experiment judge
        7. Store results in result DB

    TODO: Implement training pipeline orchestration.
    """
    raise NotImplementedError("Train pipeline not yet implemented")

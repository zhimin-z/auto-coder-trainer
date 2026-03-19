"""Experiment Judge — the core arbiter for training experiments.

Enforces scientific rigor: baseline alignment, seed consistency,
minimal ablation, result deduplication, and failure attribution.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Verdict(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_ABLATION = "needs_ablation"
    NEEDS_RERUN = "needs_rerun"


@dataclass
class JudgementResult:
    """Result of experiment judgement."""
    verdict: Verdict
    recipe_id: str
    reasoning: str
    checks: dict[str, bool] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


class ExperimentJudge:
    """Judges experiment validity and decides whether to accept results.

    Judgement pipeline:
        1. check_baseline() — verify baseline run exists and is comparable
        2. check_seeds() — verify seed consistency across runs
        3. check_ablation() — verify minimal ablation coverage
        4. check_dedup() — check for duplicate experiments in result DB
        5. attribute_failure() — if below baseline, analyze why
        6. judge() — produce final verdict
    """

    def __init__(self, result_db: Any = None):
        self.result_db = result_db

    def check_baseline(self, recipe_id: str, results: dict[str, Any]) -> bool:
        """Verify that a corresponding baseline experiment exists.

        TODO: Query result DB for baseline run with same model/dataset but default params.
        """
        raise NotImplementedError

    def check_seeds(self, results: list[dict[str, Any]], expected_seeds: list[int]) -> bool:
        """Verify that all expected seeds were run.

        TODO: Check that results cover all seeds in recipe.eval.seeds.
        """
        raise NotImplementedError

    def check_ablation(self, recipe_id: str, ablation_config: list[dict]) -> bool:
        """Verify that minimal ablation experiments were conducted.

        TODO: Check result DB for ablation runs matching recipe.ablation spec.
        """
        raise NotImplementedError

    def check_dedup(self, recipe_id: str, results: dict[str, Any]) -> bool:
        """Check if an equivalent experiment already exists in the result DB.

        TODO: Query result DB for experiments with equivalent config.
        """
        raise NotImplementedError

    def attribute_failure(self, recipe_id: str, results: dict[str, Any], baseline: dict[str, Any]) -> str:
        """Analyze why an experiment performed below baseline.

        TODO: Implement failure attribution logic.
        """
        raise NotImplementedError

    def judge(self, recipe_id: str, results: dict[str, Any]) -> JudgementResult:
        """Run full judgement pipeline and return verdict.

        TODO: Orchestrate all checks and produce final verdict.
        """
        raise NotImplementedError

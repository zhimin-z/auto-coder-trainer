"""Auto-generate technical reports from experiment results."""

from typing import Any
from pathlib import Path


class ReportGenerator:
    """Generates Markdown/LaTeX technical reports from result DB data.

    Report structure:
    1. Method description (from recipe)
    2. Experimental setup (model, data, hyperparams)
    3. Results table (main + ablation)
    4. Analysis (vs baseline, failure attribution)
    5. Conclusions and next steps
    """

    def __init__(self, result_db: Any):
        self.result_db = result_db

    def generate_markdown(self, experiment_ids: list[str], output_path: str | Path) -> str:
        """Generate a Markdown report for the given experiments.

        TODO: Implement report generation.
        """
        raise NotImplementedError

    def generate_latex(self, experiment_ids: list[str], output_path: str | Path) -> str:
        """Generate a LaTeX report (compatible with ARIS paper-writing workflow).

        TODO: Implement LaTeX report generation.
        """
        raise NotImplementedError

    def generate_comparison_table(self, recipe_ids: list[str]) -> str:
        """Generate a comparison table across multiple recipes.

        TODO: Implement comparison table.
        """
        raise NotImplementedError

"""Evaluation report formatting utilities."""

from typing import Any
from evaluators.base import BenchmarkResult


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format evaluation results as a Markdown table.

    TODO: Implement table formatting.
    """
    raise NotImplementedError("Results table formatting not yet implemented")


def compare_with_baseline(current: BenchmarkResult, baseline: BenchmarkResult) -> dict[str, Any]:
    """Compare current results with a baseline.

    TODO: Implement comparison with delta computation.
    """
    raise NotImplementedError("Baseline comparison not yet implemented")

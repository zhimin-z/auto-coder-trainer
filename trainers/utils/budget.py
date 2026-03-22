"""Budget tracking for training runs.

Enforces wall-clock time limits declared in recipe budget configs
(e.g. max_gpu_hours) so that runaway training jobs are caught early.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when a training run exceeds its declared budget."""


class BudgetTracker:
    """Lightweight wall-clock budget tracker.

    Args:
        budget: Budget dict from compiled recipe config. Recognised keys:
            - max_gpu_hours (float): Maximum wall-clock hours allowed.
            - max_cost_usd (float): Informational cost cap (logged, not enforced
              here since we have no billing integration).
            - gpu_type (str): Informational GPU type label.
    """

    def __init__(self, budget: dict[str, Any]) -> None:
        self.max_gpu_hours: float | None = budget.get("max_gpu_hours")
        self.max_cost_usd: float | None = budget.get("max_cost_usd")
        self.gpu_type: str | None = budget.get("gpu_type")
        self._start_time: float | None = None

        if self.max_gpu_hours is not None:
            logger.info(
                "BudgetTracker initialised: max_gpu_hours=%.2f, gpu_type=%s",
                self.max_gpu_hours,
                self.gpu_type or "unspecified",
            )

    def start(self) -> None:
        """Record the start time of the training run."""
        self._start_time = time.monotonic()
        logger.info("BudgetTracker started")

    def elapsed_hours(self) -> float:
        """Return elapsed wall-clock hours since start().

        Returns:
            Elapsed hours, or 0.0 if start() has not been called.
        """
        if self._start_time is None:
            return 0.0
        return (time.monotonic() - self._start_time) / 3600.0

    def check(self) -> None:
        """Check whether the budget has been exceeded.

        Raises:
            BudgetExceededError: If elapsed time exceeds max_gpu_hours.
        """
        if self.max_gpu_hours is None:
            return
        elapsed = self.elapsed_hours()
        if elapsed > self.max_gpu_hours:
            raise BudgetExceededError(
                f"Budget exceeded: {elapsed:.2f} hours elapsed, "
                f"max_gpu_hours={self.max_gpu_hours:.2f}"
            )

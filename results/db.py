"""SQLite-backed result database for experiment tracking."""

import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "results.db"


class ResultDB:
    """SQLite-backed experiment result database.

    Stores experiments, ablation results, and judge verdicts.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect to the database and initialize schema if needed.

        TODO: Implement connection + schema initialization from schema.sql.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def insert_experiment(self, experiment: dict[str, Any]) -> str:
        """Insert a new experiment record. Returns experiment ID.

        TODO: Implement insert.
        """
        raise NotImplementedError

    def insert_ablation(self, ablation: dict[str, Any]) -> int:
        """Insert an ablation result. Returns ablation ID.

        TODO: Implement insert.
        """
        raise NotImplementedError

    def insert_verdict(self, verdict: dict[str, Any]) -> int:
        """Insert a judge verdict. Returns verdict ID.

        TODO: Implement insert.
        """
        raise NotImplementedError

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get an experiment by ID.

        TODO: Implement query.
        """
        raise NotImplementedError

    def find_by_recipe(self, recipe_id: str) -> list[dict[str, Any]]:
        """Find all experiments for a given recipe.

        TODO: Implement query.
        """
        raise NotImplementedError

    def find_by_config_hash(self, config_hash: str) -> list[dict[str, Any]]:
        """Find experiments with matching config hash (dedup check).

        TODO: Implement query.
        """
        raise NotImplementedError

    def get_best_recipe(self, metric: str = "resolve_rate") -> dict[str, Any] | None:
        """Get the best-performing recipe by a given metric.

        TODO: Implement ranking query.
        """
        raise NotImplementedError

"""Tests for newly added features: BudgetTracker, shared LoRA, rerun CLI, WAL mode, checkpoint verification."""

import sqlite3
import time
from argparse import Namespace
from pathlib import Path

import pytest

from results.db import ResultDB
from trainers.utils.budget import BudgetExceededError, BudgetTracker


# ---------------------------------------------------------------------------
# 1. BudgetTracker
# ---------------------------------------------------------------------------


def test_budget_tracker_starts_and_tracks_elapsed() -> None:
    tracker = BudgetTracker(budget={"max_gpu_hours": 100})
    tracker.start()
    assert tracker.elapsed_hours() > 0


def test_budget_tracker_raises_on_exceeded() -> None:
    tracker = BudgetTracker(budget={"max_gpu_hours": 0.0001})
    tracker.start()
    time.sleep(0.5)
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_budget_tracker_no_budget_is_noop() -> None:
    tracker = BudgetTracker(budget={})
    tracker.start()
    tracker.check()  # should not raise


# ---------------------------------------------------------------------------
# 2. Shared LoRA
# ---------------------------------------------------------------------------


def test_apply_lora_raises_without_peft(monkeypatch) -> None:
    import trainers.utils.lora as lora_mod

    # Force the peft import inside apply_lora to fail by injecting a
    # failing import into the builtins.
    import builtins

    _real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "peft":
            raise ImportError("no peft")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="LoRA/QLoRA requires peft"):
        lora_mod.apply_lora(
            model=None,
            adapter="lora",
            training_params={},
            logger=__import__("logging").getLogger("test"),
        )


# ---------------------------------------------------------------------------
# 3. Rerun CLI
# ---------------------------------------------------------------------------


def _seed_db_with_pending_report_task(db_path: Path) -> None:
    """Create a DB with an experiment and a pending generate_report task."""
    db = ResultDB(db_path)
    db.connect()
    try:
        db.insert_experiment(
            {
                "id": "exp-rerun-1",
                "recipe_id": "recipe-rerun",
                "config_hash": "hash-rerun",
                "status": "success",
                "trainer_type": "sft",
                "backend": "trl",
                "model_base": "demo-model",
                "metrics_json": {"resolve_rate": 0.5},
                "train_metrics_json": {},
                "recipe_json": {"id": "recipe-rerun"},
                "budget_json": {},
                "checkpoint_path": None,
                "error": None,
            }
        )
        db.upsert_task(
            {
                "id": "task-report-1",
                "recipe_id": "recipe-rerun",
                "experiment_id": "exp-rerun-1",
                "kind": "generate_report",
                "title": "Generate report for recipe-rerun",
                "status": "pending",
                "priority": "medium",
                "payload_json": {},
                "notes": None,
            }
        )
    finally:
        db.close()


def test_rerun_dry_run_shows_plan(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    _seed_db_with_pending_report_task(db_path)

    from cli.rerun import run_rerun

    run_rerun(Namespace(recipe_id="recipe-rerun", dry_run=True))

    captured = capsys.readouterr().out
    assert "generate_report" in captured
    assert "dry-run" in captured.lower()

    # Verify the task is still pending (no side-effects)
    db = ResultDB(db_path)
    db.connect()
    try:
        tasks = db.get_tasks(recipe_id="recipe-rerun")
        report_task = [t for t in tasks if t["kind"] == "generate_report"][0]
        assert report_task["status"] == "pending"
    finally:
        db.close()


def test_rerun_dispatches_generate_report(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    _seed_db_with_pending_report_task(db_path)

    import cli.report as report_mod

    monkeypatch.setattr(report_mod, "run_report", lambda args: None)

    from cli.rerun import run_rerun

    run_rerun(Namespace(recipe_id="recipe-rerun", dry_run=False))

    db = ResultDB(db_path)
    db.connect()
    try:
        tasks = db.get_tasks(recipe_id="recipe-rerun")
        report_task = [t for t in tasks if t["kind"] == "generate_report"][0]
        assert report_task["status"] == "completed"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. SQLite WAL mode
# ---------------------------------------------------------------------------


def test_db_uses_wal_mode(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    db = ResultDB(db_path)
    db.connect()
    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA journal_mode").fetchone()
        conn.close()
        assert result[0] == "wal"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. Checkpoint verification
# ---------------------------------------------------------------------------


def _insert_experiment_with_checkpoint(db: ResultDB, experiment_id: str, checkpoint_path: str | None) -> None:
    db.insert_experiment(
        {
            "id": experiment_id,
            "recipe_id": "recipe-ckpt",
            "config_hash": "hash-ckpt",
            "status": "success",
            "trainer_type": "sft",
            "backend": "trl",
            "model_base": "demo-model",
            "metrics_json": {},
            "train_metrics_json": {},
            "recipe_json": {},
            "budget_json": {},
            "checkpoint_path": checkpoint_path,
            "error": None,
        }
    )


def test_verify_checkpoint_returns_true_for_existing(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "model.safetensors").write_text("stub")

    db = ResultDB(db_path)
    db.connect()
    try:
        _insert_experiment_with_checkpoint(db, "exp-ckpt-1", str(ckpt_dir))
        assert db.verify_checkpoint("exp-ckpt-1") is True
    finally:
        db.close()


def test_verify_checkpoint_returns_false_for_missing(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "results.db"
    monkeypatch.setenv("ACT_RESULTS_DB", str(db_path))

    db = ResultDB(db_path)
    db.connect()
    try:
        _insert_experiment_with_checkpoint(db, "exp-ckpt-2", "/tmp/nonexistent-checkpoint-path-xyz")
        assert db.verify_checkpoint("exp-ckpt-2") is False
    finally:
        db.close()

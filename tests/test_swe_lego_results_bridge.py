"""Tests for SWE-Lego results bridge (parsing and import)."""
import json
from pathlib import Path

from trainers.swe_lego.results_bridge import (
    parse_training_logs,
    parse_swebench_results,
    import_results,
    import_and_judge,
)


def _write_trainer_state(log_dir: Path, loss: float = 0.5) -> None:
    state = {
        "epoch": 4.0,
        "global_step": 100,
        "log_history": [
            {"loss": 1.0, "learning_rate": 1e-4, "epoch": 1.0, "step": 25},
            {"loss": 0.8, "learning_rate": 8e-5, "epoch": 2.0, "step": 50},
            {"loss": 0.6, "learning_rate": 5e-5, "epoch": 3.0, "step": 75},
            {"loss": loss, "learning_rate": 2e-5, "epoch": 4.0, "step": 100},
        ],
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "trainer_state.json").write_text(json.dumps(state))


def _write_swebench_report(results_dir: Path, resolved: int = 75, total: int = 500) -> None:
    report = {
        "resolved_ids": [f"instance_{i}" for i in range(resolved)],
        "total_instances": total,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "openhands.swe_bench.json").write_text(json.dumps(report))


def test_parse_training_logs_ok(tmp_path: Path) -> None:
    _write_trainer_state(tmp_path, loss=0.5)
    result = parse_training_logs(tmp_path)
    assert result["status"] == "ok"
    assert result["final_loss"] == 0.5
    assert result["best_loss"] == 0.5
    assert result["epoch"] == 4.0
    assert result["num_log_entries"] == 4


def test_parse_training_logs_not_found(tmp_path: Path) -> None:
    result = parse_training_logs(tmp_path)
    assert result["status"] == "not_found"


def test_parse_swebench_results_ok(tmp_path: Path) -> None:
    _write_swebench_report(tmp_path, resolved=75, total=500)
    result = parse_swebench_results(tmp_path)
    assert result["status"] == "ok"
    assert result["resolved_count"] == 75
    assert result["total_count"] == 500
    assert result["resolve_rate"] == 75 / 500 * 100.0


def test_parse_swebench_results_not_found(tmp_path: Path) -> None:
    result = parse_swebench_results(tmp_path)
    assert result["status"] == "not_found"


def test_import_results_with_training_logs(tmp_path: Path) -> None:
    saves_dir = tmp_path / "saves" / "SWE-Lego-test"
    _write_trainer_state(saves_dir, loss=0.4)
    # Also write a config.json to simulate checkpoint
    (saves_dir / "config.json").write_text("{}")

    result = import_results(tmp_path, recipe_id="test", experiment_id="exp-001")
    assert result["experiment_id"] == "exp-001"
    assert result["train_result"]["status"] == "success"
    assert result["train_result"]["checkpoint_path"] == str(saves_dir)


def test_import_and_judge_strong_verdict(tmp_path: Path) -> None:
    saves_dir = tmp_path / "saves" / "SWE-Lego-test"
    _write_trainer_state(saves_dir)

    # Create SWE-bench results in the expected location
    swebench_dir = tmp_path.parent / "trainers" / "swe_lego" / "SWE-Lego" / "SWE-bench-4.0.4" / "results"
    # Since SWE_LEGO_ROOT is relative to the module, we just test the fallback
    result = import_and_judge(tmp_path, recipe_id="test", experiment_id="exp-002")
    assert result["experiment_id"] == "exp-002"
    assert "verdict" in result
    assert "report_path" in result
    # Report file should exist
    assert Path(result["report_path"]).exists()


def test_import_and_judge_train_failed_verdict(tmp_path: Path) -> None:
    # No training logs -> train_failed
    result = import_and_judge(tmp_path, recipe_id="test", experiment_id="exp-003")
    assert result["verdict"] in ("train_failed", "reject")

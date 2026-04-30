import json
from pathlib import Path
from typing import cast

from trainers.tinyzero.results_bridge import import_results


def test_tinyzero_import_results_success_with_eval(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "tinyzero"
    results_dir = bundle_dir / "results"
    checkpoints_dir = bundle_dir / "checkpoints" / "checkpoint-1"
    results_dir.mkdir(parents=True)
    checkpoints_dir.mkdir(parents=True)

    _ = (results_dir / "train_exit_code.txt").write_text("0\n")
    _ = (results_dir / "train_metrics.json").write_text(json.dumps({"train_loss": 0.42}))
    _ = (results_dir / "eval_results.json").write_text(
        json.dumps(
            {
                "eval_results": [
                    {
                        "benchmark": "humaneval",
                        "seed": 42,
                        "metrics": {"pass@1": 0.55},
                    }
                ]
            }
        )
    )
    _ = (checkpoints_dir / "config.json").write_text("{}")

    payload = import_results(
        bundle_dir,
        recipe_id="recipe-tinyzero-import-001",
        experiment_id="exp-tinyzero-import-001",
        expected_seeds=[42],
    )

    train_result = payload["train_result"]
    assert isinstance(train_result, dict)
    train_result = cast(dict[str, object], train_result)
    assert train_result["status"] == "success"
    train_metrics = train_result["metrics"]
    assert isinstance(train_metrics, dict)
    train_metrics = cast(dict[str, object], train_metrics)
    assert train_metrics["train_loss"] == 0.42
    assert train_result["checkpoint_path"] is not None
    eval_results = payload["eval_results"]
    assert isinstance(eval_results, list)
    eval_results = cast(list[dict[str, object]], eval_results)
    assert len(eval_results) == 1
    assert isinstance(eval_results[0], dict)
    assert eval_results[0]["benchmark"] == "humaneval"
    eval_metrics = eval_results[0]["metrics"]
    assert isinstance(eval_metrics, dict)
    eval_metrics = cast(dict[str, object], eval_metrics)
    assert eval_metrics["pass@1"] == 0.55


def test_tinyzero_import_results_parses_verl_train_log(tmp_path: Path) -> None:
    """verl 0.7.x writes no metrics.json — the bridge must scrape train.log."""
    bundle_dir = tmp_path / "tinyzero"
    results_dir = bundle_dir / "results"
    results_dir.mkdir(parents=True)

    _ = (results_dir / "train_exit_code.txt").write_text("0\n")
    # Realistic verl 0.7.x stdout: tqdm progress bars prefix every step
    # except the first, so the parser must NOT anchor at line start.
    _ = (results_dir / "train.log").write_text(
        "step:1 - train/loss:1.74 - train/grad_norm:175.2 - train/lr:1e-05\n"
        "Epoch 1/1:  96%|...| 23/24 [00:11<00:00,  2.29it/s]"
        "step:24 - train/loss:0.62 - train/grad_norm:137.2 - train/lr:2e-05\n"
        "step:24 - val/loss:0.0654\n"
        "Final validation metrics: {'val/loss': 0.0654}\n"
    )

    payload = import_results(
        bundle_dir,
        recipe_id="recipe-tinyzero-log-001",
        experiment_id="exp-tinyzero-log-001",
        expected_seeds=[42],
    )

    train_metrics = cast(dict[str, object], cast(dict[str, object], payload["train_result"])["metrics"])
    # Final-step train/loss is surfaced under the flat key, not just exit_code.
    assert train_metrics["train_loss"] == 0.62
    assert train_metrics["val_loss"] == 0.0654
    assert train_metrics["exit_code"] == 0.0


def test_tinyzero_import_results_failure_without_exit_code(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "tinyzero"
    bundle_dir.mkdir(parents=True)

    payload = import_results(
        bundle_dir,
        recipe_id="recipe-tinyzero-import-002",
        experiment_id="exp-tinyzero-import-002",
        expected_seeds=[42],
    )

    train_result = payload["train_result"]
    assert isinstance(train_result, dict)
    train_result = cast(dict[str, object], train_result)
    assert train_result["status"] == "failed"
    assert payload["eval_results"] == []

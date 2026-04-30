"""Results bridge for TinyZero launcher bundles.

Parses launcher bundle artifacts written by ``trainers.tinyzero.launcher``
and converts them into the canonical import payload expected by ``cli.train``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast


def _load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload_obj = cast(object, json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError):
        return {}
    return _coerce_str_object_dict(payload_obj)


def _coerce_str_object_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    payload_dict = cast(dict[object, object], payload)
    normalized: dict[str, object] = {}
    for key, value in payload_dict.items():
        if isinstance(key, str):
            normalized[key] = value
    return normalized


def _coerce_numeric_metrics(payload: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


_VAL_LOSS_RE = re.compile(r"val/loss[: ]+([\-\d.eE+]+)")
# tqdm prefixes ("Epoch 1/1:  96%|... |") may precede the step block, so we
# search anywhere in the line rather than anchoring at start.
_SFT_TRAIN_LINE_RE = re.compile(r"step:(\d+)\s+- (train/[^\n]+)")
# verl GRPO/PPO lines look like:
#   step:24 - actor/pg_loss:0.12 - actor/kl_loss:0.001 - critic/score/mean:0.5 - ...
# Captures everything after "step:N - " for any RL line (no train/ prefix).
_RL_TRAIN_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.+)$")
# Keys we promote to flat metric names so judge/report can see them.
_RL_KEY_FLAT_MAP = {
    "actor/pg_loss": "actor_pg_loss",
    "actor/kl_loss": "actor_kl_loss",
    "actor/entropy": "actor_entropy",
    "actor/grad_norm": "actor_grad_norm",
    "actor/lr": "actor_lr",
    "critic/score/mean": "critic_score_mean",
    "critic/score/max": "critic_score_max",
    "critic/score/min": "critic_score_min",
    "critic/rewards/mean": "critic_rewards_mean",
    "critic/advantages/mean": "critic_advantages_mean",
    "response_length/mean": "response_length_mean",
    "response_length/max": "response_length_max",
    "training/global_step": "global_step",
    "training/epoch": "epoch",
}


def _parse_kv_pairs(blob: str) -> dict[str, float]:
    """Parse `key:value - key:value` style payload into floats."""
    parsed: dict[str, float] = {}
    for part in blob.split(" - "):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        try:
            parsed[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return parsed


def _parse_train_log(log_path: Path) -> dict[str, float]:
    """Extract metrics from verl's stdout log when no metrics.json is written.

    Handles two formats:
      * SFT: ``step:N - train/loss:.. - train/grad_norm:.. - ...``
      * GRPO/PPO: ``step:N - actor/pg_loss:.. - critic/score/mean:.. - ...``
    Surfaces the **last** training step's metrics under flat keys
    (``train_loss``, ``actor_pg_loss``, ``critic_score_mean``, ...).
    """
    if not log_path.exists():
        return {}
    metrics: dict[str, float] = {}
    last_sft: dict[str, float] = {}
    last_rl: dict[str, float] = {}
    try:
        for raw in log_path.read_text(errors="replace").splitlines():
            line = raw.strip()
            m = _VAL_LOSS_RE.search(line)
            if m and "val/loss" in line:
                try:
                    metrics["val_loss"] = float(m.group(1))
                except ValueError:
                    pass
            sm = _SFT_TRAIN_LINE_RE.search(line)
            if sm and "train/loss" in line:
                parsed = _parse_kv_pairs(sm.group(2))
                if parsed:
                    last_sft = parsed
                continue
            rm = _RL_TRAIN_LINE_RE.search(line)
            if rm and ("actor/pg_loss" in line or "critic/score/mean" in line):
                parsed = _parse_kv_pairs(rm.group(2))
                if parsed:
                    last_rl = parsed
    except OSError:
        return metrics
    # SFT: flatten train/* keys
    for key, value in last_sft.items():
        metrics.setdefault(key.replace("/", "_"), value)
    # RL: surface curated keys under stable flat names
    for original_key, flat_key in _RL_KEY_FLAT_MAP.items():
        if original_key in last_rl:
            metrics.setdefault(flat_key, last_rl[original_key])
    return metrics


def _find_checkpoint_path(bundle_dir: Path) -> str | None:
    checkpoints_dir = bundle_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    # Prefer the most recent nested checkpoint directory with model artifacts.
    candidates = sorted(
        [
            path
            for path in checkpoints_dir.rglob("*")
            if path.is_dir()
            and any((path / name).exists() for name in ("config.json", "adapter_config.json"))
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])

    # Fall back to the checkpoints root if anything was written there.
    has_files = any(path.is_file() for path in checkpoints_dir.rglob("*"))
    return str(checkpoints_dir) if has_files else None


def _parse_train_result(bundle_dir: Path) -> dict[str, object]:
    results_dir = bundle_dir / "results"
    exit_code_path = results_dir / "train_exit_code.txt"

    status = "failed"
    error: str | None = "TinyZero training exit code not found"
    metrics: dict[str, float] = {}

    if exit_code_path.exists():
        try:
            exit_code = int(exit_code_path.read_text().strip())
            metrics["exit_code"] = float(exit_code)
            if exit_code == 0:
                status = "success"
                error = None
            else:
                status = "failed"
                error = f"TinyZero training failed with exit code {exit_code}"
        except ValueError:
            error = "TinyZero training exit code is invalid"

    train_metrics = _load_json_if_exists(results_dir / "train_metrics.json")
    if not train_metrics:
        train_metrics = _load_json_if_exists(results_dir / "metrics.json")
    train_metrics_payload = train_metrics.get("train")
    train_payload_dict = _coerce_str_object_dict(train_metrics_payload)
    if train_payload_dict:
        metrics.update(_coerce_numeric_metrics(train_payload_dict))
    else:
        metrics.update(_coerce_numeric_metrics(train_metrics))

    # verl 0.7.x emits no metrics.json; harvest from train.log so the judge
    # and report see val_loss / train_loss instead of just exit_code.
    metrics.update(_parse_train_log(results_dir / "train.log"))

    return {
        "status": status,
        "metrics": metrics,
        "checkpoint_path": _find_checkpoint_path(bundle_dir),
        "error": error,
    }


def _parse_eval_results(
    bundle_dir: Path,
    *,
    recipe_id: str,
    expected_seeds: list[int] | None = None,
) -> list[dict[str, object]]:
    results_dir = bundle_dir / "results"
    payload = _load_json_if_exists(results_dir / "eval_results.json")
    raw_items: list[dict[str, object]] = []

    eval_results_payload = payload.get("eval_results")
    if isinstance(eval_results_payload, list):
        for item in cast(list[object], eval_results_payload):
            item_dict = _coerce_str_object_dict(item)
            if item_dict:
                raw_items.append(item_dict)
    elif payload:
        # Allow a compact dict format: {"humaneval": {"pass@1": 0.52}}
        for benchmark, metrics in payload.items():
            metrics_dict = _coerce_str_object_dict(metrics)
            if metrics_dict:
                raw_items.append(
                    {
                        "benchmark": str(benchmark),
                        "metrics": metrics_dict,
                    }
                )

    seeds = [int(seed) for seed in (expected_seeds or [42])]
    fallback_seed = seeds[0] if seeds else 42

    eval_results: list[dict[str, object]] = []
    for item in raw_items:
        benchmark_obj = item.get("benchmark", "")
        benchmark = str(benchmark_obj)
        if not benchmark:
            continue
        metrics_obj = _coerce_str_object_dict(item.get("metrics", {}))
        if not metrics_obj:
            continue
        seed_obj = item.get("seed", fallback_seed)
        seed = int(seed_obj) if isinstance(seed_obj, int) else fallback_seed
        details_obj = item.get("details", {})
        eval_results.append(
            {
                "recipe_id": recipe_id,
                "benchmark": benchmark,
                "metrics": _coerce_numeric_metrics(metrics_obj),
                "seed": seed,
                "details": details_obj if isinstance(details_obj, dict) else {},
            }
        )

    return eval_results


def _detect_trainer_type(bundle_dir: Path) -> str:
    """Detect trainer type by inspecting the bundle's run.sh.

    The launcher writes either `verl.trainer.sft_trainer` (SFT) or
    `verl.trainer.main_ppo` / `recipe.dapo.main_dapo` (RL/GRPO/DAPO).
    Defaults to "sft" if run.sh is unreadable.
    """
    run_path = bundle_dir / "run.sh"
    if not run_path.exists():
        return "sft"
    try:
        content = run_path.read_text(errors="replace")
    except OSError:
        return "sft"
    if "main_dapo" in content:
        return "dapo"
    if "main_ppo" in content:
        return "grpo"
    return "sft"


def import_results(
    bundle_dir: str | Path,
    recipe_id: str,
    experiment_id: str,
    expected_seeds: list[int] | None = None,
) -> dict[str, object]:
    """Import TinyZero launcher artifacts into a canonical result payload."""
    bundle_dir = Path(bundle_dir)
    train_result = _parse_train_result(bundle_dir)
    eval_results = _parse_eval_results(
        bundle_dir,
        recipe_id=recipe_id,
        expected_seeds=expected_seeds,
    )
    trainer_type = _detect_trainer_type(bundle_dir)

    return {
        "experiment_id": experiment_id,
        "recipe_id": recipe_id,
        "train_result": {
            "recipe_id": recipe_id,
            "trainer_type": trainer_type,
            "backend": "tinyzero",
            "status": train_result["status"],
            "metrics": train_result["metrics"],
            "checkpoint_path": train_result["checkpoint_path"],
            "error": train_result["error"],
        },
        "eval_results": eval_results,
    }

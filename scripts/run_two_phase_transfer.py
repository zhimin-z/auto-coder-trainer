#!/usr/bin/env python3
"""Two-phase RL transfer wrapper for exp04 (Countdown -> GSM8K).

Runs two TinyZero bundles in sequence:
  Phase 1: train on Countdown — generates a checkpoint.
  Phase 2: train on GSM8K, resuming actor weights from phase 1's checkpoint.

Usage:
    python scripts/run_two_phase_transfer.py \
        --phase1-recipe-id recipe-exp04-cross-task-transfer-smoke-phase1 \
        --phase2-recipe-id recipe-exp04-cross-task-transfer-smoke-phase2

Both recipes must already have been processed by `act train ... --dry-run`
so their bundles exist under outputs/<recipe-id>/tinyzero/. The wrapper:

  1. Sources phase1 env.sh, runs phase1 bash run.sh.
  2. Globs phase1's checkpoints/global_step_* dir for the latest checkpoint.
  3. Exports ACT_RESUME_FROM_PATH + ACT_RESUME_MODE so the launcher's
     hydra-overrides.txt picks them up via `${oc.env:...}`.
  4. Runs phase2 bash run.sh.

This is glue, not a new launcher backend — both phases are still standard
veRL/TinyZero runs, the only novelty is checkpoint hand-off.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _bundle_dir(recipe_id: str) -> Path:
    return (REPO_ROOT / "outputs" / recipe_id / "tinyzero").resolve()


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the newest `global_step_*` directory written by veRL."""
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(
        (p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("global_step_")),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    return candidates[-1] if candidates else None


def _run_phase(label: str, bundle_dir: Path, env: dict[str, str]) -> None:
    if not bundle_dir.exists():
        raise FileNotFoundError(f"{label} bundle missing: {bundle_dir}")
    print(f"[transfer] === {label}: bash run.sh @ {bundle_dir} ===")
    proc = subprocess.run(
        ["bash", "run.sh"],
        cwd=str(bundle_dir),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"[transfer] {label} failed (exit {proc.returncode}); aborting before next phase."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-recipe-id", required=True)
    ap.add_argument("--phase2-recipe-id", required=True)
    ap.add_argument("--phase1-train-file", help="Override ACT_TRAIN_FILE for phase 1 (default: <bundle>/data/train.parquet)")
    ap.add_argument("--phase1-val-file", help="Override ACT_VAL_FILE for phase 1")
    ap.add_argument("--phase2-train-file", help="Override ACT_TRAIN_FILE for phase 2")
    ap.add_argument("--phase2-val-file", help="Override ACT_VAL_FILE for phase 2")
    args = ap.parse_args()

    phase1 = _bundle_dir(args.phase1_recipe_id)
    phase2 = _bundle_dir(args.phase2_recipe_id)

    # Phase 1
    env1 = os.environ.copy()
    env1.pop("CUDA_VISIBLE_DEVICES", None)  # Let env.sh's :- default win
    env1["ACT_TRAIN_FILE"] = str(args.phase1_train_file or phase1 / "data" / "train.parquet")
    env1["ACT_VAL_FILE"] = str(args.phase1_val_file or phase1 / "data" / "test.parquet")
    # Make sure no stale resume var leaks in.
    env1.pop("ACT_RESUME_FROM_PATH", None)
    env1.pop("ACT_RESUME_MODE", None)
    _run_phase("phase1 (countdown)", phase1, env1)

    # Locate phase 1 checkpoint
    ckpt = _latest_checkpoint(phase1 / "checkpoints")
    if ckpt is None:
        raise SystemExit(
            f"[transfer] phase 1 produced no checkpoint under {phase1 / 'checkpoints'}; "
            f"set trainer.params.save_freq to a positive integer in the phase1 recipe."
        )
    print(f"[transfer] phase1 checkpoint = {ckpt}")

    # Phase 2 — resume from phase1's checkpoint. Note: veRL's resume restores
    # the *trainer state* (actor + optimizer + global_step). For pure-actor
    # transfer we'd want a different path; this is fine for the smoke since
    # we only care that the resume override is honoured end-to-end.
    env2 = os.environ.copy()
    env2.pop("CUDA_VISIBLE_DEVICES", None)
    env2["ACT_TRAIN_FILE"] = str(args.phase2_train_file or phase2 / "data" / "train.parquet")
    env2["ACT_VAL_FILE"] = str(args.phase2_val_file or phase2 / "data" / "test.parquet")
    env2["ACT_RESUME_FROM_PATH"] = str(ckpt)
    env2["ACT_RESUME_MODE"] = "resume_path"
    _run_phase("phase2 (gsm8k, resumed)", phase2, env2)

    print("[transfer] both phases complete.")


if __name__ == "__main__":
    main()

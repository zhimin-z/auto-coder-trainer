# Progress Tracker

> 每个会话结束前更新此文件。新会话开始时读取此文件了解全局状态。

## Current Focus

- (none — 21-build TinyZero experiment library is at POC parity; next session
  picks up whatever the user prioritises: real-GPU verifications, evaluator
  work, or a new sub-project)

## Completed

### Infrastructure

- [x] SWE-Lego backend + SLURM pipeline
- [x] Blog-style report generator
- [x] Judge system (5 checks, 4 verdicts)
- [x] **veRL 0.7.1 upgrade** (was 0.3.x) + TinyZero launcher rewrite for the
      new schema (FSDP1, oc.env interpolation, literal-int hydra fields,
      sft_trainer module rename)
- [x] **Sweep wrapper** [`scripts/run_ablation_sweep.py`](scripts/run_ablation_sweep.py):
      single-axis sweep + `--cartesian` for multi-axis recipes; `--values`,
      `--print-only`, `--ablation-name` flags.
- [x] **Two-phase resume wrapper**
      [`scripts/run_two_phase_transfer.py`](scripts/run_two_phase_transfer.py):
      runs phase 1, locates the latest `global_step_*` checkpoint, exports
      `ACT_RESUME_FROM_PATH` + `ACT_RESUME_MODE=resume_path`, runs phase 2.
- [x] **Launcher mechanisms** (all in
      [`trainers/tinyzero/launcher.py`](trainers/tinyzero/launcher.py)):
  - `${oc.env:ACT_TRAIN_FILE}` / `ACT_VAL_FILE` lazy resolution.
  - `cuda_visible_devices` int / list / str → env.sh.
  - `custom_reward_function: "path:func"` → hydra
    `custom_reward_function.path/name`; `binary` / `""` / `builtin` /
    `default` sentinels fall back to verl's built-in reward.
  - `ACT_PARAM_<KEY>` env-var passthrough for every string-typed
    `trainer.params.*` entry — single reward script can branch on a
    recipe-level switch.
  - `trainer.resume_from_path/mode` via `${oc.env:...,null/disable}` for
    optional resume.
  - FP8 mapping: `fp8_rollout=True` → `rollout.dtype/quantization=fp8`,
    `fp8_actor=True` → `actor.fsdp_config.fp8=True`.
  - LoRA mapping: `model.adapter=lora` → `model.lora_rank/lora_alpha`,
    `full` → explicit `lora_rank=0`.
- [x] **Results bridge upgrade** ([`trainers/tinyzero/results_bridge.py`](trainers/tinyzero/results_bridge.py)):
      parses both SFT (`train/loss`) and GRPO/PPO (`actor/*`, `critic/*`)
      stdout lines into flat DB metrics; auto-detects trainer type from
      `run.sh` so RL recipes don't get mis-tagged as SFT.

### TinyZero experiment library — 21/21 builds verified

End-to-end on real GPU (DB has `success` row + train log archived):

- [x] **exp01** GRPO/GSM8K baseline (0.5B, 64 samples)
- [x] **exp13** 1-shot RLVR (1.5B, 1 sample × 100 epochs × group_size 16,
      ~29 min, `critic/score/mean=1.0` by step 96)
- [x] **exp19-smoke** PPO + Countdown (0.5B, 8 PPO steps, ~88 s, peak ~19 GB)
- [x] **exp04-smoke** two-phase Countdown→GSM8K (0.5B, phase1 step 1–4 →
      phase2 step 5–8 from ckpt, ~2.5 min)

POC verified at the bundle level (sweep wrapper materialises N variants,
each schema-validates and emits the expected hydra/env, no real training):

- [x] **Build 5–9** sweep family: exp08 (rollout-n), exp09 (temperature),
      exp03 (data-size), exp06 (model-size), exp11 (4×6 cartesian)
- [x] **Build 10–11** custom-reward family: exp05 (binary/partial/process),
      exp21 (4 thinking-style modes via `ACT_PARAM_<KEY>`)
- [x] **Build 12–14** algorithm/perf side quests: exp07 (rl/grpo cartesian),
      exp02 (FP8 rollout × actor cartesian), exp10 (LoRA cartesian)
- [x] **Build 15–19** evaluator/multi-turn family: exp12 (pass@k schema),
      exp18 (retention sweep), exp15 (post-processing sweep), exp17
      (multiturn sweep), exp20 (multiturn-thinking sweep)
- [x] **Build 20–21** heavy/independent: exp14 (distill vs grpo —
      surfaces a cross-backend ablation gap), exp16 (multimodal Qwen-VL
      with multi-GPU `[6, 7]` pinning)

See [`recipes/experiments/README.md`](recipes/experiments/README.md) for the
full status table, per-build evidence, and remaining TODOs.

## Blocked

- (none for the experiment library; pre-existing ruff warnings in
  `trainers/upstream/` and `cli/train.py` were not introduced by recent
  work but are still reported by `ruff check .` — not blocking)

## Recent Changes

- feat: 21-build TinyZero experiment library — sweep wrapper, two-phase
  transfer wrapper, custom-reward sentinel, FP8/LoRA hydra mapping,
  `ACT_PARAM_<KEY>` env passthrough, results bridge GRPO/PPO parsing
- feat: Add results bridge for SSD launcher bundles (b58b3ed)
- feat: Add TinyZero experiment scripts and setup for SLURM (c3ec59c)
- feat: Enhance sync and training workflows with ablation support (14ee32e)

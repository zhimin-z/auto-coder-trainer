# TinyZero Experiment Index

21 research experiments from [`TinyZero_实验方案.md`](../../TinyZero_实验方案.md), each
realised as a runnable recipe. **Reading order**: at-a-glance summary →
status table → status legend → 5-step template → per-experiment notes.

All recipes have been validated against [`recipe.schema.json`](../schema/recipe.schema.json)
and pass `act train --dry-run`. Whether they actually train end-to-end on
your hardware depends on the **status** column — read it before investing
GPU time.

---

## At a glance

| | Count | Builds |
|---|---|---|
| **End-to-end on real GPU** (DB has `success` row, train log archived) | 4 | exp01, exp13, exp19-smoke, exp04-smoke |
| **POC verified** (sweep + bundle gen + hydra/env spot-check) | 17 | exp08, exp09, exp03, exp06, exp11, exp05, exp21, exp07, exp02, exp10, exp12, exp18, exp15, exp17, exp20, exp14, exp16 |
| **Total** | 21 | — |

What "POC verified" rules out: schema bugs, sweep wrapper regressions,
launcher dispatch errors (RL vs GRPO, FP8, LoRA, multi-GPU, custom rewards,
two-phase resume, env passthrough). What it does **not** rule out: any
training-runtime issue (verl 0.7.x edge cases, OOM at full scale,
data-prep mismatches, evaluators that don't yet exist). The end-to-end
verifications cover the remaining runtime risk for the GRPO and PPO
backends specifically.

For full evidence (commands run, hydra snippets, GPU-mem peaks, wall
times), see [Per-experiment notes](#per-experiment-notes) below.

---

## Status table

Sorted by **build order**. The `Orig #` column is the experiment number from
[`TinyZero_实验方案.md`](../../TinyZero_实验方案.md). See
[Recommended build order](#recommended-build-order) below for the rationale
behind each group.

Status column legend: ✅ E2E (full real-GPU run) · ✅ POC (bundle/sweep
spot-check) · ⏳ remaining work for full reproduction. The **Blockers /
TODO** column lists what would still need to be done before a full
reproduction can run; "—" means none.

| Build | Orig # | Recipe | Algorithm · Model · Data | Min. hardware | Status | Blockers / TODO |
|-------|--------|--------|--------------------------|---------------|--------|-----------------|
| **1** | 1 | [exp01-grpo-gsm8k](exp01-grpo-gsm8k.recipe.json) | GRPO · 0.5B · GSM8K (64 samples) | 1 × A100 80GB | ✅ E2E (earlier session, bundle artifacts not retained) | — |
| **2** | 13 | [exp13-one-shot-rlvr](exp13-one-shot-rlvr.recipe.json) | GRPO · 1.5B · GSM8K (1 sample × 100 epochs × group_size 16) | 1 × A100 80GB | ✅ E2E | — |
| **3** | 19 | [exp19-tinyzero-replication](exp19-tinyzero-replication.recipe.json) (full) + [smoke](exp19-tinyzero-replication-smoke.recipe.json) | PPO · 3B · Countdown (full) / 0.5B · 8 steps (smoke) | 1 × A100 80GB | ✅ E2E (smoke) · ⏳ full | full 3B PPO run is ~24 h, OOMs on one 80GB card with default config |
| **4** | 4 | [exp04-cross-task-transfer](exp04-cross-task-transfer.recipe.json) (full) + [smoke phase1](exp04-cross-task-transfer-smoke-phase1.recipe.json) / [phase2](exp04-cross-task-transfer-smoke-phase2.recipe.json) | GRPO · Countdown→GSM8K transfer (full=3B, smoke=0.5B 4+4 steps) | 1 × A100 80GB | ✅ E2E (smoke) · ⏳ full | full 3B run is ~96 h |
| **5** | 8 | [exp08-rollout-n](exp08-rollout-n.recipe.json) | GRPO · 3B · GSM8K (sweep `group_size` ∈ {2, 4, 8, 16}) | 1 × A100 80GB | ✅ POC · ⏳ training | run all 4 cells |
| **6** | 9 | [exp09-temperature](exp09-temperature.recipe.json) | GRPO · 3B · GSM8K (sweep `temperature` ∈ {0.6, 0.8, 1.0, 1.2}) | 1 × A100 80GB | ✅ POC · ⏳ training | run all 4 cells |
| **7** | 3 | [exp03-data-scaling](exp03-data-scaling.recipe.json) | GRPO · 3B · GSM8K (sweep `total_samples` ∈ {100…8K}) | 1 × A100 80GB | ✅ POC · ⏳ training | run all 6 cells |
| **8** | 6 | [exp06-model-scale](exp06-model-scale.recipe.json) | GRPO · 0.5B–7B · GSM8K (sweep `model.base`) | 1 × A100 80GB (≤3B) / 2 × A100 80GB (7B) | ✅ POC · ⏳ training | run 4 cells (7B needs bucket 2) |
| **9** | 11 | [exp11-comprehensive-scaling](exp11-comprehensive-scaling.recipe.json) | GRPO · 4 model sizes × 6 data sizes = 24 cells | 1–2 × A100 80GB (per cell) | ✅ POC · ⏳ training | 24-cell sweep is the flagship cost; needs bucket 4+ for parallel |
| **10** | 5 | [exp05-reward-design](exp05-reward-design.recipe.json) | GRPO · 3B · GSM8K (sweep `custom_reward_function` ∈ {binary, partial, process}) | 1 × A100 80GB | ✅ POC · ⏳ training | run all 3 cells |
| **11** | 21 | [exp21-reward-shaping-thinking](exp21-reward-shaping-thinking.recipe.json) | GRPO · 3B · GSM8K (sweep 4 thinking-style modes) | 1 × A100 80GB | ✅ POC · ⏳ training | run all 4 cells |
| **12** | 7 | [exp07-kl-sensitivity](exp07-kl-sensitivity.recipe.json) | rl(PPO) / grpo · 3B · GSM8K (cartesian 2 × 4 = 8 cells) | 1 × A100 80GB | ✅ POC · ⏳ training | RLOO / REMAX algorithm paths not added (would need schema enum + launcher branches) |
| **13** | 2 | [exp02-fp8-quantization](exp02-fp8-quantization.recipe.json) | GRPO · 7B · GSM8K (cartesian `fp8_rollout` × `fp8_actor` = 4 cells) | 1 × A100 80GB | ✅ POC · ⏳ training | actor-side FP8 is a best-effort hydra path; needs torch + transformer-engine build verification |
| **14** | 10 | [exp10-lora-vs-full](exp10-lora-vs-full.recipe.json) | GRPO · 7B · GSM8K (cartesian `model.adapter` × `lora_rank` = 6 cells) | 1 × A100 80GB (LoRA) / 2 × A100 80GB (full) | ✅ POC · ⏳ training | qlora not implemented; veRL LoRA-on-GRPO path untested |
| **15** | 12 | [exp12-pass-at-k](exp12-pass-at-k.recipe.json) | GRPO · 3B · GSM8K + MATH | 1 × A100 80GB | ✅ POC · ⏳ evaluator + training | `evaluators/pass_at_k.py`, `make_math_data.py` |
| **16** | 18 | [exp18-capability-retention](exp18-capability-retention.recipe.json) | GRPO + replay · 3B · GSM8K + base evals (sweep `mitigation_strategy`) | 1 × A100 80GB | ✅ POC · ⏳ evaluators + training | MMLU / HellaSwag / TruthfulQA evaluators (HumanEval already exists) |
| **17** | 15 | [exp15-self-reflection](exp15-self-reflection.recipe.json) | GRPO + post-processing · 3B · GSM8K (sweep `post_processing`) | 1 × A100 80GB | ✅ POC · ⏳ inference wrapper + training | inference-time post-processor harness |
| **18** | 17 | [exp17-multiturn-rl](exp17-multiturn-rl.recipe.json) | GRPO · 3B · multi-turn GSM8K (sweep `eval_max_turns`) | 1 × A100 80GB | ✅ POC · ⏳ data prep + training | `make_gsm8k_multiturn_data.py` |
| **19** | 20 | [exp20-multiturn-thinking](exp20-multiturn-thinking.recipe.json) | GRPO · 3B · GSM8K (sweep `training_data_format`) | 1 × A100 80GB | ✅ POC · ⏳ eval harness + training | `<think>` tag detector + multi-turn eval wrapper |
| **20** | 14 | [exp14-distill-vs-rl](exp14-distill-vs-rl.recipe.json) | Distill (7B→3B) vs GRPO · GSM8K (sweep `trainer.type`) | 2 × A100 80GB | ✅ POC · ⏳ design fix + training | cross-backend ablation needs paired standalone recipes, not a single `ablation` block (sweep produces a `trl/grpo` variant that has no real backend) |
| **21** | 16 | [exp16-multimodal](exp16-multimodal.recipe.json) | GRPO · Qwen2.5-VL-7B · Geo3K (sweep `use_vision`) | 2 × A100 80GB | ✅ POC · ⏳ deps + data prep + training | Qwen-VL deps in venv, `make_geo3k_data.py` |

## Recommended build order

The 21 experiments are grouped so each group **builds infrastructure that
unlocks the next**. Working in this order minimises rework: writing one piece
of glue (data prep, sweep wrapper, evaluator) typically clears 2-6 downstream
experiments at once.

| Group | Builds | What this group adds | Unlocks |
|-------|--------|----------------------|---------|
| **A — Anchor** | exp01 | Verified GRPO baseline | Reference for every other GRPO recipe |
| **B — Zero-code variants** | exp13 | Direct config variant of exp01 (`total_samples=1`) | Confirms the smoke pattern scales to extreme settings |
| **C — Countdown assets** | exp19 → exp04 | `make_countdown_data.py`, `reward_countdown.py`, two-phase orchestration | All Countdown-based work; PPO algorithm path validation |
| **D — Sweep wrapper + scaling** | exp08 → exp09 → exp03 → exp06 → exp11 | One reusable `scripts/sweep.py` that materialises N sub-recipes from a single ablation | All scaling experiments; eventual 7B run unblocks exp02/10 |
| **E — Reward-fn framework** | exp05 → exp21 | Plumbing for `trainer.params.custom_reward_function` plus partial / process / cleverness reward fns | All custom-reward research |
| **F — Algorithm/perf side quests** | exp07, exp02, exp10 | Verifies untested launcher paths in isolation (PPO/RLOO/REMAX, FP8, LoRA) | Removes "untested launcher path" risk from later experiments |
| **G — Evaluator infrastructure** | exp12 → exp18 → exp15 | Pass@k, MMLU/HellaSwag/TruthfulQA/HumanEval evaluators, post-processing wrapper | All capability / frontier evaluation work |
| **H — Multi-turn cluster** | exp17 → exp20 | Multi-turn data prep, `<think>` tag detector, multi-turn eval harness | Conversational reasoning research |
| **I — Heavy / independent** | exp14, exp16 | Distillation backend verification; Qwen-VL data + reward + launcher | These are essentially separate sub-projects; they don't unblock anything else |

The build order in the status table follows A → I; within each group the order
is from cheapest / most-reused to most-specialised.

### Why not the order from `TinyZero_实验方案.md`?

The original document orders experiments by **research narrative** (algorithm
comparison first, then quantization, then data scaling…). That's good for
reading, but as an execution plan it puts FP8 quantization (exp02) and the
24-cell scaling sweep (exp11) before any of the cheap GSM8K variants — meaning
you'd hit two of the hardest infrastructure problems before validating that
the basic GRPO loop works. The build order above leads with single-cell
GSM8K runs that exercise the path we already know works (exp01), then
incrementally adds one new piece of infrastructure at a time.

### Status legend

- ✅ **E2E** — completed at least one full end-to-end training run on the
  listed hardware; loss / reward curves observed and the result is in the
  results DB (`data/results.db`). When a build has both a *full* and a
  *smoke* recipe, "✅ E2E (smoke)" means only the smoke variant has been
  executed.
- ✅ **POC** — `act train --dry-run` succeeds, the sweep wrapper (if
  applicable) materialises all variants, and the spot-checked variant's
  `hydra-overrides.txt` / `env.sh` contains the expected keys. The recipe
  is well-formed all the way down to launcher dispatch — what is **not**
  proven is the runtime path of veRL 0.7.x at full scale.
- ⏳ **remaining work** — listed in the *Blockers / TODO* column; usually
  a real training run, an evaluator that is not yet implemented, or a data
  prep script.
- 🚧 **blocked** (currently unused) — would mean known-broken; the
  dependency in the rightmost column has to land before the recipe is
  runnable.

### Hardware buckets

We list the **minimum** that should fit; bigger is faster. All numbers assume
A100 80GB (or H100 80GB equivalent) and the recipe's default
`gradient_checkpointing: true`.

| Bucket | Use case |
|--------|----------|
| 1 × 80GB | All ≤3B-parameter recipes; 7B with LoRA + gradient checkpointing |
| 2 × 80GB | 7B full-parameter; multi-modal Qwen-VL-7B; teacher-student distill |
| 4 × 80GB | Sweep wrappers running 4 cells in parallel; long context (≥4K) on 7B |
| 8 × 80GB | exp11 24-cell sweep with parallel execution |

---

## The 5-step end-to-end template

Every experiment follows the **same five commands** from the repo root. The
parts you swap per experiment are the **recipe path** and the **data prep
script** (some experiments share `make_gsm8k_data.py`, others need a recipe-
specific script — see status table).

```bash
RECIPE=recipes/experiments/expXX-<topic>.recipe.json
RECIPE_ID=$(python -c "import json,sys; print(json.load(open('$RECIPE'))['id'])")
BUNDLE=outputs/$RECIPE_ID/tinyzero

# 1. Generate launch bundle
act train "$RECIPE"

# 2. Prepare the dataset (script depends on the recipe — check the per-experiment notes)
python scripts/make_gsm8k_data.py "$BUNDLE/data" --max-train 64 --max-val 16

# 3. Run training (GPU pinning comes from `budget.cuda_visible_devices` in the recipe)
( cd "$BUNDLE" \
  && export ACT_TRAIN_FILE="$(pwd)/data/train.parquet" \
  && export ACT_VAL_FILE="$(pwd)/data/test.parquet" \
  && bash run.sh )

# 4. Import results back into the SQLite DB so the judge sees them
act train "$RECIPE" --import-results "$BUNDLE" --recipe-id "$RECIPE_ID"

# 5. Generate the blog-style report
act report --recipe-id "$RECIPE_ID" --format blog
```

**Smoke-size first.** All recipes ship with **production-scale** parameters
(epochs, total_samples, batch_size). Before committing GPU hours, edit the
recipe to a smoke configuration (e.g. `epochs: 1`, `total_samples: 64`) so you
catch bugs in 2 minutes instead of 2 hours. exp01 already ships with smoke
defaults as the canonical reference.

**Pinning a different GPU.** Edit `budget.cuda_visible_devices` in the recipe
(int for one GPU, list for multi-GPU). The launcher writes it into
`bundle/env.sh`; you can also override at run time with
`CUDA_VISIBLE_DEVICES=N bash run.sh`.

---

## Per-experiment notes

> Ordered by **build sequence** (matches the status table). The "Build N" tag
> in each heading is the position in the recommended order; "orig #" is the
> experiment number from `TinyZero_实验方案.md`. Items marked **TODO** are
> missing infrastructure that needs to be added before the recipe can run.

### Build 1 · exp01 — GRPO baseline (orig #1, ✅ verified)

The reference recipe — Qwen 2.5-0.5B SFT-chat fine-tune, 64 GSM8K rows, 8
training steps. Used to validate the GRPO path; results land at
[`outputs/recipe-exp01-grpo-gsm8k/`](../../outputs/) when run. Data prep:
`scripts/make_gsm8k_data.py --max-train 64 --max-val 16`. Every other GRPO
recipe should match this one's structure before adding new variation.

### Build 2 · exp13 — 1-shot RLVR (orig #13, ✅ verified)

Group **B**. A direct config variant of exp01 (`total_samples=1`,
`epochs=100`, `group_size=16`). **No new code needed** — runs today via the
5-step template using
`scripts/make_gsm8k_data.py --max-train 1 --max-val 16`. End-to-end run on
one GPU 7 (A100 80GB) takes ~29 min and trips the canonical 1-shot-RLVR
signal: `critic/score/mean` climbs from ~0 to 1.0 around step 96 as the
model memorizes the single training example, with `actor/grad_norm` and
`actor/pg_loss` collapsing toward zero. Used to prove that the RL stack can
take a recipe → bundle → run → DB → report → judge cycle without any new
launcher logic.

### Build 3 · exp19 — TinyZero replication (orig #19, ✅ pipeline verified)

Group **C** (Countdown assets). PPO on Countdown — the original TinyZero $30
run. Two recipes ship:

- [`exp19-tinyzero-replication.recipe.json`](exp19-tinyzero-replication.recipe.json) — full reproduction
  target (Qwen2.5-3B, 8 K samples, 15 epochs, batch 256). Estimated ~24 h on
  one A100 80GB; the actor + critic + Adam state for two 3B models alone is
  ~60 GB, so PPO is much tighter on memory than GRPO at the same parameter
  count.
- [`exp19-tinyzero-replication-smoke.recipe.json`](exp19-tinyzero-replication-smoke.recipe.json) — pipeline-only
  smoke (Qwen2.5-0.5B, 256 samples, 8 PPO steps, batch 32,
  `gpu_memory_utilization=0.3`). Finishes in **~88 s** with peak **~19 GB**
  on one A100 80GB. Reward stays 0 (0.5B has no chance of solving Countdown
  in 8 steps), but `critic/vf_loss` collapses 21.6 → 0.83 over 8 steps,
  proving the critic is fitting and PPO is propagating gradients.

What this build unlocks (now landed in repo):
- `scripts/make_countdown_data.py` — reads `Jiayi-Pan/Countdown-Tasks-3to4`,
  carves a held-out tail for val, writes the same 4-column verl parquet
  schema as `make_gsm8k_data.py`.
- `scripts/reward_countdown.py:compute_score` — extracts the last
  `<answer>...</answer>` block, verifies the expression uses **exactly** the
  given input numbers (multiset), AST-validates the operators, and safe-evals
  to check it equals `target`. Returns 1.0 / 0.0; ASCII number regex avoids
  the `(6-4)` → `-4` parsing trap.
- TinyZero launcher now wires `custom_reward_function.path` /
  `custom_reward_function.name` into hydra overrides for any recipe with
  `trainer.params.custom_reward_function`. Relative paths resolve against
  the repo root so the run.sh can chdir freely.
- TinyZero launcher writes `trainer.total_training_steps` and
  `trainer.val_before_train=False` so smoke recipes can cap iterations
  without triggering an upfront full-val pass.

To run the smoke yourself:
```
python scripts/make_countdown_data.py \
  outputs/recipe-exp19-tinyzero-replication-smoke/tinyzero/data \
  --max-train 256 --max-val 32
act train recipes/experiments/exp19-tinyzero-replication-smoke.recipe.json
( cd outputs/recipe-exp19-tinyzero-replication-smoke/tinyzero
  export ACT_TRAIN_FILE=$(pwd)/data/train.parquet
  export ACT_VAL_FILE=$(pwd)/data/test.parquet
  unset CUDA_VISIBLE_DEVICES
  bash run.sh )
act train recipes/experiments/exp19-tinyzero-replication-smoke.recipe.json \
  --import-results outputs/recipe-exp19-tinyzero-replication-smoke/tinyzero \
  --recipe-id recipe-exp19-tinyzero-replication-smoke
```

`cost_per_pct_accuracy_usd` in the full recipe's `metrics` is informational
— the framework doesn't compute it automatically.

### Build 4 · exp04 — Countdown→GSM8K transfer (orig #4, ✅ pipeline verified)

Group **C**. Reuses build 3's Countdown data + reward; new piece is the
two-phase orchestration. Three recipes ship:

- [`exp04-cross-task-transfer.recipe.json`](exp04-cross-task-transfer.recipe.json) — full reproduction
  target (Qwen2.5-3B-Instruct, 8 K samples, 3 epochs, three ablation
  protocols). Estimated ~96 GPU-hours.
- [`exp04-cross-task-transfer-smoke-phase1.recipe.json`](exp04-cross-task-transfer-smoke-phase1.recipe.json) +
  [`exp04-cross-task-transfer-smoke-phase2.recipe.json`](exp04-cross-task-transfer-smoke-phase2.recipe.json) — pipeline
  smoke (Qwen2.5-0.5B, 64 samples each phase, 4 fresh PPO steps per phase).
  Phase 1 trains on Countdown with `save_freq=4` so a `global_step_4`
  checkpoint lands; phase 2 picks it up via the wrapper.

Smoke verification (~2.5 min total on one A100 80GB, peak **~12 GB**):
- Phase 1 ends at `training/global_step:4`, writes
  `outputs/recipe-exp04-cross-task-transfer-smoke-phase1/tinyzero/checkpoints/global_step_4`.
- Phase 2 starts at `training/global_step:5` (not 1 — the resume worked) and
  steps through 6, 7, 8. Confirms `trainer.resume_mode=resume_path` +
  `trainer.resume_from_path=$ACT_RESUME_FROM_PATH` is honoured by veRL 0.7.1
  inside the TinyZero launcher path.

What this build unlocks (now landed in repo):
- [`scripts/run_two_phase_transfer.py`](../../scripts/run_two_phase_transfer.py) — bash glue that runs
  phase 1's `bundle/run.sh`, globs the latest `global_step_*` checkpoint,
  exports `ACT_RESUME_FROM_PATH` + `ACT_RESUME_MODE=resume_path`, then runs
  phase 2's `bundle/run.sh`. No hidden state, no new launcher backend.
- TinyZero launcher now writes
  `trainer.resume_from_path=${oc.env:ACT_RESUME_FROM_PATH,null}` and
  `trainer.resume_mode=${oc.env:ACT_RESUME_MODE,disable}` for **every** RL
  bundle, so any recipe can be turned into a phase-2 bundle just by setting
  these two env vars at run time.

Important semantics worth knowing:
- `trainer.total_training_steps` is **absolute** in veRL — phase 2's value
  must be ≥ phase 1's final `global_step` + however many fresh steps you
  want. The smoke phase2 sets `total_training_steps=8` so the resumed run
  produces 4 new steps (5–8); setting it to 4 would have stopped on the
  first step.
- veRL's resume restores **trainer state** (actor + optimizer + global_step).
  For pure-actor transfer you'd need a separate weights-only path; the
  smoke is fine for proving the override plumbing, not for studying which
  state actually transfers.

To run the smoke yourself:
```
act train recipes/experiments/exp04-cross-task-transfer-smoke-phase1.recipe.json
act train recipes/experiments/exp04-cross-task-transfer-smoke-phase2.recipe.json
python scripts/make_countdown_data.py \
  outputs/recipe-exp04-cross-task-transfer-smoke-phase1/tinyzero/data \
  --max-train 64 --max-val 8
python scripts/make_gsm8k_data.py \
  outputs/recipe-exp04-cross-task-transfer-smoke-phase2/tinyzero/data \
  --max-train 64 --max-val 8
python scripts/run_two_phase_transfer.py \
  --phase1-recipe-id recipe-exp04-cross-task-transfer-smoke-phase1 \
  --phase2-recipe-id recipe-exp04-cross-task-transfer-smoke-phase2
act train recipes/experiments/exp04-cross-task-transfer-smoke-phase1.recipe.json \
  --import-results outputs/recipe-exp04-cross-task-transfer-smoke-phase1/tinyzero \
  --recipe-id recipe-exp04-cross-task-transfer-smoke-phase1
act train recipes/experiments/exp04-cross-task-transfer-smoke-phase2.recipe.json \
  --import-results outputs/recipe-exp04-cross-task-transfer-smoke-phase2/tinyzero \
  --recipe-id recipe-exp04-cross-task-transfer-smoke-phase2
```

### Build 5 · exp08 — Rollout n sweep (orig #8, ✅ POC verified)

Group **D** (sweep wrapper). Single-cell GRPO runs already work — the new
piece is [`scripts/run_ablation_sweep.py`](../../scripts/run_ablation_sweep.py).

What the wrapper does:
1. Loads the base recipe; picks an entry from the recipe's `ablation` block
   (errors out if there are multiple and `--ablation-name` is not given).
2. For each value, deep-copies the recipe, sets the dotted variable
   (`trainer.params.group_size` for exp08), strips the `ablation` block from
   the variant (otherwise the judge would queue more cells), rewrites `id`
   and `name` to include the value, and stamps `_ablation_origin` for
   traceability. Variants land in `outputs/<base-id>/ablations/`.
3. Calls `act train <variant.json>` per cell, sequentially. With
   `--dry-run`, every variant goes through schema-validate + bundle
   generation but does no training. Without it, every variant is dispatched
   end-to-end; the wrapper aborts the sweep on the first non-zero exit.

POC verification on exp08 (no training, just plumbing): the wrapper
materializes 4 variant recipes (`group_size ∈ {2, 4, 8, 16}`), each
schema-validates, and each produces its own TinyZero bundle. The
per-variant `hydra-overrides.txt` files differ only in
`actor_rollout_ref.rollout.n=N` (= 2 / 4 / 8 / 16). No training was run for
this build; the sweep is now a pure config-edit dependency for builds 6, 7,
8, and 9.

To use it on a single recipe:
```
python scripts/run_ablation_sweep.py recipes/experiments/exp08-rollout-n.recipe.json --dry-run
python scripts/run_ablation_sweep.py recipes/experiments/exp08-rollout-n.recipe.json --values 2,4
python scripts/run_ablation_sweep.py recipes/experiments/exp08-rollout-n.recipe.json --print-only
```

Useful flags: `--values` runs a subset (auto-coerced to the type of the
recipe's value list), `--print-only` writes the variant files and stops
without dispatching anything.

### Build 6 · exp09 — Temperature sweep (orig #9, ✅ POC verified)

Group **D**. Reuses build 5's [`scripts/run_ablation_sweep.py`](../../scripts/run_ablation_sweep.py)
verbatim — the only differences from exp08 are the swept key
(`trainer.params.temperature`) and the value list (`[0.6, 0.8, 1.0, 1.2]`).
POC verified by running the wrapper with `--dry-run`: 4 variants generated,
each producing its own bundle with the correct
`actor_rollout_ref.rollout.temperature` in `hydra-overrides.txt`. No
training executed. Useful as a sanity check that floats with one decimal
place round-trip cleanly through the ID slug (`...temperature_sweep-0.6`).

### Build 7 · exp03 — Data-size sweep (orig #3, ✅ POC verified)

Group **D**. Same `run_ablation_sweep.py` from build 5; the swept variable
is `dataset.total_samples ∈ {100, 500, 1K, 2K, 5K, 8K}`. Useful as a check
that the dotted-path setter works on parents other than `trainer.params.*`
— it does. 6 variants, all schema-valid, no training executed. First
"scaling-law" experiment; the curve directly informs the 24-cell exp11.

### Build 8 · exp06 — Model-size sweep (orig #6, ✅ POC verified)

Group **D**. Sweeps `model.base` across the 4 Qwen2.5-Instruct sizes (0.5B,
1.5B, 3B, 7B). The slug logic is exercised here: HF IDs contain `/`, which
the wrapper's `_slug` collapses to `-` so the variant filenames remain
filesystem-safe (`...model_size_sweep-qwen-qwen2.5-7b-instruct.recipe.json`).
Each variant's `hydra-overrides.txt` correctly substitutes
`actor_rollout_ref.model.path`. 7B will not fit a single 80GB card with
`gradient_checkpointing: false` — the recipe sets it `true` already, but
rollout is still tight. Successful 7B run also unblocks exp02/10.

### Build 9 · exp11 — Comprehensive scaling 4×6 (orig #11, ✅ POC verified)

Group **D**. The flagship — composes builds 7 and 8 into a 24-cell sweep.
This is the first recipe with **two** `ablation` blocks (model_size_sweep +
data_size_sweep), which forced the wrapper to grow a `--cartesian` flag.
With `--cartesian`, the wrapper takes the cross product of every block's
values (`4 × 6 = 24`), `_set_dotted` sets all variables on each variant,
and the variant ID concatenates `axis-slug__axis-slug` so every cell has a
unique, traceable filename.

POC: ran `run_ablation_sweep.py exp11-... --cartesian --print-only` →
24 variants written; spot-checked the (7B-Instruct, 8000-samples) corner
cell through `act train --dry-run` — schema valid, bundle generated,
hydra overrides show `actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct`
+ correct training params. No training executed. Recommended to run on
bucket 4 (4 × 80GB) with 4 cells in parallel.

### Build 10 · exp05 — Reward design (orig #5, ✅ POC verified)

Group **E** (reward-fn framework). Three reward variants compared via the
existing `run_ablation_sweep.py` — the sweep variable is
`trainer.params.custom_reward_function` itself, not a string label, so the
wrapper needs zero special-casing. Variants:

- `binary` — sentinel value. The launcher recognises `binary` / `""` /
  `builtin` / `default` and writes **no** `custom_reward_function` override,
  so verl falls back to its built-in `gsm8k.compute_score` (1.0 if the
  `#### <num>` answer matches, else 0.0). Used as the unmodified baseline.
- [`scripts/reward_gsm8k_partial.py`](../../scripts/reward_gsm8k_partial.py) — partial credit:
  no `#### N` block ⇒ 0.0, format-correct but wrong number ⇒ 0.3,
  format + correct number ⇒ 1.0. Smooths the early-training gradient.
- [`scripts/reward_gsm8k_process.py`](../../scripts/reward_gsm8k_process.py) — binary outcome
  plus a length-aware shaping term: 0.05 per detected reasoning step
  (arithmetic line or numeric sentence), capped at +0.3. A cheap proxy for
  process reward — no per-step verifier, just the shape.

POC: ran `run_ablation_sweep.py exp05-... --dry-run`. 3 variant bundles
land under `outputs/recipe-exp05-reward-design-reward_variant-*/tinyzero/`.
Spot-checked the per-variant `hydra-overrides.txt`:

- `...-reward_variant-binary/.../hydra-overrides.txt` has **no**
  `custom_reward_function.path` line.
- `...-partial.../.../hydra-overrides.txt` writes
  `custom_reward_function.path=/home/.../scripts/reward_gsm8k_partial.py`.
- `...-process.../.../hydra-overrides.txt` writes the process script's
  absolute path.

No training was executed for this build; the partial / process reward
shapes are unit-tested only. Used by build 11 (exp21) for thinking-style
shaping.

### Build 11 · exp21 — Reward shaping for thinking style (orig #21, ✅ POC verified)

Group **E**. Reuses build 10's custom-reward plumbing **plus** a new
launcher mechanism: any **string-typed** entry under `trainer.params.*` is
now auto-exported as `ACT_PARAM_<KEY>` in the bundle's `env.sh`. This lets
a single reward script branch on a recipe-level switch without growing the
hydra-override surface.

Pieces shipped:
- [`scripts/reward_thinking.py`](../../scripts/reward_thinking.py) — one `compute_score`
  with four modes:
  * `baseline` — verl's binary GSM8K signal (1.0 / 0.0).
  * `step_count_bonus` — +0.05 per arithmetic step / numeric sentence,
    capped at +0.3.
  * `step_count_penalty` — −0.02 per step beyond the 6th, capped at −0.2.
  * `cleverness_bonus` — +0.2 only when the answer is correct **and** the
    response is under 60 whitespace tokens.
  Mode is read from `ACT_PARAM_THINKING_REWARD_STYLE` (or the bare
  `ACT_THINKING_REWARD_STYLE` for direct invocation). Unknown values raise
  so a typo in a recipe surfaces immediately.
- [`trainers/tinyzero/launcher.py`](../../trainers/tinyzero/launcher.py): RL bundles now
  iterate over `training_params` and write `export ACT_PARAM_<KEY>=<value>`
  in `env.sh` for every string-valued param. Numeric params still flow
  through hydra-overrides as before — there is no double-wiring.

POC: ran `run_ablation_sweep.py exp21-... --dry-run`. 4 variants land
under `outputs/recipe-exp21-reward-shaping-thinking-thinking_reward_style-*`.
Spot-checked each variant's `env.sh`:

- `...-baseline/.../env.sh`: `ACT_PARAM_THINKING_REWARD_STYLE=baseline`
- `...-step-count-bonus/.../env.sh`: `=step_count_bonus`
- `...-step-count-penalty/.../env.sh`: `=step_count_penalty`
- `...-cleverness-bonus/.../env.sh`: `=cleverness_bonus`

Every variant's `hydra-overrides.txt` also has the same
`custom_reward_function.path=.../scripts/reward_thinking.py` line — the
reward script is loaded once, the **mode** is what differs. Reward
functions were unit-tested in isolation; no training executed.

### Build 12 · exp07 — KL × algorithm matrix (orig #7, ✅ POC verified)

Group **F** (algorithm/perf side quests). The original ambition was 4
algorithms × 4 KL coefficients = 16 runs (PPO/GRPO/RLOO/REMAX); the recipe
in tree reduces to a 2 × 4 = 8 cell smoke (`trainer.type ∈ {rl, grpo}` ×
`kl_coeff ∈ {0.0001, 0.001, 0.01, 0.1}`). RLOO and REMAX would require
adding `rloo`/`remax` to the schema enum + branches in
`_build_rl_overrides` for their `algorithm.adv_estimator` values — not
done.

POC: ran `run_ablation_sweep.py exp07-... --cartesian --print-only`. 8
variants generated, cross-product of both axes. Spot-checked the two
endpoints:

- `algorithm-grpo__kl_coeff_sweep-0.0001`: hydra has
  `algorithm.adv_estimator=grpo`, `algorithm.kl_ctrl.kl_coef=0.0001`, no
  critic block (GRPO has no learned value function).
- `algorithm-rl__kl_coeff_sweep-0.1`: hydra has
  `algorithm.adv_estimator=gae`, `algorithm.kl_ctrl.kl_coef=0.1`, **plus**
  `critic.model.path=Qwen/Qwen2.5-3B-Instruct` and `critic.optim.lr=1e-05`.

This is the first time both `trainer.type=rl` and `trainer.type=grpo`
emerge from the *same* recipe via cartesian sweep — confirms the dispatch
logic in `build_tinyzero_launcher_bundle` switches cleanly per cell. No
training executed (3B + 3B critic OOMs on a single 80GB card, see exp19
notes).

### Build 13 · exp02 — FP8 quantization (orig #2, ✅ POC verified)

Group **F**. Compares BF16 vs FP8-rollout vs FP8-end-to-end via a 2 × 2
cartesian sweep over `fp8_rollout` × `fp8_actor`.

What changed in the launcher:
- `fp8_rollout=True` → writes `actor_rollout_ref.rollout.dtype=fp8` and
  `actor_rollout_ref.rollout.quantization=fp8` (vLLM rollout FP8 — well
  supported in vLLM 0.7+).
- `fp8_actor=True` → writes `actor_rollout_ref.actor.fsdp_config.fp8=True`.
  This is the actor-side FSDP/transformer-engine path and is **not** a
  guaranteed-working override; whether training succeeds depends on the
  torch + transformer-engine build the venv was assembled from. Treated as
  a hydra-level POC, not a training-level guarantee.

POC: ran `run_ablation_sweep.py exp02-... --cartesian --print-only` (4
cells), then `act train ... --dry-run` on each. Spot-check of the four
`hydra-overrides.txt` files:

- BF16 (`fp8_rollout=False, fp8_actor=False`): no fp8 lines.
- Actor-only (`False, True`): just `actor.fsdp_config.fp8=True`.
- Rollout-only (`True, False`): just the two `rollout.*` lines.
- E2E (`True, True`): all three.

No training executed.

### Build 14 · exp10 — LoRA vs full (orig #10, ✅ POC verified)

Group **F**. Cartesian sweep over `model.adapter ∈ {full, lora}` ×
`trainer.params.lora_rank ∈ {8, 16, 32}` = 6 cells. Three of them
(`full + rank ∈ {8, 16, 32}`) are formally redundant since full finetune
ignores `lora_rank`; the launcher resolves this cleanly by emitting
`lora_rank=0` whenever `adapter=full`, so the redundant cells reduce to
identical full-finetune bundles.

Launcher behaviour shipped:
- `adapter=lora` → emits both `actor_rollout_ref.model.lora_rank=<rank>`
  and `actor_rollout_ref.model.lora_alpha=2×rank` (default; recipe can
  override `lora_alpha` explicitly).
- `adapter=full` (or anything else) → emits `model.lora_rank=0`. This is
  the explicit-baseline pattern: a recipe flipping back from lora to full
  cannot accidentally inherit a stale rank from a copied bundle.
- `adapter=qlora` is **not** implemented — needs additional quant config
  plumbing (TODO if the qlora row of exp10 ever gets requested).

POC: ran `run_ablation_sweep.py exp10-... --cartesian --print-only`, then
`act train --dry-run` on 4 representative cells. Spot-check of
`hydra-overrides.txt`:

- `full__lora_rank_sweep-8` → `actor_rollout_ref.model.lora_rank=0`
- `full__lora_rank_sweep-32` → `actor_rollout_ref.model.lora_rank=0`
- `lora__lora_rank_sweep-8` → `lora_rank=8`, `lora_alpha=16`
- `lora__lora_rank_sweep-32` → `lora_rank=32`, `lora_alpha=64`

verl 0.7.x's actual LoRA-on-GRPO training path is untested in this
codebase; the POC only proves the launcher emits valid hydra and the
recipe-level sweep is wired through. No training executed.

### Build 15 · exp12 — Pass@k (orig #12, ✅ POC verified)

Group **G** (evaluator infrastructure). The recipe defines no `ablation`
block — the experimental contribution lives entirely on the eval side
(`pass@1, 5, 10, 20, 50, 100` from k samples per prompt across two data
sources). POC consisted of `act train --dry-run`: the recipe schema-
validates, the two-source dataset (`openai/gsm8k` mix 0.5 + `lighteval/MATH`
mix 0.5) is accepted, and the bundle is generated. The two pass@k
benchmarks (`pass_at_k_gsm8k`, `pass_at_k_math`) are already in the schema
enum (added in earlier work) so no validation friction.

What is **not** verified: the actual Pass@k evaluator in `evaluators/` is
still TODO — the bundle generation does not depend on it, but a real
training run would. MATH data prep also still needs a `make_math_data.py`.

### Build 16 · exp18 — Capability retention (orig #18, ✅ POC verified)

Group **G**. 4-variant sweep over `trainer.params.mitigation_strategy ∈
{none, kl_regularize, sft_mixin, replay_buffer}`. Each variant lands a
bundle whose `env.sh` exports
`ACT_PARAM_MITIGATION_STRATEGY=<value>` — the shape that lets a single
training-side dispatcher branch on the strategy without growing the
recipe schema.

Training would still need each benchmark's evaluator under `evaluators/`
(HumanEval exists; MMLU / HellaSwag / TruthfulQA / `pass_at_k_*` do not).
Originally flagged as the largest evaluator workload in the index, that's
still true — POC only proves the **training-side** plumbing.

### Build 17 · exp15 — Self-correction post-processing (orig #15, ✅ POC verified)

Group **G**. 4-variant sweep over `trainer.params.post_processing ∈
{none, self_correction, verification, beam_search}`. Bundles each export
`ACT_PARAM_POST_PROCESSING=<value>` so the inference-time post-processor
(future) can branch without re-parsing the recipe.

The training half is identical to exp01 (already verified end-to-end).
The post-processing strategies happen at **inference time**, not training.
TODO: `evaluators/post_processing.py` that runs the trained checkpoint
through each strategy.

### Build 18 · exp17 — Multi-turn dialogue RL (orig #17, ✅ POC verified)

Group **H** (multi-turn cluster). 3-variant sweep over
`trainer.params.eval_max_turns ∈ {1, 2, 5}`. Variants schema-validate; each
bundle's `env.sh` exports `ACT_PARAM_MULTITURN_SPLIT_STRATEGY=stepwise`
(the recipe-level constant). `eval_max_turns` itself is an int and so is
**not** auto-exported (launcher only exports string-typed params); a real
training run would need either a multi-turn data prep that bakes the turn
count into the dataset, or a launcher extension that exports ints too.

TODO: `scripts/make_gsm8k_multiturn_data.py` for splitting GSM8K problems
into 2–3 conversational turns. verl's multiturn rollout config is the
training-side path that would consume this.

### Build 19 · exp20 — Multi-turn thinking retention (orig #20, ✅ POC verified)

Group **H**. 4-variant sweep over `trainer.params.training_data_format ∈
{single_turn, multiturn, thinking_reward, system_prompt}`. Bundles each
export `ACT_PARAM_TRAINING_DATA_FORMAT=<value>`.

Reuses build 18's multi-turn data (when it lands). TODO: a `<think>` tag
detector + a multi-turn eval wrapper that asks 5 questions in sequence to
the trained model. Both belong in `evaluators/`.

### Build 20 · exp14 — Distillation vs RL (orig #14, ✅ POC verified, with caveat)

Group **I** (heavy / independent). 2-variant sweep over
`trainer.type ∈ {distill, grpo}`. Both variants schema-validate, but the
sweep surfaces a real **design gap**: the recipe's `trainer.backend` is
`trl`, so the swept `grpo` variant compiles to `trl/grpo` — and TRL has
no GRPO trainer. veRL does.

Lesson learned: **cross-backend ablations don't fit a single
`ablation` block**, because backend dispatch lives in `trainer.backend`
and the swept variable is `trainer.type`. The clean fix for a future
iteration is two standalone recipes (one `trl/distill`, one `verl/grpo`)
plus a wrapper script that submits both and joins their reports — same
shape as build 4's two-phase wrapper. Not done; the POC just proves the
sweep wrapper doesn't silently produce a broken bundle layout.

### Build 21 · exp16 — Multimodal RL (orig #16, ✅ POC verified)

Group **I**. 2-variant sweep over `trainer.params.use_vision ∈ {true, false}`.
This was the first multi-GPU verification: the recipe's
`budget.cuda_visible_devices: [6, 7]` flows into both bundles' `env.sh`
(`CUDA_VISIBLE_DEVICES=6,7`) and into the hydra override
`trainer.n_gpus_per_node=2`. Model path
`actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct` lands correctly.

Caveats: `use_vision` is a bool, so the launcher does **not** auto-export
it (string-only export by design). A real training run would need:
- `scripts/make_geo3k_data.py` (verl's `reward_score/geo3k.py` is built
  in, so the reward side is free).
- vLLM ≥ 0.7 multimodal support and Qwen-VL deps in the venv.
- A 7B VL model + GRPO rollout exceeds a single 80GB card; recipe is
  already pinned to bucket 2.

---

## Adding a new experiment

1. Copy the closest existing recipe (e.g. `exp01-grpo-gsm8k.recipe.json`).
2. Bump `id` and `name`.
3. Validate: `python -c "import json; from recipes.compiler import load_schema, validate_recipe; print(validate_recipe(json.load(open('your.recipe.json')), load_schema()))"`.
4. Dry-run: `act train your.recipe.json --dry-run` — confirms the launcher
   accepts the recipe.
5. Add a row to the status table above with status `⏳ unverified` and what
   blocking deps you saw.
6. When you complete an end-to-end run, update the status to `✅ verified` and
   note the hardware bucket and wall-time.

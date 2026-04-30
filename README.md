# Auto-Coder-Trainer

**Research Operating System for Coding Agent Training**

> From papers to trained models in a closed loop — automatically collect research,
> compose training recipes, run experiments, and generate publication-ready reports.

```
         collect           compose            train             report
      ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
      │  arXiv   │────▶│  Recipe  │────▶│ SFT / RL │────▶│  Blog /  │
      │  GitHub  │     │   IR     │     │ Distill  │     │  LaTeX   │
      └──────────┘     └──────────┘     └──────────┘     └──────────┘
           ▲                                                   │
           └────────────── judge verdict ──────────────────────┘
                          feedback loop
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [One-Command Pipeline](#one-command-pipeline)
- [Architecture](#architecture)
- [CLI Reference](#cli-reference)
- [Recipe IR](#recipe-ir)
- [Training Backends](#training-backends)
- [Experiment Judge](#experiment-judge)
- [Blog-Style Reports](#blog-style-reports)
- [Recovery & State](#recovery--state)
- [ARIS Research Plane](#aris-research-plane)
- [Prompt Caching](#prompt-caching)
- [Project Structure](#project-structure)
- [Development](#development)
- [Status](#status)
- [License](#license)

---

## Quick Start

### Prerequisites

- Linux + at least one NVIDIA GPU (the smoke test below needs ~2 GB of free GPU memory)
- **Python 3.11+** — Python 3.10 will fail because `scipy>=1.17` (pulled in by the `[all]` extra) dropped 3.10 support

### Install (≈ 5 min)

```bash
git clone https://github.com/chenghaoYang/auto-coder-trainer.git
cd auto-coder-trainer
pip install -e ".[all,dev]"

# flash-attn is imported at module load by verl's SFT trainer but is not
# pulled in automatically (it has to compile or fetch a CUDA-specific wheel).
pip install flash-attn --no-build-isolation
```

### Smoke test (≈ 2 min, 1 GPU)

This runs a 0.5B-parameter SFT on 96 rows of toy arithmetic, then sends the
results back through the judge and report generator — exercising every stage
(`act train` → `torchrun` → FSDP → checkpoint → judge → report) end-to-end
before you invest in real data.

> The recipe pins training to **GPU 7** by default (see
> [`smoke-tinyzero.recipe.json`](recipes/examples/smoke-tinyzero.recipe.json#L36-L40)).
> If your machine has a different free GPU, edit `budget.cuda_visible_devices`
> in that file before continuing — single integer for one GPU, e.g. `0`, or a
> list like `[0, 1]` for multi-GPU (multi-GPU additionally requires bumping
> `trainer.params.batch_size` to a multiple of GPU count).

Run all five steps from the **repo root** in one go:

```bash
# 1. Generate the launch bundle (writes outputs/recipe-smoke-001/)
act train recipes/examples/smoke-tinyzero.recipe.json

# 2. Materialise the toy dataset (verl expects a `messages` parquet column)
python scripts/make_toy_data.py outputs/recipe-smoke-001/tinyzero/toy_data

# 3. Run training inside the bundle (GPU pinning comes from the recipe)
( cd outputs/recipe-smoke-001/tinyzero \
  && export ACT_TRAIN_FILE=$(pwd)/toy_data/train.parquet \
  && export ACT_VAL_FILE=$(pwd)/toy_data/val.parquet \
  && bash run.sh )

# 4. Import the bundle's results back into the SQLite DB so the judge sees them
act train recipes/examples/smoke-tinyzero.recipe.json \
  --import-results outputs/recipe-smoke-001/tinyzero \
  --recipe-id recipe-smoke-001

# 5. Generate the blog-format report
act report --recipe-id recipe-smoke-001 --format blog
```

**Expected**: `train/loss` falls from ~1.4 to <0.1, `val/loss` prints, a
checkpoint lands in `outputs/recipe-smoke-001/tinyzero/checkpoints/`,
`results/train_exit_code.txt` is `0`, the judge returns a verdict (most likely
`needs_rerun` — toy data has no baseline to compare against), and the report is
written to `outputs/recipe-smoke-001/reports/report.md`.

### Real training

Once smoke test passes, swap the model, dataset, and benchmarks in your own
recipe — see [Recipe IR](#recipe-ir) for the full schema and
[`recipes/examples/`](recipes/examples/) for SFT / RL / distillation
templates. The same five steps work for real training; the difference is your
recipe references real HF datasets (you provide the parquet conversion) and
real benchmarks (e.g. `swe-bench-lite`, `humaneval`).

The closed-loop variant `act pipeline` automates collect → compose → train →
judge → loop, but its `act collect` stage currently requires the optional
`aris/` package; if `act collect` fails with `No module named 'aris'`, use
the five-step recipe-driven flow above instead.

### Research experiment library

For the 21 reproducible TinyZero/veRL experiments described in
[`TinyZero_实验方案.md`](TinyZero_实验方案.md) (algorithm comparison,
data scaling, Pass@k, distillation, multi-modal, etc.), see the experiment
index at [`recipes/experiments/README.md`](recipes/experiments/README.md).
Each experiment has a runnable recipe, a status marker, a minimum-hardware
bucket, and a list of remaining blockers.

Current verification state (see the index for per-build evidence):

- **End-to-end on real GPU**: exp01 (GRPO/GSM8K baseline), exp13 (1-shot
  RLVR, 100 epochs × 1 sample on 1.5B), exp19-smoke (PPO + Countdown +
  custom reward, 0.5B, 8 steps), exp04-smoke (two-phase
  Countdown→GSM8K with checkpoint hand-off).
- **POC verified (recipe schema-validates and the launcher emits the
  correct hydra/env, no real training observed)**: the remaining 17
  experiments. Many ship a smoke recipe alongside the full reproduction
  recipe so the GRPO/PPO/LoRA/FP8 dispatch can be checked in seconds.

The infrastructure that landed for these verifications — sweep wrapper,
two-phase resume wrapper, custom-reward sentinel, FP8/LoRA hydra mapping,
`ACT_PARAM_<KEY>` env-var passthrough — is documented in the
per-experiment notes under
[`recipes/experiments/README.md`](recipes/experiments/README.md).

---

## One-Command Pipeline

The `act pipeline` command orchestrates the **entire agent team** as a closed loop:

```bash
# Full pipeline: collect → compose → train → judge → report
act pipeline --query "coding agent training" --report-format blog

# Start from an existing recipe (skip collect/compose)
act pipeline --recipe recipes/examples/baseline-sft.recipe.json

# Control iteration depth and output
act pipeline --recipe recipe.json --max-iterations 5 --report-dir reports/

# Validate without training
act pipeline --recipe recipe.json --dry-run
```

**Automatic decision logic** after each judge verdict:

| Verdict | Action |
| --- | --- |
| `accept` | Generate final blog report, done |
| `needs_rerun` | Re-run missing seeds or full experiment, loop |
| `needs_ablation` | Run ablation variants, loop |
| `reject` | Generate report with failure analysis, stop |

The pipeline iterates up to `--max-iterations` (default 3), then generates a final report regardless.

---

## Architecture

```
┌───────────────────────────────────────────────────┐
│           ARIS Research Plane  (aris/)             │
│   75+ skills  ·  4 MCP servers  ·  arXiv tools    │
└────────────────────┬──────────────────────────────┘
                     │ method atoms
              ┌──────▼──────┐
              │  Recipe IR  │  ← JSON Schema (Draft 2020-12)
              │ (recipes/)  │
              └──────┬──────┘
                     │ compiled config
┌────────────────────▼──────────────────────────────┐
│           Training Plane                           │
│   trainers/ · evaluators/ · judge/ · results/      │
│   7 backends · 2 evaluators · 5 judge checks       │
└───────────────────────────────────────────────────┘
```

**Two planes, one IR.** The ARIS Research Plane handles paper collection, idea generation, and research orchestration. The Training Plane handles experiment execution, evaluation, and result persistence. The **Recipe IR** — a structured JSON schema — bridges them.

---

## CLI Reference

| Command | Description | Example |
| --- | --- | --- |
| `act pipeline` | Full closed-loop pipeline | `act pipeline --query "RL for code" --report-format blog` |
| `act collect` | Discover papers & repos → method atoms | `act collect "coding agent training"` |
| `act compose` | Assemble atoms → Recipe IR JSON | `act compose --atoms swe-fuse,entropy-rl` |
| `act train` | Execute experiment from recipe | `act train recipe.json` |
| `act report` | Generate report (blog / markdown / latex) | `act report --recipe-id R --format blog` |
| `act status` | Show experiments & open tasks | `act status --open-only` |
| `act rerun` | Auto-dispatch pending judge tasks | `act rerun --recipe-id R` |

**Claude Code skills** (equivalent):

```
/collect "coding agent trajectory training"
/compose swe-fuse + entropy-rl
/train recipes/examples/trajectory-rl.recipe.json
/report exp_001
```

---

## Recipe IR

Every experiment is defined by a **Recipe IR** — a structured JSON validated against [`recipe.schema.json`](recipes/schema/recipe.schema.json):

```jsonc
{
  "id": "recipe-trajectory-rl-001",
  "name": "GRPO on Coding Trajectories with Entropy-Aware Reward",
  "model": { "base": "Qwen/Qwen2.5-Coder-7B-Instruct", "adapter": "full" },
  "dataset": {
    "sources": [{ "name": "swe-rebench-trajectories", "path": "openhands/..." }],
    "filters": [{ "type": "issue_free" }, { "type": "length", "params": { "max_turns": 30 } }]
  },
  "trainer": {
    "type": "grpo",                       // sft | rl | grpo | distill | dpo
    "backend": "tinyzero",                // trl | verl | tinyzero | openr1 | agent_distill | redi
    "reward": { "type": "entropy_aware" }
  },
  "eval": {
    "benchmarks": ["swe-bench-verified"],
    "seeds": [42, 123, 456]
  },
  "ablation": [{
    "name": "reward_type",
    "variable": "trainer.reward.type",
    "values": ["binary_pass", "entropy_aware"]
  }],
  "budget": { "max_gpu_hours": 48, "gpu_type": "A100-80GB" }
}
```

**Distillation recipes** add an optional `distill` block for teacher-trajectory training:

```jsonc
{
  "trainer": { "type": "distill", "backend": "trl" },
  "distill": {
    "strategy": "trajectory",
    "teacher_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "stages": ["positive_sft", "pairwise_refine"],
    "refine_algorithm": "dpo",
    "condense": { "strategy": "edge_preserving", "max_chars": 12000 }
  }
}
```

See [`recipes/examples/`](recipes/examples/) for complete examples covering SFT, RL/GRPO, and distillation.

---

## Training Backends

| Backend | Type | Used For | Framework |
| --- | --- | --- | --- |
| **TRL** | Native | SFT, distillation, DPO refinement | [huggingface/trl](https://github.com/huggingface/trl) |
| **veRL** | Native | RL, GRPO, PPO | [volcengine/verl](https://github.com/volcengine/verl) |
| **TinyZero** | Launcher | Baseline SFT & RL launch bundles | [Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero) |
| **Open-R1** | Launcher | Reasoning recipe launcher | [huggingface/open-r1](https://github.com/huggingface/open-r1) |
| **Agent Distillation** | Launcher | Teacher-agent trajectory distillation | [Nardien/agent-distillation](https://github.com/Nardien/agent-distillation) |
| **REDI** | Launcher | Negative-signal reinforcement distillation | [Tim-Siu/reinforcement-distillation](https://github.com/Tim-Siu/reinforcement-distillation) |

**Native** trainers run directly. **Launcher** backends generate Hydra configs + shell scripts for external execution. See [UPSTREAM_INTEGRATION.md](UPSTREAM_INTEGRATION.md) for the upstream-first integration policy.

---

## Experiment Judge

Every experiment passes through an automated judge before results are accepted:

| Check | Description |
| --- | --- |
| Baseline alignment | Verifies a baseline run exists; allows 5% regression tolerance |
| Seed consistency | Confirms all seeds evaluated; coefficient of variation < 10% |
| Ablation coverage | Ensures ablation experiments cover all recipe variables |
| Deduplication | Prevents redundant experiments via config hash matching |
| Failure attribution | Analyzes and explains underperforming experiments |

**Verdicts**: `accept` · `reject` · `needs_ablation` · `needs_rerun`

When used with `act pipeline`, verdicts automatically drive the next action — no manual intervention needed.

---

## Blog-Style Reports

Reports can be generated in three formats: `markdown`, `latex`, and `blog`.

The **blog** format follows the structure of Sebastian Raschka's [LoRA Insights](https://lightning.ai/blog/lora-insights) — an experiment diary style optimized for readability:

```bash
act report --recipe-id recipe-001 --format blog
```

**Report structure**:

| Section | Content |
| --- | --- |
| **TL;DR** | Key takeaways, best metrics, seed stability summary |
| **Introduction** | Motivation, method description, source papers |
| **Experimental Setup** | Model, dataset, hyperparameters, hardware, evaluation config |
| **Experiments** | Sequential diary — each run with question, setup, results table, judge verdict, key finding |
| **Ablation Studies** | Per-variable comparison tables with best setting highlighted |
| **Cost & Efficiency** | GPU budget, performance-per-GPU-hour analysis |
| **Practical Recommendations** | Synthesized from judge suggestions, ablation results, seed stability |
| **Reproducibility** | Commands to re-run, config hashes, seed list |
| **Conclusion** | Accept/reject summary, best results, next steps |

---

## Recovery & State

The project maintains resumable state so agents can pick up where they left off:

| Store | Location | Purpose |
| --- | --- | --- |
| Result DB | `data/results.db` | Experiments, eval runs, verdicts, artifacts, tasks (SQLite, WAL mode) |
| Task ledger | `outputs/<recipe-id>/task-ledger.{json,md}` | Human/agent-readable snapshot of completed and pending work |
| Execution plan | `outputs/<recipe-id>/execution-plan.{json,md}` | Next steps for blocked or prepared experiments |

```bash
act status --open-only                                    # What's pending?
act status --recipe-id recipe-001 --output status.md      # Save status report
act rerun --recipe-id recipe-001                          # Auto-dispatch pending tasks
```

Override the DB path for testing: `ACT_RESULTS_DB=/path/to/test.db`

---

## ARIS Research Plane

The [ARIS (Auto-Research-In-Sleep)](aris/README.md) subsystem provides **75+ Claude Code skills** for autonomous ML research:

| Category | Skills |
| --- | --- |
| **Research** | `/research-lit`, `/arxiv`, `/idea-discovery`, `/novelty-check` |
| **Experiment** | `/experiment-plan`, `/experiment-bridge`, `/run-experiment` |
| **Writing** | `/paper-write`, `/paper-figure`, `/paper-slides`, `/paper-poster` |
| **Review** | `/auto-review-loop`, `/research-review`, cross-model review |
| **Pipeline** | `/research-pipeline` — end-to-end from idea to paper |

Plus **4 MCP servers**: claude-review, feishu-bridge, llm-chat, minimax-chat.

See the [ARIS README](aris/README.md) for full documentation.

---

## Prompt Caching

The `prompt_cache/` module implements cache-safe prompt construction for long-running agent sessions, based on Anthropic's prefix-matching cache architecture.

| Layer | Content | Stability |
| --- | --- | --- |
| 0 | System prompt + tool definitions | Frozen at session start |
| 1 | Project context (CLAUDE.md) | Frozen per project |
| 2 | Session context (task, plan) | Append-only within session |
| 3 | Conversation messages | Dynamic, not cached |

**Six rules** (enforced by `prompt_cache/rules.py`): static prefix ordering, messages for updates, tool set stability, model consistency, state via tools, compaction prefix sharing.

```python
from prompt_cache import PromptBuilder, CacheMonitor

builder = PromptBuilder()
builder.set_system_prompt("You are a coding agent...")
builder.set_tools([...])                    # Set once, frozen
builder.add_project_context(claude_md)       # Layer 1
builder.add_session_context("Task: ...")     # Layer 2

monitor = CacheMonitor(alert_threshold=0.8)
monitor.record(response.usage)
if not monitor.is_healthy():
    print(monitor.diagnose_cache_miss())
```

---

## Project Structure

```
auto-coder-trainer/
│
├── cli/                          # CLI entry points
│   ├── main.py                   #   `act` command dispatcher
│   ├── pipeline.py               #   Full agent team orchestrator
│   ├── collect.py                #   Paper/repo discovery → method atoms
│   ├── compose.py                #   Atoms → Recipe IR
│   ├── train.py                  #   Recipe → training execution
│   ├── report.py                 #   Results → blog / markdown / LaTeX
│   ├── status.py                 #   Task & experiment summary
│   └── rerun.py                  #   Auto-dispatch pending tasks
│
├── recipes/                      # Recipe IR layer
│   ├── schema/recipe.schema.json #   JSON Schema (Draft 2020-12)
│   ├── registry/                 #   Method atom registry
│   ├── examples/                 #   3 example recipes (SFT, RL, distill)
│   └── compiler.py               #   Recipe → TrainingConfig compiler
│
├── trainers/                     # Training backends
│   ├── base.py                   #   Abstract TrainResult / EvalResult
│   ├── sft/                      #   SFT trainer (TRL)
│   ├── rl/                       #   RL/GRPO trainer (veRL)
│   ├── distill/                  #   Distillation trainer (SFT + DPO)
│   ├── tinyzero/                 #   TinyZero launcher
│   ├── upstream/                 #   Open-R1, Agent Distill, REDI launchers
│   └── utils/                    #   Budget, checkpoints, LoRA, seeds
│
├── evaluators/                   # Evaluation harness
│   ├── swe_bench.py              #   SWE-bench integration
│   ├── pass_at_k.py              #   pass@k metrics
│   └── runner.py                 #   Multi-benchmark orchestration
│
├── judge/                        # Experiment judge (5 checks, 4 verdicts)
│   ├── judge.py                  #   Core arbiter
│   ├── baseline.py               #   Baseline alignment
│   ├── ablation.py               #   Ablation validation
│   ├── attribution.py            #   Failure analysis
│   └── dedup.py                  #   Config hash deduplication
│
├── results/                      # Persistence & reporting
│   ├── db.py                     #   SQLite result database (8 tables)
│   ├── schema.sql                #   DB schema
│   ├── ledger.py                 #   Task ledger for crash recovery
│   └── report_generator.py       #   Markdown / LaTeX / blog reports
│
├── prompt_cache/                 # Cache-safe prompt construction
│   ├── builder.py                #   4-layer prompt builder
│   ├── monitor.py                #   Cache hit rate monitoring
│   ├── compaction.py             #   Context compression
│   └── rules.py                  #   6 cache safety rules
│
├── aris/                         # ARIS Research Plane
│   ├── skills/                   #   75+ Claude Code skills
│   ├── mcp-servers/              #   4 MCP servers
│   ├── tools/                    #   arXiv fetcher, utilities
│   └── docs/                     #   Setup guides
│
├── skills/                       # Top-level Claude Code skills
│   ├── collect/                  #   /collect
│   ├── compose/                  #   /compose
│   ├── train/                    #   /train
│   └── report/                   #   /report
│
├── scripts/                      # Data prep + orchestration helpers
│   ├── make_gsm8k_data.py        #   GSM8K → verl parquet
│   ├── make_countdown_data.py    #   Countdown → verl parquet
│   ├── make_toy_data.py          #   smoke-test arithmetic dataset
│   ├── reward_countdown.py       #   custom reward: Countdown
│   ├── reward_gsm8k_partial.py   #   custom reward: GSM8K partial credit
│   ├── reward_gsm8k_process.py   #   custom reward: GSM8K + step shaping
│   ├── reward_thinking.py        #   custom reward: 4-mode thinking shaping
│   ├── run_ablation_sweep.py     #   ablation sweep (single + cartesian)
│   └── run_two_phase_transfer.py #   phase-1 → checkpoint → phase-2 wrapper
│
├── tests/                        # Test suite
├── pyproject.toml                # Package config (Python >= 3.10)
├── Makefile                      # Convenience targets
└── UPSTREAM_INTEGRATION.md       # Upstream integration policy
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[all,dev]"

# Run tests
make test                     # or: python -m pytest tests/ -q

# Lint
make lint                     # or: ruff check .

# Run specific test
python -m pytest tests/test_pipeline_and_blog_report.py -v
```

**Optional install extras**:

| Extra | Packages |
| --- | --- |
| `sft` | trl, transformers, datasets, peft, accelerate |
| `rl` | verl |
| `eval` | swebench |
| `dev` | pytest, ruff |
| `all` | All of the above |

---

## Status

- [x] ARIS Research Plane (75+ skills, 4 MCP servers)
- [x] Recipe IR JSON Schema + compiler
- [x] CLI automation (`collect`, `compose`, `train`, `report`, `status`, `rerun`)
- [x] **Full pipeline orchestrator** (`act pipeline`) — closed-loop agent team
- [x] **Blog-style report generator** — LoRA Insights-inspired experiment diary
- [x] Experiment judge (5 checks, 4 verdicts, auto-decision loop)
- [x] SFT trainer (TRL backend, Full/LoRA/QLoRA)
- [x] RL/GRPO trainer (veRL 0.7.1 backend, 4 reward types)
- [x] PPO trainer with critic (veRL `main_ppo`, validated on Countdown smoke)
- [x] Distillation trainer (trajectory SFT + optional DPO refinement)
- [x] SWE-bench & pass@k evaluators
- [x] Upstream launchers (TinyZero, Open-R1, Agent Distillation, REDI)
- [x] Persistent experiment recovery (SQLite DB, task ledgers, execution plans)
- [x] Prompt cache infrastructure (builder, monitor, compaction, 6 rules)
- [x] **TinyZero experiment library** — 21 recipes, 4 verified end-to-end on
      GPU, 17 verified at the bundle/sweep level. See
      [`recipes/experiments/README.md`](recipes/experiments/README.md).
- [x] **Sweep wrapper** ([`scripts/run_ablation_sweep.py`](scripts/run_ablation_sweep.py))
      — single-axis and `--cartesian` (multi-axis) ablation sweeps.
- [x] **Two-phase resume wrapper**
      ([`scripts/run_two_phase_transfer.py`](scripts/run_two_phase_transfer.py))
      — checkpoint hand-off for transfer experiments (exp04).
- [x] Tests: tinyzero launcher + bridge suites green (8 tests)
- [ ] Case studies and reproductions
- [ ] Pre-existing ruff warnings under `trainers/upstream/` and `cli/train.py`

---

## License

[MIT](LICENSE)

## Citation

```bibtex
@software{auto_coder_trainer,
  title   = {Auto-Coder-Trainer: Research Operating System for Coding Agent Training},
  author  = {Chenghao Yang},
  year    = {2025},
  url     = {https://github.com/chenghaoYang/auto-coder-trainer}
}
```

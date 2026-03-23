# Recipe Compiler — Developer Guide

The Recipe Compiler (`recipes/compiler.py`) translates Recipe IR JSON into executable `TrainingConfig` objects consumed by the training plane.

## Pipeline

```
Recipe JSON  →  normalize  →  validate  →  compile  →  TrainingConfig
                (strip None)  (JSON Schema)  (map fields)  (dataclass)
```

### 1. Normalization (`normalize_recipe`)

Recursively removes `None` values from the recipe dict. This ensures that optional fields left unset by the composer do not cause schema validation failures.

### 2. Validation (`validate_recipe`)

Validates the recipe against `recipes/schema/recipe.schema.json` (JSON Schema Draft 2020-12). Returns a list of human-readable error strings. An empty list means the recipe is valid.

### 3. Compilation (`compile_recipe`)

Maps recipe fields into a `TrainingConfig` dataclass:

| Recipe Field | TrainingConfig Field | Notes |
|---|---|---|
| `id` | `recipe_id` | Unique recipe identifier |
| `trainer.type` | `trainer_type` | `sft`, `rl`, `grpo`, `distill`, `dpo` |
| `trainer.backend` | `backend` | Auto-defaults: `trl` for SFT/DPO/distill, `verl` for RL/GRPO |
| `model` | `model_config` | `{base, size, adapter}` |
| `dataset` | `data_config` | `{sources, filters, total_samples}` |
| `trainer.params` + `trainer.reward` | `training_params` | Merged into a single dict |
| `distill` | `distill_config` | Strategy, teacher, stages, refinement |
| `eval` | `eval_config` | Benchmarks, metrics, seeds (default: `[42, 123, 456]`) |
| `ablation` | `ablation_configs` | List of ablation sweep specs |
| `budget` | `budget` | GPU hours, type, cost limit |

## TrainingConfig Dataclass

```python
@dataclass
class TrainingConfig:
    recipe_id: str
    trainer_type: str          # sft | rl | grpo | distill | dpo
    backend: str               # trl | verl | tinyzero | openr1 | agent_distill | redi
    model_config: dict         # {base: str, size: str?, adapter: str}
    data_config: dict          # {sources: list, filters: list, total_samples: int?}
    training_params: dict      # All training hyperparameters + reward config
    distill_config: dict       # Distillation-specific settings
    eval_config: dict          # {benchmarks: list, metrics: list, seeds: list}
    ablation_configs: list     # [{name, variable, values}]
    budget: dict               # {max_gpu_hours, gpu_type, max_cost_usd}
```

## Backend Resolution

When the recipe does not specify `trainer.backend` explicitly, the compiler applies these defaults:

| Trainer Type | Default Backend |
|---|---|
| `sft`, `dpo`, `distill` | `trl` |
| `rl`, `grpo` | `verl` |

External backends (`tinyzero`, `openr1`, `agent_distill`, `redi`) must always be specified explicitly in the recipe.

## Trainer Registry

After compilation, `cli/train.py` uses the **trainer registry** (`trainers/registry.py`) to resolve the correct trainer class:

```python
from trainers.registry import get_trainer_class, register

# Look up built-in trainer
cls = get_trainer_class("sft", "trl")  # → SFTTrainer

# Register a custom trainer
register("my_algo", "my_backend", MyTrainerClass)
```

Resolution order:
1. Exact match on `(trainer_type, backend)`
2. Fallback match on `(trainer_type, None)` — catches any backend
3. `None` if no match (triggers execution plan generation)

### Adding a New Backend

To add a new training backend:

1. Create a trainer class inheriting from `BaseTrainer`
2. Implement `prepare_data()`, `train()`, `evaluate()`
3. Register it in `trainers/registry.py`:
   ```python
   register("sft", "my_framework", MySFTTrainer)
   ```
4. Add the backend name to the JSON Schema enum in `recipes/schema/recipe.schema.json`

No changes to `cli/train.py` dispatch logic are needed.

## CLI Usage

```bash
# Validate and compile a recipe (no training)
python3 -m recipes.compiler recipes/examples/baseline-sft.recipe.json

# Train with dry-run (generates execution plan)
python3 -m cli.main train recipes/examples/baseline-sft.recipe.json --dry-run

# Full training run
python3 -m cli.main train recipes/examples/baseline-sft.recipe.json --output-dir outputs/
```

## Schema Reference

The full JSON Schema is at `recipes/schema/recipe.schema.json`. Key constraints:

- **Required fields**: `id`, `name`, `model`, `dataset`, `trainer`
- **Trainer types**: `sft`, `rl`, `dpo`, `grpo`, `distill`
- **Backends**: `trl`, `verl`, `tinyzero`, `openr1`, `agent_distill`, `redi`
- **Reward types**: `binary_pass`, `weighted_pass`, `entropy_bonus`, `entropy_aware`, `length_penalty`, `composite`
- **Benchmarks**: `swe-bench-lite`, `swe-bench-verified`, `swe-rebench`, `humaneval`, `mbpp`, `custom`

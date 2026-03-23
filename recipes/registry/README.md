# Method Atoms Registry

Method atoms are reusable building blocks extracted from research papers. Each atom encapsulates a specific technique (reward function, data filter, training algorithm variant, etc.) as a **recipe fragment** that can be composed into full training recipes.

## Atom Structure

```json
{
  "id": "atom-grpo-base",
  "name": "Group Relative Policy Optimization",
  "source_paper": "2402.03300",
  "category": "training_algorithm",
  "description": "GRPO training with group-relative advantage normalization.",
  "tags": ["rl", "grpo"],
  "validated": true,
  "dependencies": [],
  "conflicts": [],
  "recipe_fragment": {
    "trainer": {
      "type": "grpo",
      "params": { "ppo_epochs": 4, "kl_coeff": 0.05 }
    }
  }
}
```

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (prefix with `atom-`) |
| `name` | string | Human-readable name |
| `source_paper` | string | arXiv ID, DOI, or `"n/a"` for custom atoms |
| `category` | enum | `reward_shaping`, `data_curation`, `training_algorithm`, `evaluation`, `distillation` |
| `description` | string | What this atom contributes |
| `recipe_fragment` | object | Partial recipe JSON merged during composition |

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `dependencies` | list[string] | Atom IDs this atom requires |
| `conflicts` | list[string] | Atom IDs incompatible with this atom |
| `tags` | list[string] | Free-form tags for filtering |
| `validated` | boolean | Whether tested in a real training run |

## How Composition Works

When `cli compose` builds a recipe, it:

1. Selects atoms based on the research idea and method analysis
2. Deep-merges `recipe_fragment` fields from each atom into the base recipe
3. Validates the composed recipe against `recipes/schema/recipe.schema.json`
4. Checks for dependency/conflict violations

## Adding a New Atom

1. Identify the technique from a paper or experiment
2. Express it as a minimal `recipe_fragment`
3. Add the atom to `method_atoms.json`
4. Set `validated: false` until tested in a real run
5. Document `dependencies` and `conflicts` if applicable

"""Recipe Compiler — translates Recipe IR JSON into executable training configurations.

Usage:
    python -m recipes.compiler recipes/examples/baseline-sft.recipe.json

Reads a Recipe IR JSON file, validates it against the schema, and compiles it
into a training configuration that can be consumed by trainers/.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any


SCHEMA_PATH = Path(__file__).parent / "schema" / "recipe.schema.json"


@dataclass
class TrainingConfig:
    """Compiled training configuration ready for trainer consumption."""
    recipe_id: str
    trainer_type: str
    backend: str
    model_config: dict[str, Any]
    data_config: dict[str, Any]
    training_params: dict[str, Any]
    eval_config: dict[str, Any]
    ablation_configs: list[dict[str, Any]]
    budget: dict[str, Any]


def load_schema() -> dict:
    """Load the Recipe IR JSON Schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_recipe(recipe: dict, schema: dict) -> list[str]:
    """Validate a recipe against the schema. Returns list of errors (empty if valid)."""
    # TODO: implement with jsonschema
    raise NotImplementedError("Schema validation not yet implemented")


def compile_recipe(recipe: dict) -> TrainingConfig:
    """Compile a validated Recipe IR into an executable TrainingConfig."""
    # TODO: implement compilation logic
    raise NotImplementedError("Recipe compilation not yet implemented")


def main():
    """CLI entry point for recipe compilation."""
    if len(sys.argv) < 2:
        print(f"Usage: python -m recipes.compiler <recipe.json>")
        sys.exit(1)

    recipe_path = Path(sys.argv[1])
    with open(recipe_path) as f:
        recipe = json.load(f)

    schema = load_schema()
    errors = validate_recipe(recipe, schema)
    if errors:
        print(f"Validation errors: {errors}")
        sys.exit(1)

    config = compile_recipe(recipe)
    print(f"Compiled recipe '{config.recipe_id}' -> {config.backend}/{config.trainer_type}")


if __name__ == "__main__":
    main()

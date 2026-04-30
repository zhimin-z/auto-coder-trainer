#!/usr/bin/env python3
"""Ablation sweep wrapper.

Reads a recipe's `ablation` block, materializes one *variant* recipe per
value into `outputs/<base-id>/ablations/`, and either dry-runs each variant
or dispatches them through `act train` sequentially.

Why this is a wrapper, not a launcher backend: a sweep is just N independent
runs. The launcher already knows how to compile any single recipe to a
TinyZero/SWE-Lego/etc. bundle. We just need to enumerate the cells.

Usage:
    python scripts/run_ablation_sweep.py recipes/experiments/exp08-rollout-n.recipe.json --dry-run
    python scripts/run_ablation_sweep.py <recipe> --ablation-name rollout_n_sweep --values 2,4
    python scripts/run_ablation_sweep.py <recipe>  # full sweep, dispatches via act train

The variable path follows recipe-IR dotted form (e.g. `trainer.params.group_size`).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _slug(value: Any) -> str:
    """Filesystem-safe slug for a value (e.g. 4 -> '4', 1e-5 -> '1e-05', 'hi there' -> 'hi-there')."""
    s = str(value).lower()
    s = re.sub(r"[^a-z0-9.+-]+", "-", s).strip("-")
    return s or "v"


def _set_dotted(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = d
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Variable path {path!r} missing at segment {part!r}")
        cur = cur[part]
    if not isinstance(cur, dict):
        raise KeyError(f"Variable path {path!r} does not point to a dict-leaf parent")
    cur[parts[-1]] = value


def _select_ablation(recipe: dict[str, Any], wanted: str | None) -> dict[str, Any]:
    ablations: list[dict[str, Any]] = recipe.get("ablation", []) or []
    if not ablations:
        raise SystemExit(f"Recipe {recipe.get('id')!r} has no ablation block — nothing to sweep.")
    if wanted is None:
        if len(ablations) > 1:
            names = ", ".join(a.get("name", "?") for a in ablations)
            raise SystemExit(
                f"Recipe has multiple ablations ({names}); pick one with --ablation-name "
                f"or use --cartesian to sweep them all jointly."
            )
        return ablations[0]
    for abl in ablations:
        if abl.get("name") == wanted:
            return abl
    raise SystemExit(f"No ablation named {wanted!r} in recipe.")


def _materialize_cartesian(
    base_recipe: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Cross-product across **all** ablation blocks.

    For an N-block ablation list with sizes (k1, k2, ..., kN) this produces
    `prod(k_i)` variants — each setting *every* swept variable. Use sparingly
    — exp11 (4 × 6) already lands at 24 cells.
    """
    import itertools

    ablations = base_recipe.get("ablation") or []
    if not ablations:
        raise SystemExit("--cartesian needs at least one ablation block in the recipe.")

    out_dir.mkdir(parents=True, exist_ok=True)
    variant_paths: list[Path] = []
    base_id = base_recipe["id"]

    grids = [[(a["variable"], a.get("name", "axis"), v) for v in a["values"]] for a in ablations]
    total = 1
    for g in grids:
        total *= len(g)
    print(f"[sweep] cartesian product: {' × '.join(str(len(g)) for g in grids)} = {total} cells")

    for combo in itertools.product(*grids):
        variant = copy.deepcopy(base_recipe)
        variant.pop("ablation", None)

        slug_parts: list[str] = []
        origin_assignments: list[dict[str, Any]] = []
        for variable, ax_name, value in combo:
            _set_dotted(variant, variable, value)
            slug_parts.append(f"{ax_name}-{_slug(value)}")
            origin_assignments.append({"variable": variable, "axis": ax_name, "value": value})

        variant_id = f"{base_id}-cartesian-" + "__".join(slug_parts)
        variant["id"] = variant_id
        variant["name"] = base_recipe.get("name", base_id) + " [" + " ; ".join(
            f"{a['variable']}={a['value']}" for a in origin_assignments
        ) + "]"
        variant["_ablation_origin"] = {
            "base_recipe_id": base_id,
            "kind": "cartesian",
            "assignments": origin_assignments,
        }

        variant_path = out_dir / f"{variant_id}.recipe.json"
        variant_path.write_text(json.dumps(variant, indent=2) + "\n")
        variant_paths.append(variant_path)

    print(f"[sweep] wrote {len(variant_paths)} cartesian variants under {out_dir}")
    return variant_paths


def _materialize_variants(
    base_recipe: dict[str, Any],
    ablation: dict[str, Any],
    out_dir: Path,
    selected_values: list[Any] | None,
) -> list[Path]:
    variable = ablation["variable"]
    values = selected_values if selected_values is not None else ablation["values"]

    out_dir.mkdir(parents=True, exist_ok=True)
    variant_paths: list[Path] = []
    base_id = base_recipe["id"]

    for value in values:
        variant = copy.deepcopy(base_recipe)
        # Strip the ablation block from the variant — each variant is a single
        # cell, not a sweep itself, and otherwise the judge would queue more.
        variant.pop("ablation", None)
        _set_dotted(variant, variable, value)

        variant_id = f"{base_id}-{ablation['name']}-{_slug(value)}"
        variant["id"] = variant_id
        variant["name"] = f"{base_recipe.get('name', base_id)} [{variable}={value}]"
        variant["_ablation_origin"] = {
            "base_recipe_id": base_id,
            "ablation_name": ablation["name"],
            "variable": variable,
            "value": value,
        }

        variant_path = out_dir / f"{variant_id}.recipe.json"
        variant_path.write_text(json.dumps(variant, indent=2) + "\n")
        variant_paths.append(variant_path)
        print(f"[sweep] wrote variant: {variant_path}")
    return variant_paths


def _dispatch(variant_paths: list[Path], dry_run: bool) -> None:
    for variant_path in variant_paths:
        cmd = ["act", "train", str(variant_path)]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[sweep] $ {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if proc.returncode != 0:
            sys.exit(
                f"[sweep] variant {variant_path.name} failed (exit {proc.returncode}); "
                "stopping sweep — fix and re-run."
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("recipe", type=Path, help="Path to a recipe with an `ablation` block")
    ap.add_argument("--ablation-name", help="Pick which ablation to sweep (required if multiple)")
    ap.add_argument(
        "--cartesian",
        action="store_true",
        help="Cross-product **all** ablation blocks (e.g. 4 sizes × 6 data points = 24 cells)",
    )
    ap.add_argument(
        "--values",
        help="Comma-separated subset of values to sweep (default: all values from the recipe)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Pass --dry-run to each act train call")
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Materialize variant recipes but do not dispatch act train",
    )
    args = ap.parse_args()

    base_recipe = _load(args.recipe)

    base_id = base_recipe["id"]
    out_dir = REPO_ROOT / "outputs" / base_id / "ablations"

    if args.cartesian:
        if args.values or args.ablation_name:
            raise SystemExit("--cartesian is incompatible with --ablation-name / --values")
        variant_paths = _materialize_cartesian(base_recipe, out_dir)
        if args.print_only:
            return
        _dispatch(variant_paths, dry_run=args.dry_run)
        print(f"[sweep] done. {len(variant_paths)} cartesian variant(s) processed.")
        return

    ablation = _select_ablation(base_recipe, args.ablation_name)

    selected: list[Any] | None = None
    if args.values:
        # Re-coerce comma-separated values to the type of the original list (best effort).
        original = ablation["values"]
        sample = original[0] if original else ""
        cast: Any = (lambda v: v.strip())  # noqa: E731
        if isinstance(sample, bool):
            cast = lambda v: v.strip().lower() in ("1", "true", "yes")  # noqa: E731
        elif isinstance(sample, int):
            cast = lambda v: int(v.strip())  # noqa: E731
        elif isinstance(sample, float):
            cast = lambda v: float(v.strip())  # noqa: E731
        selected = [cast(token) for token in args.values.split(",") if token.strip()]

    variant_paths = _materialize_variants(base_recipe, ablation, out_dir, selected)

    if args.print_only:
        print(f"[sweep] {len(variant_paths)} variants written under {out_dir}")
        return

    _dispatch(variant_paths, dry_run=args.dry_run)
    print(f"[sweep] done. {len(variant_paths)} variant(s) processed.")


if __name__ == "__main__":
    main()

"""Compose command — assemble method atoms into a training recipe.

Takes selected method atoms and composes them into a valid Recipe IR JSON,
validated against the schema.
"""

import argparse


def run_compose(args: argparse.Namespace) -> None:
    """Execute the compose pipeline.

    Pipeline:
        1. Load method atoms from registry
        2. Select atoms by name
        3. Compose into Recipe IR JSON
        4. Validate against schema
        5. Write output recipe file

    TODO: Implement composition logic.
    """
    raise NotImplementedError("Compose pipeline not yet implemented")

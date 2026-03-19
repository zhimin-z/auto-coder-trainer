"""Collect command — gather papers, projects, and methods into method cards.

Bridges ARIS Research Plane skills (research-lit, arxiv) to produce
structured method atoms in recipes/registry/.
"""

import argparse


def run_collect(args: argparse.Namespace) -> None:
    """Execute the collect pipeline.

    Pipeline:
        1. Search arXiv/Scholar for papers matching query
        2. Extract method descriptions from each paper
        3. Structure as method atom cards
        4. Save to recipes/registry/method_atoms.json

    TODO: Integrate with ARIS research-lit and arxiv skills.
    """
    raise NotImplementedError("Collect pipeline not yet implemented")

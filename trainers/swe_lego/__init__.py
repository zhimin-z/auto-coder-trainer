"""SWE-Lego training backend — LLaMA-Factory based SFT for coding agents."""

from trainers.swe_lego.launcher import (
    build_swe_lego_launcher_bundle,
    write_swe_lego_launcher_bundle,
)
from trainers.swe_lego.inference import write_inference_scripts
from trainers.swe_lego.model_registry import ModelProfile, resolve_model_profile
from trainers.swe_lego.verifier import (
    build_verifier_train_bundle,
    write_verifier_train_bundle,
)
from trainers.swe_lego.results_bridge import import_and_judge, import_results

__all__ = [
    "build_swe_lego_launcher_bundle",
    "write_swe_lego_launcher_bundle",
    "write_inference_scripts",
    "ModelProfile",
    "resolve_model_profile",
    "build_verifier_train_bundle",
    "write_verifier_train_bundle",
    "import_and_judge",
    "import_results",
]

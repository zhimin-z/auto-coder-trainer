"""Tests for SWE-Lego full pipeline script generation.

Verifies that write_swe_lego_launcher_bundle produces all 5 scripts
expected by the SLURM pipeline: run.sh, serve_and_infer.sh, eval.sh,
verifier_train.sh, tts.sh.
"""
import json
from pathlib import Path

from recipes.compiler import compile_recipe
from trainers.swe_lego import build_swe_lego_launcher_bundle, write_swe_lego_launcher_bundle


def _swe_lego_recipe() -> dict:
    return {
        "id": "recipe-pipeline-test",
        "name": "Pipeline script test",
        "model": {"base": "Qwen/Qwen3-8B", "size": "8B", "adapter": "full"},
        "dataset": {
            "sources": [
                {"name": "swe-lego-real", "path": "SWE-Lego/SWE-Lego-Real-Data"},
            ]
        },
        "trainer": {
            "type": "sft",
            "backend": "swe_lego",
            "params": {
                "lr": 1e-4,
                "epochs": 4,
                "batch_size": 1,
                "turn_mask": True,
                "template": "qwen3_nothink",
                "deepspeed": "z2_offload",
            },
        },
        "eval": {"benchmarks": ["swe-bench-verified"], "seeds": [42]},
        "budget": {"max_gpu_hours": 96, "gpu_type": "1xH200-141GB"},
    }


def test_bundle_generates_all_pipeline_scripts(tmp_path: Path) -> None:
    """All 5 pipeline scripts must exist after write_swe_lego_launcher_bundle."""
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    expected_scripts = [
        "run_script",
        "serve_and_infer_script",
        "eval_script",
        "verifier_train_script",
        "tts_script",
    ]
    for key in expected_scripts:
        assert key in paths, f"Missing key {key!r} in returned paths"
        assert Path(paths[key]).exists(), f"Script {key} does not exist at {paths[key]}"
        assert Path(paths[key]).stat().st_mode & 0o111, f"Script {key} is not executable"


def test_serve_and_infer_script_content(tmp_path: Path) -> None:
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["serve_and_infer_script"]).read_text()
    assert "vllm" in content
    assert "ACT_CHECKPOINT_PATH" in content or "ACT_SWE_LEGO_ROOT" in content
    assert "set -euo pipefail" in content


def test_eval_script_content(tmp_path: Path) -> None:
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["eval_script"]).read_text()
    assert "swebench" in content.lower() or "SWE-bench" in content
    assert "run_evaluation" in content


def test_verifier_train_script_content(tmp_path: Path) -> None:
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["verifier_train_script"]).read_text()
    assert "llamafactory-cli train" in content

    # Verifier YAML config should also exist
    bundle_dir = Path(paths["bundle_dir"])
    verifier_yaml = bundle_dir / "verifier" / "verifier_train.yaml"
    assert verifier_yaml.exists(), "Verifier YAML config missing"


def test_tts_script_content(tmp_path: Path) -> None:
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["tts_script"]).read_text()
    assert "TTS" in content or "tts" in content
    assert "verifier" in content.lower()
    assert "ACT_VERIFIER_MODEL_PATH" in content or "VERIFIER_MODEL" in content


def test_slurm_pipeline_scripts_at_expected_locations(tmp_path: Path) -> None:
    """Verify scripts are in bundle_dir root, matching SLURM submitter expectations."""
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    bundle_dir = Path(paths["bundle_dir"])
    expected_files = [
        "run.sh",
        "serve_and_infer.sh",
        "eval.sh",
        "verifier_train.sh",
        "tts.sh",
    ]
    for fname in expected_files:
        assert (bundle_dir / fname).exists(), f"{fname} not found in bundle_dir"

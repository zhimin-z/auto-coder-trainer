"""Tests for SWE-Lego full pipeline script generation.

Verifies that write_swe_lego_launcher_bundle produces all scripts
expected by the SLURM pipeline and the post-eval import bridge.
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
    """All pipeline scripts must exist after write_swe_lego_launcher_bundle."""
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
        "import_results_script",
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


def test_import_results_script_content(tmp_path: Path) -> None:
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    bundle["experiment_id"] = "exp-import-test"
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["import_results_script"]).read_text()
    assert "python -m cli.main train" in content
    assert "--import-results" in content
    assert "--report-format" in content
    assert "exp-import-test" in content


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
        "import_results.sh",
    ]
    for fname in expected_files:
        assert (bundle_dir / fname).exists(), f"{fname} not found in bundle_dir"


# ---------------------------------------------------------------------------
# Qwen3.5 model support tests
# ---------------------------------------------------------------------------


def _swe_lego_qwen3_5_recipe() -> dict:
    return {
        "id": "recipe-qwen3.5-9b-test",
        "name": "Qwen3.5-9B pipeline test",
        "model": {"base": "Qwen/Qwen3.5-9B", "size": "9B", "adapter": "full"},
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
                "template": "qwen3",
                "deepspeed": "z2_offload",
            },
        },
        "eval": {"benchmarks": ["swe-bench-verified"], "seeds": [42]},
        "budget": {"max_gpu_hours": 96, "gpu_type": "1xH200-141GB"},
    }


def test_qwen3_5_bundle_uses_correct_model_config(tmp_path: Path) -> None:
    """Qwen3.5 bundle should use qwen3.5 model profile, not qwen3 defaults."""
    from trainers.swe_lego.model_registry import resolve_model_profile

    recipe = _swe_lego_qwen3_5_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)

    # Profile is resolved at write time; verify via model_registry
    model_base = bundle["_model_config"]["base"]
    profile = resolve_model_profile(model_base)
    assert profile.template == "qwen3"
    assert profile.openhands_model_config == "llm.eval_qwen3_5"
    assert profile.max_model_len == 262144


def test_qwen3_5_serve_and_infer_has_correct_config(tmp_path: Path) -> None:
    """serve_and_infer.sh for Qwen3.5 should use llm.eval_qwen3_5 and 262144 context."""
    recipe = _swe_lego_qwen3_5_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["serve_and_infer_script"]).read_text()
    assert "MODEL_CONFIG=llm.eval_qwen3_5" in content
    assert "--max-model-len 262144" in content
    assert "--reasoning-parser" in content
    assert "llm.eval_qwen3_8b" not in content


def test_qwen3_5_eval_script_has_correct_model_name(tmp_path: Path) -> None:
    """eval.sh should reference Qwen3.5-9B in the output path, not Qwen3-8B."""
    recipe = _swe_lego_qwen3_5_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["eval_script"]).read_text()
    assert "Qwen3.5-9B" in content
    assert "Qwen3-8B" not in content


def test_qwen3_5_train_config_uses_correct_template(tmp_path: Path) -> None:
    """Training YAML should use the template from the recipe, not hardcoded qwen3_nothink."""
    recipe = _swe_lego_qwen3_5_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    train_yaml = Path(paths["train_config"]).read_text()
    assert "template: qwen3" in train_yaml
    assert "Qwen/Qwen3.5-9B" in train_yaml


def test_qwen3_5_all_pipeline_scripts_generated(tmp_path: Path) -> None:
    """Qwen3.5 bundle should produce all pipeline scripts."""
    recipe = _swe_lego_qwen3_5_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    bundle_dir = Path(paths["bundle_dir"])
    for fname in ["run.sh", "serve_and_infer.sh", "eval.sh", "verifier_train.sh", "tts.sh", "import_results.sh"]:
        assert (bundle_dir / fname).exists(), f"{fname} not found for Qwen3.5 bundle"


def test_qwen3_original_recipe_still_works(tmp_path: Path) -> None:
    """Ensure Qwen3-8B recipe still generates correct configs (no regression)."""
    recipe = _swe_lego_recipe()
    config = compile_recipe(recipe)
    bundle = build_swe_lego_launcher_bundle(config.__dict__, tmp_path)
    paths = write_swe_lego_launcher_bundle(bundle)

    content = Path(paths["serve_and_infer_script"]).read_text()
    assert "MODEL_CONFIG=llm.eval_qwen3_8b" in content
    assert "--max-model-len 163840" in content

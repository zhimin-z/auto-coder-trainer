"""Tests for the model registry."""

from trainers.swe_lego.model_registry import (
    ModelProfile,
    resolve_model_profile,
    _extract_family,
    get_known_families,
)


def test_extract_family_qwen3():
    assert _extract_family("Qwen/Qwen3-8B") == "qwen3"


def test_extract_family_qwen3_5():
    assert _extract_family("Qwen/Qwen3.5-9B") == "qwen3.5"


def test_extract_family_qwen3_5_moe():
    assert _extract_family("Qwen/Qwen3.5-35B-A3B") == "qwen3.5"


def test_extract_family_qwen2_5_coder():
    assert _extract_family("Qwen/Qwen2.5-Coder-32B-Instruct") == "qwen2.5"


def test_extract_family_unknown_defaults_to_qwen3():
    assert _extract_family("some-random-model") == "qwen3"


def test_resolve_qwen3_profile():
    profile = resolve_model_profile("Qwen/Qwen3-8B")
    assert profile.template == "qwen3_nothink"
    assert profile.openhands_model_config == "llm.eval_qwen3_8b"
    assert profile.max_model_len == 163840


def test_resolve_qwen3_5_profile():
    profile = resolve_model_profile("Qwen/Qwen3.5-9B")
    assert profile.template == "qwen3"
    assert profile.openhands_model_config == "llm.eval_qwen3_5"
    assert profile.max_model_len == 262144
    assert "--reasoning-parser" in profile.vllm_extra_flags
    assert profile.min_llamafactory_version == "0.9.5"
    assert profile.min_vllm_version == "0.17.0"


def test_resolve_qwen2_5_profile():
    profile = resolve_model_profile("Qwen/Qwen2.5-Coder-7B-Instruct")
    assert profile.template == "qwen2.5"
    assert profile.max_model_len == 131072


def test_resolve_with_overrides():
    profile = resolve_model_profile(
        "Qwen/Qwen3.5-9B",
        overrides={"max_model_len": 131072, "template": "custom_template"},
    )
    assert profile.max_model_len == 131072
    assert profile.template == "custom_template"
    # Non-overridden fields stay the same
    assert profile.openhands_model_config == "llm.eval_qwen3_5"


def test_get_known_families():
    families = get_known_families()
    assert "qwen3" in families
    assert "qwen3.5" in families
    assert "qwen2.5" in families

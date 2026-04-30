"""Build TinyZero-compatible launch bundles from compiled recipe configs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


def build_tinyzero_launcher_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compile a training config into a TinyZero/veRL launch bundle."""
    recipe_id = config.get("recipe_id", "unknown")
    trainer_type = config.get("trainer_type", "unknown")
    # Resolve to absolute so hydra-overrides paths (default_local_dir, ...)
    # remain valid when run.sh executes inside bundle_dir as cwd.
    bundle_dir = (Path(output_dir) / recipe_id / "tinyzero").resolve()

    model_cfg = config.get("model_config", {})
    data_cfg = config.get("data_config", {})
    training_params = config.get("training_params", {})
    budget = config.get("budget", {})

    dataset_binding = _build_dataset_binding(data_cfg.get("sources", []))

    if trainer_type == "sft":
        entrypoint = {
            "kind": "torchrun",
            "module": "verl.trainer.sft_trainer",
            "command_prefix": [
                "torchrun",
                "--standalone",
                "--nnodes=${ACT_NNODES}",
                "--nproc_per_node=${ACT_NPROC_PER_NODE}",
                "-m",
                "verl.trainer.sft_trainer",
            ],
        }
        env = {
            "ACT_NNODES": "1",
            "ACT_NPROC_PER_NODE": _default_gpu_count(budget),
            "ACT_TRAIN_FILE": dataset_binding["train_file"],
            "ACT_VAL_FILE": dataset_binding["val_file"],
        }
        cuda_devices = _cuda_visible_devices(budget)
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        overrides = _build_sft_overrides(recipe_id, bundle_dir, model_cfg, training_params)
        warnings = list(dataset_binding["warnings"])
        warnings.extend(_check_runtime_deps())
        if model_cfg.get("adapter", "full") != "full":
            warnings.append(
                "TinyZero SFT baselines are FSDP/full-finetune oriented. "
                "LoRA or QLoRA needs a custom launcher or the native TRL backend."
            )
    elif trainer_type in ("rl", "grpo"):
        entrypoint = {
            "kind": "python",
            "module": "verl.trainer.main_ppo",
            "command_prefix": [
                "python3",
                "-m",
                "verl.trainer.main_ppo",
            ],
        }
        env = {
            "ACT_NNODES": "1",
            "ACT_N_GPUS": _default_gpu_count(budget),
            "ACT_ROLLOUT_TP_SIZE": training_params.get("rollout_tp_size", "1"),
            "ACT_TRAIN_FILE": dataset_binding["train_file"],
            "ACT_VAL_FILE": dataset_binding["val_file"],
            "VLLM_ATTENTION_BACKEND": training_params.get("vllm_attention_backend", "XFORMERS"),
        }
        cuda_devices = _cuda_visible_devices(budget)
        if cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        # Expose any string-typed entry under `trainer.params.*` as an
        # `ACT_PARAM_<KEY>` env var. Numeric values are already wired via
        # hydra-overrides; strings are typically reward-style flags or
        # mode switches that custom reward functions read at run time.
        for key, value in training_params.items():
            if isinstance(value, str) and value:
                env[f"ACT_PARAM_{key.upper()}"] = value
        overrides = _build_rl_overrides(
            recipe_id=recipe_id,
            bundle_dir=bundle_dir,
            trainer_type=trainer_type,
            model_cfg=model_cfg,
            training_params=training_params,
            budget=budget,
        )
        warnings = list(dataset_binding["warnings"])
        warnings.extend(_check_runtime_deps())
    else:
        raise ValueError(
            f"TinyZero launcher only supports trainer types 'sft', 'rl', or 'grpo'; got {trainer_type!r}"
        )

    hydra_file = bundle_dir / "hydra-overrides.txt"
    env_file = bundle_dir / "env.sh"
    run_file = bundle_dir / "run.sh"
    command_preview = " ".join(entrypoint["command_prefix"] + ["<hydra overrides>"])

    return {
        "backend": "tinyzero",
        "recipe_id": recipe_id,
        "trainer_type": trainer_type,
        "artifact_dir": str(bundle_dir),
        "entrypoint": entrypoint,
        "command_preview": command_preview,
        "env": env,
        "overrides": overrides,
        "warnings": warnings,
        "requirements": [
            "Install a TinyZero/veRL-compatible environment before launch.",
            "Map Recipe dataset sources to parquet train/val files in env.sh.",
            "Run the generated run.sh script and append any extra Hydra overrides as needed.",
        ],
        "source_dataset_refs": [
            {
                "name": source.get("name", f"source-{idx}"),
                "path": source.get("path", ""),
                "mix_weight": source.get("mix_weight", 1.0),
            }
            for idx, source in enumerate(data_cfg.get("sources", []))
        ],
        "files": {
            "bundle_dir": str(bundle_dir),
            "hydra_overrides": str(hydra_file),
            "env": str(env_file),
            "run_script": str(run_file),
            "launcher_json": str(bundle_dir / "launcher.json"),
            "train_log": str(bundle_dir / "results" / "train.log"),
            "train_exit_code": str(bundle_dir / "results" / "train_exit_code.txt"),
        },
    }


def write_tinyzero_launcher_bundle(bundle: dict[str, Any]) -> dict[str, str]:
    """Persist a TinyZero launch bundle to disk."""
    bundle_dir = Path(bundle["artifact_dir"])
    bundle_dir.mkdir(parents=True, exist_ok=True)

    overrides_path = Path(bundle["files"]["hydra_overrides"])
    env_path = Path(bundle["files"]["env"])
    run_path = Path(bundle["files"]["run_script"])
    launcher_path = Path(bundle["files"]["launcher_json"])

    overrides_path.write_text(_render_overrides(bundle["overrides"]))
    env_path.write_text(_render_env(bundle))
    run_path.write_text(_render_run_script(bundle))
    launcher_path.write_text(json.dumps(bundle, indent=2))
    run_path.chmod(0o755)

    return {
        "bundle_dir": str(bundle_dir),
        "hydra_overrides": str(overrides_path),
        "env": str(env_path),
        "run_script": str(run_path),
        "launcher_json": str(launcher_path),
    }


def _default_gpu_count(budget: dict[str, Any]) -> str:
    # If recipe pins specific GPU indices, the count of those is authoritative
    # — otherwise gpu_type ("2xA100-80GB" -> "2") is the fallback.
    pinned = budget.get("cuda_visible_devices")
    if isinstance(pinned, list) and pinned:
        return str(len(pinned))
    if isinstance(pinned, str) and pinned.strip():
        return str(len([s for s in pinned.split(",") if s.strip()]))
    if isinstance(pinned, int):
        return "1"
    gpu_type = str(budget.get("gpu_type", "")).strip()
    if "x" in gpu_type.lower():
        maybe_count = gpu_type.lower().split("x", 1)[0].strip()
        if maybe_count.isdigit():
            return maybe_count
    return "1"


def _cuda_visible_devices(budget: dict[str, Any]) -> str | None:
    """Return CUDA_VISIBLE_DEVICES value if recipe pinned specific GPUs.

    Recipe schema: budget.cuda_visible_devices may be an int (single GPU index),
    a list of ints, or a comma-separated string. Returns None if unset, leaving
    the env unchanged so SLURM allocation or shell-level overrides apply.
    """
    raw = budget.get("cuda_visible_devices")
    if raw is None:
        return None
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, list):
        return ",".join(str(int(x)) for x in raw)
    return str(raw).strip()


def _check_runtime_deps() -> list[str]:
    """Warn early about runtime deps verl imports unconditionally."""
    warnings: list[str] = []
    try:
        import importlib.metadata as _meta

        try:
            _meta.version("flash_attn")
        except _meta.PackageNotFoundError:
            warnings.append(
                "flash-attn is not installed but verl.trainer.sft_trainer "
                "imports it at module load. Install with: "
                "pip install flash-attn --no-build-isolation"
            )
    except ImportError:
        pass
    return warnings


def _build_dataset_binding(sources: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    train_file = "<replace-with-train.parquet>"
    val_file = "<replace-with-val.parquet>"

    if not sources:
        warnings.append(
            "Recipe has no dataset sources. Set ACT_TRAIN_FILE and ACT_VAL_FILE before launch."
        )
        return {
            "train_file": train_file,
            "val_file": val_file,
            "warnings": warnings,
        }

    first_path = str(sources[0].get("path", "")).strip()
    if first_path.endswith(".parquet"):
        train_file = first_path
        val_file = first_path
        warnings.append(
            "Only one parquet source was detected, so ACT_VAL_FILE defaults to the same path. "
            "Override it before real experiments."
        )
    else:
        warnings.append(
            "Recipe dataset sources are registry or HF references, not TinyZero-ready parquet files. "
            "Map them to ACT_TRAIN_FILE and ACT_VAL_FILE in env.sh before launch."
        )

    return {
        "train_file": train_file,
        "val_file": val_file,
        "warnings": warnings,
    }


def _build_sft_overrides(
    recipe_id: str,
    bundle_dir: Path,
    model_cfg: dict[str, Any],
    training_params: dict[str, Any],
) -> list[str]:
    batch_size = training_params.get("batch_size", 4)
    fsdp_strategy = training_params.get("fsdp_strategy", "fsdp")
    wrap_min_params = training_params.get("fsdp_wrap_min_num_params", 10_000_000)
    return [
        "data.train_files=${oc.env:ACT_TRAIN_FILE}",
        "data.val_files=${oc.env:ACT_VAL_FILE}",
        f"data.messages_key={training_params.get('messages_key', 'messages')}",
        f"data.max_length={training_params.get('max_length', training_params.get('max_prompt_length', 4096))}",
        f"data.train_batch_size={batch_size}",
        f"data.micro_batch_size_per_gpu={training_params.get('micro_batch_size', batch_size)}",
        f"data.use_dynamic_bsz={_hydra_bool(training_params.get('use_dynamic_bsz', False))}",
        f"model.path={model_cfg.get('base', '')}",
        f"model.enable_gradient_checkpointing={_hydra_bool(training_params.get('gradient_checkpointing', False))}",
        f"engine.strategy={fsdp_strategy}",
        f"engine.wrap_policy.min_num_params={wrap_min_params}",
        f"trainer.default_local_dir={bundle_dir / 'checkpoints'}",
        "trainer.default_hdfs_dir=null",
        f"trainer.project_name={training_params.get('project_name', 'act-sft')}",
        f"trainer.experiment_name={recipe_id}",
        f"trainer.total_epochs={training_params.get('epochs', 1)}",
        "trainer.logger=['console']",
        f"optim.lr={training_params.get('lr', 1e-5)}",
        f"optim.lr_warmup_steps_ratio={training_params.get('warmup_ratio', 0.1)}",
    ]


def _build_rl_overrides(
    *,
    recipe_id: str,
    bundle_dir: Path,
    trainer_type: str,
    model_cfg: dict[str, Any],
    training_params: dict[str, Any],
    budget: dict[str, Any],
) -> list[str]:
    rollout_batch_size = int(training_params.get("rollout_batch_size", 256))
    ppo_micro_batch_size = int(training_params.get("ppo_micro_batch_size", training_params.get("micro_batch_size", 8)))
    is_grpo = trainer_type == "grpo"
    # verl validate_config does `n_gpus = n_gpus_per_node * nnodes` which fails
    # if these come in as strings (oc.env always returns str). Compute literal
    # ints from budget + recipe and bake them into the override so verl gets ints.
    n_gpus = int(_default_gpu_count(budget))
    n_nodes = int(training_params.get("nnodes", 1))
    rollout_tp = int(training_params.get("rollout_tp_size", 1))
    overrides = [
        "data.train_files=${oc.env:ACT_TRAIN_FILE}",
        "data.val_files=${oc.env:ACT_VAL_FILE}",
        f"data.train_batch_size={training_params.get('train_batch_size', rollout_batch_size)}",
        f"data.val_batch_size={training_params.get('val_batch_size', rollout_batch_size)}",
        f"data.max_prompt_length={training_params.get('max_prompt_length', 2048)}",
        f"data.max_response_length={training_params.get('max_response_length', 4096)}",
        f"actor_rollout_ref.model.path={model_cfg.get('base', '')}",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_hydra_bool(training_params.get('gradient_checkpointing', True))}",
        f"actor_rollout_ref.actor.optim.lr={training_params.get('lr', 1e-6)}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={training_params.get('ppo_mini_batch_size', max(1, rollout_batch_size // 4))}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={training_params.get('ppo_epochs', 1)}",
        f"actor_rollout_ref.actor.use_kl_loss={_hydra_bool(training_params.get('use_kl_loss', is_grpo))}",
        f"actor_rollout_ref.actor.kl_loss_coef={training_params.get('kl_coeff', 0.001)}",
        f"actor_rollout_ref.actor.entropy_coeff={training_params.get('entropy_coeff', 0.0)}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={ppo_micro_batch_size}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={training_params.get('gpu_memory_utilization', 0.4)}",
        f"actor_rollout_ref.rollout.n={training_params.get('group_size', 4) if is_grpo else 1}",
        f"actor_rollout_ref.rollout.temperature={training_params.get('temperature', 1.0)}",
        f"actor_rollout_ref.rollout.top_p={training_params.get('top_p', 1.0)}",
        f"algorithm.adv_estimator={'grpo' if is_grpo else 'gae'}",
        f"algorithm.kl_ctrl.kl_coef={training_params.get('kl_coeff', 0.001)}",
        f"trainer.default_local_dir={bundle_dir / 'checkpoints'}",
        "trainer.default_hdfs_dir=null",
        f"trainer.project_name={training_params.get('project_name', 'tinyzero')}",
        f"trainer.experiment_name={recipe_id}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.nnodes={n_nodes}",
        f"trainer.total_epochs={training_params.get('epochs', training_params.get('total_epochs', 1))}",
        "trainer.logger=['console']",
        f"trainer.save_freq={training_params.get('save_freq', -1)}",
        f"trainer.test_freq={training_params.get('test_freq', -1)}",
        f"trainer.total_training_steps={training_params.get('total_training_steps', -1)}",
        f"trainer.val_before_train={_hydra_bool(training_params.get('val_before_train', False))}",
    ]

    # Optional checkpoint resume — wrappers (e.g. two-phase transfer) set
    # ACT_RESUME_FROM_PATH at runtime, so we wire it through hydra without
    # pinning a path at bundle-generation time. `oc.env` defaults to `null`
    # when the env var is unset, which keeps the standard fresh-train path
    # working unchanged.
    overrides.append(
        'trainer.resume_from_path=${oc.env:ACT_RESUME_FROM_PATH,null}'
    )
    overrides.append(
        'trainer.resume_mode=${oc.env:ACT_RESUME_MODE,disable}'
    )

    # FP8 wiring (exp02). vLLM rollout supports FP8 via `dtype` /
    # `quantization`. Actor FP8 is FSDP/transformer-engine territory and
    # depends on the verl/torch build — we surface the override but flag a
    # warning if the recipe asks for it.
    if training_params.get("fp8_rollout"):
        overrides.append("actor_rollout_ref.rollout.dtype=fp8")
        overrides.append("actor_rollout_ref.rollout.quantization=fp8")
    if training_params.get("fp8_actor"):
        # Best-effort hydra path; actual support depends on the torch /
        # transformer-engine build the venv was assembled from.
        overrides.append("actor_rollout_ref.actor.fsdp_config.fp8=True")

    # LoRA wiring (exp10). verl reads lora keys from
    # `actor_rollout_ref.model.lora_rank` etc. — `lora_rank=0` means full
    # finetune. We honour `model.adapter` ("full" / "lora" / "qlora"):
    # full → write `lora_rank=0` (explicit baseline); lora → write the
    # recipe's rank + a 2× alpha by default. qlora needs additional
    # quant-config plumbing not landed here.
    adapter = (model_cfg.get("adapter") or "full").lower()
    if adapter == "lora":
        rank = int(training_params.get("lora_rank", 16))
        alpha = int(training_params.get("lora_alpha", rank * 2))
        overrides.append(f"actor_rollout_ref.model.lora_rank={rank}")
        overrides.append(f"actor_rollout_ref.model.lora_alpha={alpha}")
    else:
        # Make the baseline explicit so a recipe flipping back from lora
        # to full doesn't accidentally inherit a stale rank.
        overrides.append("actor_rollout_ref.model.lora_rank=0")

    if is_grpo:
        overrides.append("trainer.critic_warmup=0")
    else:
        overrides.extend(
            [
                f"critic.model.path={model_cfg.get('base', '')}",
                f"critic.optim.lr={training_params.get('critic_lr', 1e-5)}",
                f"critic.ppo_micro_batch_size_per_gpu={training_params.get('critic_micro_batch_size', ppo_micro_batch_size)}",
            ]
        )

    custom_reward = training_params.get("custom_reward_function")
    # Sentinel values that mean "use verl's built-in reward for this
    # data_source" — write no override, let `data.train_files`'s
    # `data_source` field route to verl/utils/reward_score/<name>.py.
    if isinstance(custom_reward, str) and custom_reward.strip().lower() in {"", "binary", "builtin", "default"}:
        custom_reward = None
    if custom_reward:
        reward_path, _, reward_name = str(custom_reward).partition(":")
        reward_name = reward_name or "compute_score"
        # Resolve relative paths against the repo root so verl can import the
        # script regardless of the cwd run.sh is invoked from.
        reward_path_obj = Path(reward_path)
        if not reward_path_obj.is_absolute():
            reward_path_obj = (Path.cwd() / reward_path_obj).resolve()
        overrides.append(f"custom_reward_function.path={reward_path_obj}")
        overrides.append(f"custom_reward_function.name={reward_name}")

    return overrides


def _render_overrides(overrides: list[str]) -> str:
    return "\n".join(overrides) + "\n"


def _render_env(bundle: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by auto-coder-trainer.",
        f"export ACT_RECIPE_ID={shlex.quote(bundle['recipe_id'])}",
    ]
    for source in bundle.get("source_dataset_refs", []):
        lines.append(
            "# Source ref: "
            f"{source.get('name', 'unknown')} -> {source.get('path', '')}"
        )
    for key, value in bundle["env"].items():
        lines.append(f'export {key}="${{{key}:-{value}}}"')
    lines.append("")
    return "\n".join(lines)


def _render_run_script(bundle: dict[str, Any]) -> str:
    entry = bundle["entrypoint"]
    command_prefix = " ".join(entry["command_prefix"])
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'source "$ROOT_DIR/env.sh"',
            'RESULTS_DIR="$ROOT_DIR/results"',
            'mkdir -p "$RESULTS_DIR"',
            "",
            'mapfile -t HYDRA_OVERRIDES < <(grep -v "^[[:space:]]*$" "$ROOT_DIR/hydra-overrides.txt")',
            "",
            'set +e',
            f"{command_prefix} \"${{HYDRA_OVERRIDES[@]}}\" \"$@\" 2>&1 | tee \"$RESULTS_DIR/train.log\"",
            'CMD_RC=${PIPESTATUS[0]}',
            'set -e',
            'echo "$CMD_RC" > "$RESULTS_DIR/train_exit_code.txt"',
            'if [[ "$CMD_RC" -ne 0 ]]; then',
            '  exit "$CMD_RC"',
            'fi',
            "",
        ]
    )


def _hydra_bool(value: Any) -> str:
    return "True" if bool(value) else "False"

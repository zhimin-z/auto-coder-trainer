#!/usr/bin/env python3
"""Materialise Countdown into the parquet format verl 0.7.x expects for RL/PPO.

The Countdown task (TinyZero original): given 3-4 numbers and a target, build
an arithmetic expression that combines all numbers (each used exactly once,
any of + - * /) to reach the target. The reward is `1` if the model's answer
evaluates to the target and uses every input number, else `0`.

Schema written here matches `make_gsm8k_data.py`:

    prompt        list[dict]  chat-format messages
    data_source   str         "Jiayi-Pan/Countdown-Tasks-3to4" — paired with reward_countdown.compute_score
    reward_model  dict        {"style": "rule", "ground_truth": {"target": int, "nums": list[int]}}
    extra_info    dict        split/index metadata

Note: ground_truth is a dict (not a string) because the Countdown reward
needs both `target` and `nums` to verify the expression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets

_DATA_SOURCE = "Jiayi-Pan/Countdown-Tasks-3to4"

_PROMPT_TEMPLATE = (
    "Using the numbers {nums}, build an arithmetic expression that equals {target}. "
    "You may use +, -, *, / and parentheses; each number must be used exactly once. "
    "Show your reasoning, then put the final expression on its own line in the format "
    "<answer>EXPRESSION</answer>."
)


def _to_verl_row(example: dict, idx: int, split: str) -> dict:
    nums = list(example["nums"])
    target = int(example["target"])
    return {
        "data_source": _DATA_SOURCE,
        "prompt": [
            {"role": "user", "content": _PROMPT_TEMPLATE.format(nums=nums, target=target)}
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": target, "nums": nums},
        },
        "extra_info": {"split": split, "index": idx, "target": target, "nums": nums},
    }


def main(out_dir: Path, max_train: int | None, max_val: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {_DATA_SOURCE} from HuggingFace ...")
    raw = datasets.load_dataset(_DATA_SOURCE, split="train")

    # Countdown only ships a `train` split; carve a held-out tail for validation.
    val_size = max_val if max_val is not None else 256
    train_size = max_train if max_train is not None else len(raw) - val_size
    train_size = min(train_size, len(raw) - val_size)

    train = raw.select(range(train_size))
    val = raw.select(range(len(raw) - val_size, len(raw)))

    train = train.map(lambda ex, i: _to_verl_row(ex, i, "train"), with_indices=True)
    val = val.map(lambda ex, i: _to_verl_row(ex, i, "test"), with_indices=True)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train.to_parquet(str(train_path))
    val.to_parquet(str(test_path))

    print(f"wrote {train_path} ({len(train)} rows)")
    print(f"wrote {test_path} ({len(val)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir", type=Path, help="Where to write train.parquet / test.parquet")
    ap.add_argument("--max-train", type=int, default=None, help="Subsample train (default: full minus val tail)")
    ap.add_argument("--max-val", type=int, default=None, help="Held-out val size (default: 256)")
    args = ap.parse_args()
    main(args.out_dir, args.max_train, args.max_val)

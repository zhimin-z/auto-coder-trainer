#!/usr/bin/env python3
"""Materialise GSM8K into the parquet format verl 0.7.x expects for RL/GRPO.

verl's reward manager (`verl.workers.reward_manager.naive`) reads the following
columns from each parquet row:

    prompt        list[dict]  chat-format messages, e.g. [{"role": "user", "content": "..."}]
    data_source   str         which built-in reward_score module to use ("openai/gsm8k" -> verl/utils/reward_score/gsm8k.py)
    reward_model  dict        {"style": "rule", "ground_truth": "<final answer>"}
    extra_info    dict        optional metadata (split, index, raw question/answer)

This script mirrors the official preprocessor at
verl 0.7.1 examples/data_preprocess/gsm8k.py, minus the HDFS branch — we only
need local files for single-node training.

Usage:
    python scripts/make_gsm8k_data.py <output_dir>
        Pulls openai/gsm8k from HuggingFace; writes train.parquet + test.parquet.
    python scripts/make_gsm8k_data.py <output_dir> --max-train 200 --max-val 50
        Subsamples to make a smoke-test-sized dataset.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import datasets

_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_DATA_SOURCE = "openai/gsm8k"


def _extract_answer(answer_field: str) -> str:
    """Pull the final number from GSM8K's `#### N` solution suffix."""
    match = re.search(r"#### (\-?[0-9\.\,]+)", answer_field)
    if match is None:
        raise ValueError(f"GSM8K row missing '#### <answer>' suffix: {answer_field!r}")
    return match.group(1).replace(",", "")


def _to_verl_row(example: dict, idx: int, split: str) -> dict:
    question_raw = example["question"]
    answer_raw = example["answer"]
    return {
        "data_source": _DATA_SOURCE,
        "prompt": [{"role": "user", "content": f"{question_raw} {_INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": _extract_answer(answer_raw)},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
    }


def main(out_dir: Path, max_train: int | None, max_val: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {_DATA_SOURCE} from HuggingFace ...")
    raw = datasets.load_dataset(_DATA_SOURCE, "main")

    train = raw["train"]
    test = raw["test"]
    if max_train is not None:
        train = train.select(range(min(max_train, len(train))))
    if max_val is not None:
        test = test.select(range(min(max_val, len(test))))

    train = train.map(lambda ex, i: _to_verl_row(ex, i, "train"), with_indices=True)
    test = test.map(lambda ex, i: _to_verl_row(ex, i, "test"), with_indices=True)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train.to_parquet(str(train_path))
    test.to_parquet(str(test_path))

    print(f"wrote {train_path} ({len(train)} rows)")
    print(f"wrote {test_path} ({len(test)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="Where to write train.parquet / test.parquet")
    ap.add_argument("--max-train", type=int, default=None, help="Subsample train split (default: full ~7.5k rows)")
    ap.add_argument("--max-val", type=int, default=None, help="Subsample test split (default: full ~1.3k rows)")
    args = ap.parse_args()
    main(args.out_dir, args.max_train, args.max_val)

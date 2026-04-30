#!/usr/bin/env python3
"""Generate a toy `messages`-format parquet dataset for the smoke test.

verl 0.7.x's MultiTurnSFTDataset expects each row to have a `messages` column
containing a list of {"role": ..., "content": ...} dicts. This script writes
train.parquet (100 rows) and val.parquet (20 rows) of trivial arithmetic Q&A
so the smoke recipe can verify the full training loop end-to-end.

Usage:
    python scripts/make_toy_data.py <output_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _row(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 96 train / 16 val: divisible by common multi-GPU configs (1/2/4/8 workers
    # with batch_size 4) so the smoke test scales to multi-GPU without verl's
    # "data size must be divisible by ..." assertion firing.
    train = [_row(f"What is {i}+{i}?", f"{i + i}") for i in range(1, 97)]
    val = [_row(f"What is {i}*2?", f"{i * 2}") for i in range(1, 17)]

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    pd.DataFrame(train).to_parquet(train_path, index=False)
    pd.DataFrame(val).to_parquet(val_path, index=False)

    print(f"wrote {train_path} ({len(train)} rows)")
    print(f"wrote {val_path} ({len(val)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))

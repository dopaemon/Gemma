#!/usr/bin/env python3
"""Build re_data/{train,valid}.jsonl from atul10's RE dataset for mlx_lm.lora."""

import json
from pathlib import Path

from datasets import load_dataset

N_ROWS = 2000
OUT_DIR = Path(__file__).parent / "re_data"
OUT_DIR.mkdir(exist_ok=True)

ds = load_dataset(
    "atul10/prompt_reverse_engineering_code_dataset_O0_x86_O0",
    split=f"train[:{N_ROWS}]",
)

rows = [
    {"prompt": r["prompt_2"], "completion": r["clean_raw_generation"]}
    for r in ds
    if r["prompt_2"] and r["clean_raw_generation"]
]

split = int(len(rows) * 0.9)
train_rows, valid_rows = rows[:split], rows[split:]

for name, subset in [("train", train_rows), ("valid", valid_rows)]:
    with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"train: {len(train_rows)} rows, valid: {len(valid_rows)} rows -> {OUT_DIR}")

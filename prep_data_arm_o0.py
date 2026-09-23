#!/usr/bin/env python3
"""Build re_data_arm_o0/{train,valid}.jsonl (function-name recovery task, O0 arm)."""

import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "re_data_arm_o0"
OUT_DIR.mkdir(exist_ok=True)

ds = load_dataset(
    "atul10/prompt_reverse_engineering_code_dataset_O0_arm_O0",
    split="train",
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

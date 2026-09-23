#!/usr/bin/env python3
"""Build re_data_arm_o2/{train,valid}.jsonl (function-name recovery task, O2 arm)."""

import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "re_data_arm_o2"
OUT_DIR.mkdir(exist_ok=True)

INSTRUCTION = (
    "You are an expert in software reverse engineering. The following is a "
    "decompiled C-like function with its symbols stripped (compiled at -O2, "
    "so expect inlining/optimization). Propose the most likely original "
    "function name.\n\n{code}"
)

ds = load_dataset(
    "atul10/reverse_engineering_code_dataset_O2_arm_O2",
    split="train",
)

rows = [
    {
        "prompt": INSTRUCTION.format(code=r["decompiled_code_stripped"]),
        "completion": r["original_function_name"],
    }
    for r in ds
    if r["decompiled_code_stripped"] and r["original_function_name"]
]

split = int(len(rows) * 0.9)
train_rows, valid_rows = rows[:split], rows[split:]

for name, subset in [("train", train_rows), ("valid", valid_rows)]:
    with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"train: {len(train_rows)} rows, valid: {len(valid_rows)} rows -> {OUT_DIR}")

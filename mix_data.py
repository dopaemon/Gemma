#!/usr/bin/env python3
"""Merge the prepared datasets into one shuffled corpus.

Training sources back-to-back lets whichever ran last dominate the adapter, and
these teach different output shapes - a JSON name mapping, C source, prose
analysis. Shuffled together the model learns to pick the shape from the prompt.

Two filters run first: samples longer than the trainer's window, and - for the
name-recovery source - samples whose answer is already visible in the prompt.
"""
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

# (dir, check_leakage). Leakage only means something for name recovery. The
# other three are translation or explanation tasks, where the answer is
# *supposed* to reuse text from the input.
SOURCES = [
    ("nd_gennm", True),
    ("nd_decompile", False),
    ("nd_reason", False),
    ("nd_think", False),
]
OUT = Path("re_data_all")

# Must match max_seq_length in lora_config.yaml. mlx_lm truncates longer samples
# to their *first* N tokens, which discards the answer.
MAX_SEQ = 2048
TEMPLATE_OVERHEAD = 16

tokenizer = AutoTokenizer.from_pretrained("./gemma-mlx-4bit")


def fits(batch):
    p = tokenizer([r["prompt"] for r in batch], add_special_tokens=False)["input_ids"]
    c = tokenizer([r["completion"] for r in batch], add_special_tokens=False)["input_ids"]
    return [len(a) + len(b) + TEMPLATE_OVERHEAD <= MAX_SEQ for a, b in zip(p, c)]


def leaks(row):
    """True if the recovered names are already sitting in the prompt.

    The answer is {placeholder: real_name}, and the placeholders are in the
    prompt by construction - it is the *values* that must not be, or the sample
    is teaching the model to copy rather than to infer.
    """
    try:
        names = json.loads(row["completion"]).values()
    except Exception:
        return False
    checkable = [n for n in names if isinstance(n, str) and len(n) >= 3]
    if not checkable:
        return False
    return all(re.search(r"\b" + re.escape(n) + r"\b", row["prompt"]) for n in checkable)


def merge(split):
    rows = []
    for src, check_leak in SOURCES:
        path = Path(src) / f"{split}.jsonl"
        if not path.exists():
            print(f"  {src}/{split}.jsonl: missing, skipped")
            continue
        parsed = [json.loads(line) for line in path.open()]

        unleaked = [r for r in parsed if not (check_leak and leaks(r))]
        kept = []
        for i in range(0, len(unleaked), 1000):
            chunk = unleaked[i : i + 1000]
            kept += [r for r, ok in zip(chunk, fits(chunk)) if ok]

        rows += [json.dumps(r, ensure_ascii=False) for r in kept]
        print(
            f"  {src}/{split}.jsonl: {len(kept)} kept"
            f" | {len(parsed) - len(unleaked)} leaked"
            f" | {len(unleaked) - len(kept)} over {MAX_SEQ} tokens"
        )
    random.shuffle(rows)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    print(f"  -> {OUT}/{split}.jsonl: {len(rows)}\n")


if __name__ == "__main__":
    random.seed(0)
    for split in ("train", "valid"):
        merge(split)

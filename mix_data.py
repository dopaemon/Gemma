#!/usr/bin/env python3
"""Interleave the four RE datasets into one shuffled corpus.

Training them back-to-back the way chain_train.sh does means whichever ran last
dominates the adapter. That matters here because the two families want different
answers: the -O0 sets expect ~200 words of reasoning, the -O2 sets expect a bare
identifier. Ending on arm -O2 (the largest set, and a -O2 one) would leave the
adapter terse on every prompt, including the ones asking for an explanation.
Shuffling them together teaches both formats at once, keyed off the prompt.
"""
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

SOURCES = ["re_data", "re_data_o2", "re_data_arm_o0", "re_data_arm_o2"]
OUT = Path("re_data_all")

# Must match --max-seq-length in train_re.sh. mlx_lm truncates anything longer to
# the *first* MAX_SEQ tokens, which throws away the completion - so an over-long
# sample teaches the model to stop partway through the decompiled C. Dropping it
# is better than training on it; at 2048 that costs ~7% of the corpus.
MAX_SEQ = 2048
TEMPLATE_OVERHEAD = 16  # chat-template turn markers mlx_lm adds around the pair

tokenizer = AutoTokenizer.from_pretrained("./gemma-mlx-4bit")


def fits(batch):
    lens = [
        len(p) + len(c)
        for p, c in zip(
            tokenizer([r["prompt"] for r in batch], add_special_tokens=False)["input_ids"],
            tokenizer([r["completion"] for r in batch], add_special_tokens=False)["input_ids"],
        )
    ]
    return [n + TEMPLATE_OVERHEAD <= MAX_SEQ for n in lens]


def merge(split):
    rows, dropped = [], 0
    for src in SOURCES:
        with (Path(src) / f"{split}.jsonl").open() as f:
            parsed = [json.loads(line) for line in f]
        kept = []
        for i in range(0, len(parsed), 1000):
            chunk = parsed[i : i + 1000]
            kept += [r for r, ok in zip(chunk, fits(chunk)) if ok]
        dropped += len(parsed) - len(kept)
        rows += [json.dumps(r, ensure_ascii=False) for r in kept]
        print(f"  {src}/{split}.jsonl: {len(kept)} kept, {len(parsed) - len(kept)} over {MAX_SEQ} tokens")
    random.shuffle(rows)
    (OUT / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    print(f"  -> {OUT}/{split}.jsonl: {len(rows)} ({dropped} dropped)\n")


if __name__ == "__main__":
    missing = [s for s in SOURCES if not (Path(s) / "train.jsonl").exists()]
    if missing:
        raise SystemExit(f"Run the prep scripts first, missing: {', '.join(missing)}")

    random.seed(0)  # so a re-run after adding a dataset isn't a whole new ordering
    OUT.mkdir(exist_ok=True)
    for split in ("train", "valid"):
        merge(split)

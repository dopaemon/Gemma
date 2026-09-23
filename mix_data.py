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
import re
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

# The -O2 sets claim stripped symbols but often kept the signature Ghidra
# recovered, so the answer sits in the prompt: 39% of re_data_o2 and 55% of
# re_data_arm_o2, against 1-3% for the -O0 sets. Those rows teach copying an
# identifier out of the input, which is exactly the shortcut that fails on a
# genuinely stripped binary where every name is FUN_0040dead.
NAME_IN_PROSE = re.compile(r"Function\s*Name\s*:\s*`?([A-Za-z_][\w:~<>.]*)")


def answer_name(completion):
    m = NAME_IN_PROSE.search(completion)
    return (m.group(1) if m else completion.strip()) or None


def leaks(row):
    name = answer_name(row["completion"])
    if not name:
        return False
    # Compare on the bare identifier: `~iostream` and `getline<wchar_t,...>` both
    # hinge on the part a decompiler would print in the signature.
    core = name.lstrip("~").split("<")[0].split("::")[-1]
    if len(core) < 3:  # 1-2 char names collide with variables by chance
        return False
    return re.search(r"\b" + re.escape(core) + r"\b", row["prompt"]) is not None


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
    rows = []
    for src in SOURCES:
        with (Path(src) / f"{split}.jsonl").open() as f:
            parsed = [json.loads(line) for line in f]

        unleaked = [r for r in parsed if not leaks(r)]
        kept = []
        for i in range(0, len(unleaked), 1000):
            chunk = unleaked[i : i + 1000]
            kept += [r for r, ok in zip(chunk, fits(chunk)) if ok]

        rows += [json.dumps(r, ensure_ascii=False) for r in kept]
        print(
            f"  {src}/{split}.jsonl: {len(kept)} kept"
            f" | {len(parsed) - len(unleaked)} leaked the answer"
            f" | {len(unleaked) - len(kept)} over {MAX_SEQ} tokens"
        )
    random.shuffle(rows)
    (OUT / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    print(f"  -> {OUT}/{split}.jsonl: {len(rows)}\n")


if __name__ == "__main__":
    missing = [s for s in SOURCES if not (Path(s) / "train.jsonl").exists()]
    if missing:
        raise SystemExit(f"Run the prep scripts first, missing: {', '.join(missing)}")

    random.seed(0)  # so a re-run after adding a dataset isn't a whole new ordering
    OUT.mkdir(exist_ok=True)
    for split in ("train", "valid"):
        merge(split)

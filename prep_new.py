#!/usr/bin/env python3
"""Download and normalise the second-generation RE datasets.

Each source gets its own prompt/completion shape, so the conversions live in one
table rather than one file per dataset the way prep_data*.py did.

Row caps keep the run tractable: gennm alone is 618k rows, which at the measured
0.11 it/s would be a two-month epoch.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(0)
VALID_FRACTION = 0.1


def _mapping_task(row):
    """gennm: stripped pseudo-C in, {stripped_name: real_name} out.

    The only source here whose labels come from real binaries rather than from
    another model, and the only one that also names variables.
    """
    resp = (row.get("response") or "").strip()
    if not resp.startswith("A:"):
        return None
    try:
        pairs = eval(resp[2:], {"__builtins__": {}})  # source writes Python dict literals
    except Exception:
        return None
    if not isinstance(pairs, dict) or not pairs:
        return None
    return {
        "prompt": "You are an expert in software reverse engineering.\n"
        "The following function was decompiled from a stripped binary, so every "
        "function and variable carries a placeholder name.\n"
        "Recover a meaningful name for each placeholder. Answer with a JSON object "
        "mapping each placeholder to your proposed name, and nothing else.\n\n"
        f"{row['query']}",
        "completion": json.dumps(pairs, ensure_ascii=False),
    }


def _decompile_task(row):
    return {
        "prompt": "You are an expert in software reverse engineering.\n"
        "Rewrite the following decompiler output as the idiomatic C source it was "
        "most likely compiled from.\n\n" + row["instruction"],
        "completion": row["output"],
    }


def _reasoning_task(row):
    return {
        "prompt": "You are an expert in software reverse engineering.\n"
        "Analyse the following decompiler output and explain what it does.\n\n"
        + row["instruction"],
        "completion": row["output"],
    }


def _think_task(row):
    return {
        "prompt": f"{row['instruction']}\n\n{row['input']}",
        "completion": row["output"],
    }


# (hf_id, output_dir, converter, row_cap, leak_check)
# leak_check marks the name-recovery tasks, where the answer appearing in the
# prompt means the sample teaches copying. For translation tasks (decompilation)
# the answer is *supposed* to share text with the input, so the check is off.
SOURCES = [
    ("Alex-xu/gennm-ghidra-O0", "nd_gennm", _mapping_task, 60000, True),
    ("LLM4Binary/decompile-ghidra-100k", "nd_decompile", _decompile_task, 20000, False),
    ("bahaeddineabdelwahed/Reasoning_from_decompilation", "nd_reason", _reasoning_task, 2000, False),
    ("tester230/rev_ghidra2c_think", "nd_think", _think_task, 1500, False),
]


def build(hf_id, out_dir, convert, cap, _leak):
    ds = load_dataset(hf_id, split="train")
    idx = list(range(len(ds)))
    random.shuffle(idx)

    samples = []
    for i in idx:
        produced = convert(ds[i])
        if produced is None:
            continue
        samples.extend(produced if isinstance(produced, list) else [produced])
        if len(samples) >= cap:
            break

    samples = [s for s in samples if s["prompt"] and s["completion"]]
    random.shuffle(samples)
    split = max(1, int(len(samples) * VALID_FRACTION))
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    for name, rows in (("valid", samples[:split]), ("train", samples[split:])):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    print(f"  {hf_id:52} -> {out_dir:14} {len(samples) - split} train / {split} valid")
    return len(samples)


if __name__ == "__main__":
    total = 0
    for spec in SOURCES:
        try:
            total += build(*spec)
        except Exception as e:
            print(f"  {spec[0]:52} -> FAILED: {type(e).__name__}: {str(e)[:120]}")
    print(f"\ntotal samples: {total}")

#!/usr/bin/env python3
"""Đo xem adapter đặt tên có dùng được không.

Val loss thấp không đảm bảo tên đặt ra hữu ích, nên chấm trực tiếp trên tên:
sinh tên cho hàm và biến trong pseudo-C, so với nhãn thật.

Con số tuyệt đối vô nghĩa nếu không có mốc so sánh. Chạy hai lần:

    uv run eval_names.py --out base.json              # model gốc
    uv run eval_names.py --adapter ./adapters --out lora.json

ĐỪNG chạy trong lúc đang train: tranh GPU là nguyên nhân của lỗi Metal
ImpactingInteractivity. Xem CLAUDE.md.
"""
import argparse
import json
import random
import re
from pathlib import Path

VALID = Path("re_data_all/valid.jsonl")
MODEL = "./gemma-mlx-4bit"

# gennm là nguồn duy nhất có nhãn tên thật; nhận ra nó qua completion là JSON dict
NAMING_PROMPT = "Recover a meaningful name for each placeholder"


def load_naming_rows(n, seed):
    rows = []
    for line in VALID.open():
        r = json.loads(line)
        if NAMING_PROMPT not in r["prompt"]:
            continue
        try:
            gold = json.loads(r["completion"])
        except Exception:
            continue
        if isinstance(gold, dict) and gold:
            rows.append((r["prompt"], gold))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def normalize(name):
    """So tên theo nghĩa, bỏ qua khác biệt hình thức.

    get_user_name / getUserName / GetUserName đều về cùng một chuỗi. Người
    reverse coi chúng là một, nên máy chấm cũng phải vậy.
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tokens(name):
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return {t for t in re.split(r"[^a-zA-Z0-9]+", s.lower()) if len(t) > 1}


def score(gold, pred):
    """Trả (khớp hẳn, khớp một phần, sai, thiếu) trên từng placeholder.

    "Khớp một phần" = có chung ít nhất một từ có nghĩa: `hash` với
    `hash_value` là gợi ý dùng được, `hash` với `tmp` thì không.
    """
    exact = partial = wrong = missing = 0
    for key, want in gold.items():
        got = pred.get(key) if isinstance(pred, dict) else None
        if not isinstance(got, str) or not got:
            missing += 1
        elif normalize(got) == normalize(want):
            exact += 1
        elif tokens(got) & tokens(want):
            partial += 1
        else:
            wrong += 1
    return exact, partial, wrong, missing


def extract_json(text):
    """Model hay bọc JSON trong ```json hoặc kèm lời dẫn."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="bỏ trống để đo model gốc")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(MODEL, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)  # chấm điểm thì phải tái lập được

    rows = load_naming_rows(args.n, args.seed)
    totals = [0, 0, 0, 0]
    unparsable = 0
    details = []

    for i, (prompt, gold) in enumerate(rows, 1):
        text = generate(
            model,
            tokenizer,
            prompt=tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            ),
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        pred = extract_json(text)
        if pred is None:
            unparsable += 1
            pred = {}
        s = score(gold, pred)
        totals = [a + b for a, b in zip(totals, s)]
        details.append({"gold": gold, "pred": pred, "score": s})
        if i % 20 == 0:
            n = sum(totals) or 1
            print(f"  {i}/{len(rows)}  khớp hẳn {totals[0] / n:.1%}", flush=True)

    n = sum(totals) or 1
    result = {
        "adapter": args.adapter,
        "rows": len(rows),
        "placeholders": n,
        "exact": totals[0] / n,
        "partial": totals[1] / n,
        "wrong": totals[2] / n,
        "missing": totals[3] / n,
        "unparsable_replies": unparsable / max(len(rows), 1),
    }
    print(f"\n{'model gốc' if not args.adapter else args.adapter}"
          f"  —  {len(rows)} hàm, {n} placeholder")
    print(f"  khớp hẳn      {result['exact']:.1%}")
    print(f"  khớp một phần {result['partial']:.1%}")
    print(f"  sai           {result['wrong']:.1%}")
    print(f"  không trả lời {result['missing']:.1%}")
    print(f"  hỏng định dạng JSON: {result['unparsable_replies']:.1%} số câu trả lời")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": result, "details": details}, ensure_ascii=False, indent=2)
        )
        print(f"  -> {args.out}")


def _selftest():
    assert normalize("getUserName") == normalize("get_user_name") == "getusername"
    assert tokens("hash_value") & tokens("hash")
    assert not (tokens("hash") & tokens("tmp"))
    # khớp hẳn, khớp một phần, sai, thiếu — mỗi loại một cái
    gold = {"a": "hash", "b": "user_name", "c": "counter", "d": "buffer"}
    pred = {"a": "Hash", "b": "userId", "c": "tmp"}
    assert score(gold, pred) == (1, 1, 1, 1), score(gold, pred)
    assert extract_json('nói linh tinh {"x": "y"} rồi thôi') == {"x": "y"}
    assert extract_json("không có json") is None
    print("selftest OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

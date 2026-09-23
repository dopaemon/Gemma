# heretic-re-server

Local OpenAI-compatible chat server for [Gemma4-E4B-8B-Heretic-Ultra](https://huggingface.co/dopaemon/Gemma4-E4B-8B-Heretic-Ultra)
(an abliterated/uncensored Gemma-4 via [Heretic](https://github.com/p-e-w/heretic)), running on Apple Silicon
via [mlx-lm](https://github.com/ml-explore/mlx-lm). Includes a LoRA fine-tuning pipeline for a reverse-engineering
task (recovering original function names from stripped/decompiled C).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Apple Silicon (mlx is Metal-only).

```bash
uv sync
```

This creates `.venv/` with `mlx-lm` and `datasets` installed.

### Get the model

The server expects a local MLX-quantized copy of the model at `./gemma-mlx-4bit`:

```bash
uv run huggingface-cli download dopaemon/Gemma4-E4B-8B-Heretic-Ultra-MLX-4Bit --local-dir ./gemma-mlx-4bit
```

Or convert it yourself from the full-precision weights:

```bash
uv run huggingface-cli download dopaemon/Gemma4-E4B-8B-Heretic-Ultra --local-dir ./Gemma4-E4B-8B-Heretic-Ultra
uv run mlx_lm.convert --hf-path ./Gemma4-E4B-8B-Heretic-Ultra --mlx-path ./gemma-mlx-4bit -q --q-bits 4
```

## Running the server

```bash
API_KEY=secret uv run api.py
```

Serves an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, authenticated via
`Authorization: Bearer secret`. Point any OpenAI-compatible client (Codex, 9router, etc.)
at it as a custom provider.

`api.py` is a thin wrapper around `mlx_lm.server` that adds:

- **Bearer auth** — rejects requests without the right `API_KEY`.
- **Single-model routing** — every request is forced onto the one model this server was
  started with, regardless of what `model` name the client sends (useful behind a router
  like 9router that may pass through an unrelated model id).
- **`--max-tokens 32768`** — mlx_lm defaults to 512 tokens per turn when a request omits
  `max_tokens`. Callers like Codex (via a translating router) often don't set it, so the
  server-side default is raised instead.
- **Sampling defaults from the model** — `--temp 1.0 --top-p 0.95 --top-k 64`, copied from
  the model's own `generation_config.json`. mlx_lm otherwise defaults to `temp 0.0`, i.e.
  pure greedy decoding, which makes Gemma repeat itself on long answers.

### Optional: attach the RE LoRA adapter

```bash
API_KEY=secret ADAPTER_PATH=./adapters uv run api.py
```

This specializes the model for naming decompiled functions. It does not degrade general
chat — measured at 3157 completion tokens with the adapter vs 3406 without, on the same
long-form prompt, both ending naturally.

Tool calling still works with the adapter attached, so Codex can use this as a normal
coding-agent backend either way.

## Training the RE adapter

Four datasets are supported, all function-name recovery from decompiled C, across
architecture and optimization level:

```bash
uv run prep_data.py          # x86 -O0, atul10/prompt_reverse_engineering_code_dataset_O0_x86_O0
uv run prep_data_o2.py       # x86 -O2, atul10/final_recreated_reverse_engineering_code_dataset_O2_x86_O2
uv run prep_data_arm_o0.py   # arm -O0, atul10/prompt_reverse_engineering_code_dataset_O0_arm_O0
uv run prep_data_arm_o2.py   # arm -O2, atul10/reverse_engineering_code_dataset_O2_arm_O2
```

Each downloads its full dataset from HuggingFace (no row cap — tens of thousands of rows,
several GB each, cached under `~/.cache/huggingface/datasets`) and writes `{train,valid}.jsonl`
to `re_data/`, `re_data_o2/`, `re_data_arm_o0/`, `re_data_arm_o2/` respectively.

Then merge them into one corpus and train:

```bash
uv run mix_data.py   # shuffles all four into ./re_data_all
./train_re.sh        # resumes from the latest checkpoint in ./adapters if present
```

`mix_data.py` exists because training the datasets back-to-back lets whichever ran last
dominate the adapter, and the two families want different answers — the `-O0` sets expect
~200 words of reasoning, the `-O2` sets a bare identifier. Shuffled together, the model
learns to pick the format from the prompt. It also drops the ~4% of samples that exceed
`--max-seq-length`, since mlx_lm truncates those to their *first* N tokens and would
otherwise be training on answers with the tail cut off.

`train_re.sh` wraps `mlx_lm.lora --model ./gemma-mlx-4bit --train`, checkpointing to
`./adapters/` every 50 iterations. Iteration count is derived from the data rather than
hardcoded — one pass over the corpus, ~90k iters at batch 2. At ~0.25 it/s that is several
days. You are not meant to sit through it in one go — see below.

### Why the training flags are what they are

Every one of these deviates from what the scripts originally used, and each was measured
rather than guessed:

- **`--mask-prompt`** — the single biggest one. Without it, loss is computed over the prompt
  too, and since these prompts are a long decompiled function while the `-O2` completions are
  a bare identifier, ~97% of the gradient went into learning to reproduce decompiled C rather
  than to name it. Measured directly: 3317 trained tokens per 2 iterations unmasked vs 70
  masked.
- **`--max-seq-length 2048`** (mlx_lm's default; the scripts had lowered it to 1024) —
  truncation keeps the *first* N tokens, so any sample over the limit loses its tail, i.e. the
  answer. At 1024 that silently discarded 13–17% of every dataset. 2048 leaves 4%, and
  `mix_data.py` drops those rather than feeding the trainer a headless answer.
- **`--batch-size 2`** — 15.0GB peak at seq 2048 on a 32GB machine; batch 4 needs 23GB, which
  is too close to the limit if the server is also running. Costs little, since throughput is
  ~250 tokens/sec regardless of batch size — the GPU is already saturated at batch 1.
- **`--learning-rate 1e-4`** — mlx_lm defaults to 1e-5, which is a full-fine-tune value rather
  than a LoRA one. Masking the prompt also cuts the per-step loss signal sharply, so this
  offsets that. This is the one knob here chosen by convention rather than measurement.

**Safe to interrupt anytime** — Ctrl+C, closing the laptop lid (sleep just pauses the
process), or a hard shutdown loses at most the last 50 iterations. Data prep only needs
network once (results are cached locally); training itself runs offline. Re-running the
same script auto-resumes from the latest checkpoint in `./adapters/`. A partial adapter is
usable — there is no need to reach the final iteration before trying it.

## Files

| File | Purpose |
|---|---|
| `api.py` | The server (see above). |
| `prep_data.py`, `prep_data_o2.py` | Build LoRA training data from the x86 RE datasets. |
| `prep_data_arm_o0.py`, `prep_data_arm_o2.py` | Build LoRA training data from the arm RE datasets. |
| `mix_data.py` | Shuffle the four datasets into one corpus, dropping over-length samples. |
| `train_re.sh` | Train/resume the adapter. |
| `adapters/` | LoRA checkpoints (gitignored — regenerate via the scripts above). |
| `gemma-mlx-4bit/` | MLX-quantized model (gitignored — download/convert per above). |

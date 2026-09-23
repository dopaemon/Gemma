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

Hyperparameters live in `lora_config.yaml`; `train_re.sh` only handles resuming and
pruning old checkpoints.

`mix_data.py` exists because training the datasets back-to-back lets whichever ran last
dominate the adapter, and the two families want different answers — the `-O0` sets expect
~200 words of reasoning, the `-O2` sets a bare identifier. Shuffled together, the model
learns to pick the format from the prompt. It also drops the ~4% of samples that exceed
`--max-seq-length`, since mlx_lm truncates those to their *first* N tokens and would
otherwise be training on answers with the tail cut off.

`train_re.sh` wraps `mlx_lm.lora`, checkpointing to `./adapters/` every 500 iterations and
keeping the last 3 — at rank 64 each one is ~600MB, and mlx_lm never deletes them. Iteration
count is one pass over the corpus, ~90k iters at batch 2. At ~0.15 it/s that is about a week.
You are not meant to sit through it in one go — see below.

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
- **`batch_size: 2`** — 18.4GB peak at seq 2048 and the capacity below, on a 32GB machine.
  Costs little, since the GPU is already saturated at batch 1.
- **`num_layers: 42` / `rank: 64`** — mlx_lm defaults to 16 layers at rank 8, which froze the
  bottom two thirds of the model and left 6.9M trainable parameters for a 180k-sample corpus
  teaching two different answer formats. Measured with `sweep_capacity.sh`:

  | rank | layers | trainable | peak GB | it/s |
  |---|---|---|---|---|
  | 8 | 16 | 6.9M | 15.9 | 0.257 |
  | 16 | 42 | 38.7M | 17.0 | 0.154 |
  | 32 | 42 | 77.3M | 17.4 | 0.153 |
  | 64 | 42 | 154.7M | 18.4 | 0.150 |

  Unfreezing all 42 layers is where the entire 1.7x slowdown is. Rank is nearly free after
  that — 8 → 64 costs 1.4GB and 0.004 it/s for 22x the parameters — so it is set high.
- **`scale: 2.0`** — mlx_lm multiplies the adapter output by `scale` directly; it is *not*
  `alpha/rank`. The 20.0 default is therefore tied to rank 8, and carrying it over to rank 64
  would scale every update up eightfold. 2.0 is alpha 128.
- **`learning_rate: 3e-5` with warmup and cosine decay** — this repo previously used 1e-4,
  chosen by convention. At the capacity above it diverges: val loss climbs 2.511 → 5.205 over
  150 iterations, train loss 2.8 → 7.7. Measured with `probe_lr.sh`:

  | lr | val loss, iter 1 → 150 |
  |---|---|
  | 1e-4 | 2.511 → 5.205 (diverged) |
  | 1e-5 | 2.511 → 0.892 |
  | 3e-5 | 2.511 → 0.867 |
- **`val_batches: 100`** — the default 25 is 50 samples out of 20k, too noisy to read a trend
  from.

**Safe to interrupt anytime** — Ctrl+C, closing the laptop lid (sleep just pauses the
process), or a hard shutdown loses at most the last 500 iterations (~55 min). Data prep only needs
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
| `lora_config.yaml` | All training hyperparameters, with the measurement behind each. |
| `train_re.sh` | Train/resume the adapter, pruning old checkpoints. |
| `sweep_capacity.sh` | Measure peak memory and throughput per rank/num_layers. |
| `probe_lr.sh` | Measure which learning rates converge at a given capacity. |
| `adapters/` | LoRA checkpoints (gitignored — regenerate via the scripts above). |
| `gemma-mlx-4bit/` | MLX-quantized model (gitignored — download/convert per above). |

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
- **Tool-schema stripping** — this server doesn't execute tools, so any `tools`/`tool_choice`
  in the request body is dropped before reaching the model, and the tokenizer's tool-calling
  mode is disabled. Otherwise Gemma pauses mid-answer to emit a `<tool_call>` (picked up from
  a caller's system prompt, like Codex's), and mlx_lm ends the turn right there — truncating
  what should've been a full answer to a couple hundred tokens.
- **Single-model routing** — every request is forced onto the one model this server was
  started with, regardless of what `model` name the client sends (useful behind a router
  like 9router that may pass through an unrelated model id).
- **`--max-tokens 32768`** — mlx_lm defaults to 512 tokens per turn when a request omits
  `max_tokens`. Callers like Codex (via a translating router) often don't set it, so the
  server-side default is raised instead.

### Optional: attach the RE LoRA adapter

```bash
API_KEY=secret ADAPTER_PATH=./adapters uv run api.py
```

Only do this when you actually want the reverse-engineering behavior. The adapter was
trained on completions that are always a single function name, so attaching it for every
request biases the model to stop after a couple hundred tokens even on unrelated
general-purpose chat/coding — the same symptom as the tool-calling issue above, but
unfixable from the server side since it's baked into the adapter weights. Run a second
`api.py` on a different port if you need both RE and general-purpose chat available at once.

## Training the RE adapter

Two datasets are supported, both function-name recovery from decompiled C, at different
optimization levels:

```bash
uv run prep_data.py       # -O0, atul10/prompt_reverse_engineering_code_dataset_O0_x86_O0
uv run prep_data_o2.py    # -O2, atul10/final_recreated_reverse_engineering_code_dataset_O2_x86_O2
```

Each writes `{train,valid}.jsonl` to `re_data/` / `re_data_o2/`.

Then train:

```bash
./train_re.sh       # single dataset, resumes from the latest checkpoint in ./adapters if present
./chain_train.sh     # queue of dataset/iteration jobs, run back-to-back; edit the QUEUE array to add more
```

Both wrap `mlx_lm.lora --model ./gemma-mlx-4bit --train`, checkpointing to `./adapters/`
every 50 iterations. Safe to interrupt (Ctrl+C) — the next run resumes from the latest
checkpoint. `chain_train.sh` also stops cleanly if a `STOP` file appears in this directory.

## Files

| File | Purpose |
|---|---|
| `api.py` | The server (see above). |
| `prep_data.py`, `prep_data_o2.py` | Build LoRA training data from the two RE datasets. |
| `train_re.sh` | Train/resume the adapter on one dataset. |
| `chain_train.sh` | Train across a queue of datasets back-to-back. |
| `adapters/` | LoRA checkpoints (gitignored — regenerate via the scripts above). |
| `gemma-mlx-4bit/` | MLX-quantized model (gitignored — download/convert per above). |

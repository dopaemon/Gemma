---
base_model:
- dopaemon/Gemma4-E4B-8B-Heretic-Ultra-MLX-4Bit
tags:
- lora
- qlora
- mlx
- reverse-engineering
---

# Gemma4 E4B 8B Heretic Ultra — RE QLoRA adapter

QLoRA adapter trained with [mlx-lm](https://github.com/ml-explore/mlx-lm) on top of [dopaemon/Gemma4-E4B-8B-Heretic-Ultra-MLX-4Bit](https://huggingface.co/dopaemon/Gemma4-E4B-8B-Heretic-Ultra-MLX-4Bit), for the task of recovering the original function name from stripped, decompiled C-like code.

## Training data

- ~2000 rows of decompiled function/name pairs at `-O0` (unoptimized x86), then
- ~2000 rows at `-O2` (optimized x86, more inlining), resumed from the `-O0` adapter.

## Training config

| Parameter | Value |
| :-------- | :---: |
| **fine_tune_type** | lora |
| **rank** | 8 |
| **scale** | 20.0 |
| **num_layers** | 16 |
| **iters** | 1800 + 1800 (chained) |
| **learning_rate** | 1e-05 |
| **max_seq_length** | 1024 |

## Usage

```bash
mlx_lm.generate \
  --model dopaemon/Gemma4-E4B-8B-Heretic-Ultra-MLX-4Bit \
  --adapter-path <this repo> \
  --prompt "..."
```

or with `mlx_lm.server --adapter-path <this repo>`.

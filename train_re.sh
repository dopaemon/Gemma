#!/bin/bash
# Resumable QLoRA training. Safe to Ctrl+C anytime and rerun later —
# it auto-detects the last saved checkpoint and continues from there.
set -euo pipefail

MODEL="./gemma-mlx-4bit"          # converted+quantized model (step done separately)
ADAPTER_DIR="./adapters"
DATA_DIR="./re_data"
SAVE_EVERY=50                     # checkpoint every 50 iters (short study sessions = frequent saves)
BATCH_SIZE=2                      # measured 15.0GB peak at seq 2048 on 32GB; batch 4 hits 23GB
MAX_SEQ=2048                      # mlx_lm's own default. At 1024, 13-17% of samples were over
                                  # the limit, and truncation cuts the tail - which is the answer.
LEARNING_RATE=1e-4                # mlx_lm defaults to 1e-5, low for LoRA; masking the prompt (below)
                                  # also shrinks the per-step loss signal, so this compensates.

# One pass over the data. Hardcoding this used to be fine when the prep scripts
# capped at 2000 rows, but they now emit the full dataset (50k+), where a fixed
# 1800 would only ever touch ~3% of it.
ITERS=$(( $(wc -l < "$DATA_DIR/train.jsonl") / BATCH_SIZE ))

mkdir -p "$ADAPTER_DIR"

RESUME_FLAG=()
LAST_CKPT=$(ls "$ADAPTER_DIR"/*_adapters.safetensors 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from checkpoint: $LAST_CKPT"
    RESUME_FLAG=(--resume-adapter-file "$LAST_CKPT")
else
    echo "No checkpoint found, starting fresh."
fi

.venv/bin/mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --iters "$ITERS" \
    --save-every "$SAVE_EVERY" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-length "$MAX_SEQ" \
    --learning-rate "$LEARNING_RATE" \
    --grad-checkpoint \
    --mask-prompt \
    "${RESUME_FLAG[@]}"

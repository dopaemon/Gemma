#!/bin/bash
# Resumable QLoRA training. Safe to Ctrl+C anytime and rerun later —
# it auto-detects the last saved checkpoint and continues from there.
set -euo pipefail

MODEL="./gemma-mlx-4bit"          # converted+quantized model (step done separately)
ADAPTER_DIR="./adapters"
DATA_DIR="./re_data"
SAVE_EVERY=50                     # checkpoint every 50 iters (short study sessions = frequent saves)
ITERS=1800

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
    --batch-size 1 \
    --max-seq-length 1024 \
    --grad-checkpoint \
    "${RESUME_FLAG[@]}"

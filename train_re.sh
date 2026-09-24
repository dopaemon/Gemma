#!/bin/bash
# Resumable QLoRA training. Safe to Ctrl+C anytime and rerun later —
# it auto-detects the last saved checkpoint and continues from there.
# All hyperparameters live in lora_config.yaml.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="./lora_config.yaml"
ADAPTER_DIR="./adapters"
KEEP_CHECKPOINTS=3   # rank 64 makes these ~600MB each; 1800 of them would fill the disk

mkdir -p "$ADAPTER_DIR"

RESUME_FLAG=()
# Newest by mtime, NOT by iteration number: mlx_lm restarts its step counter at 1
# on every resume, so after one resume 0000500 holds more training than 0002000.
LAST_CKPT=$(ls -t "$ADAPTER_DIR"/*_adapters.safetensors 2>/dev/null | head -1 || true)
if [ -n "$LAST_CKPT" ]; then
    echo "Resuming from checkpoint: $LAST_CKPT"
    RESUME_FLAG=(--resume-adapter-file "$LAST_CKPT")
else
    echo "No checkpoint found, starting fresh."
fi

# mlx_lm never deletes old checkpoints, so do it alongside the run rather than
# discovering a full disk on day four.
prune_checkpoints() {
    while sleep 300; do
        ls -t "$ADAPTER_DIR"/*_adapters.safetensors 2>/dev/null \
            | tail -n +$((KEEP_CHECKPOINTS + 1)) \
            | while read -r old; do rm -f "$old"; done
    done
}
prune_checkpoints &
trap 'kill %1 2>/dev/null || true' EXIT

.venv/bin/mlx_lm.lora \
    --config "$CONFIG" \
    ${RESUME_FLAG[@]+"${RESUME_FLAG[@]}"}   # bash 3.2 calls an empty array unbound under set -u

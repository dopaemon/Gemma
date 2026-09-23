#!/bin/bash
# Runs one training job after another, resuming the same adapter each time.
# Safe to stop anytime (Ctrl+C or `touch STOP`) — checkpoints persist every 50 iters.
set -uo pipefail
cd "$(dirname "$0")"

ADAPTER_DIR="./adapters"
LOG_DIR="./chain_logs"
mkdir -p "$LOG_DIR"

# Queue: "prep_script:data_dir:iters:label"
QUEUE=(
    "./prep_data_o2.py:./re_data_o2:1800:RE O2 x86 (release-like optimization)"
    "./prep_data_arm_o0.py:./re_data_arm_o0:1800:RE O0 arm"
    "./prep_data_arm_o2.py:./re_data_arm_o2:1800:RE O2 arm (release-like optimization)"
)

run_job() {
    local prep_script="$1" data_dir="$2" iters="$3" label="$4"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') Starting: $label ($data_dir, $iters iters) ==="

    if [ ! -f "$data_dir/train.jsonl" ]; then
        echo "--- Preparing data via $prep_script ---"
        .venv/bin/python3 "$prep_script" >> "$LOG_DIR/$(basename "$data_dir")_prep.log" 2>&1 || {
            echo "!!! Data prep failed for $label, skipping. See $LOG_DIR/$(basename "$data_dir")_prep.log"
            return 1
        }
    fi

    local resume=()
    local last_ckpt
    last_ckpt=$(ls "$ADAPTER_DIR"/*_adapters.safetensors 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$last_ckpt" ]; then
        resume=(--resume-adapter-file "$last_ckpt")
    fi

    .venv/bin/mlx_lm.lora \
        --model ./gemma-mlx-4bit \
        --train \
        --data "$data_dir" \
        --adapter-path "$ADAPTER_DIR" \
        --iters "$iters" \
        --save-every 50 \
        --batch-size 1 \
        --max-seq-length 1024 \
        --grad-checkpoint \
        "${resume[@]}" \
        >> "$LOG_DIR/$(basename "$data_dir").log" 2>&1

    echo "=== $(date '+%Y-%m-%d %H:%M:%S') Finished: $label (exit $?) ==="
}

# Wait for the currently running O0 job (train_re.sh) to finish first —
# two mlx_lm.lora processes competing for the GPU would OOM/corrupt checkpoints.
while pgrep -f "train_re.sh" > /dev/null; do
    sleep 15
done
echo "=== $(date '+%Y-%m-%d %H:%M:%S') O0 job finished, starting chain ==="

for entry in "${QUEUE[@]}"; do
    [ -f STOP ] && { echo "STOP file found, halting chain."; break; }
    IFS=':' read -r prep_script data_dir iters label <<< "$entry"
    run_job "$prep_script" "$data_dir" "$iters" "$label"
done

echo "=== Chain complete: $(date '+%Y-%m-%d %H:%M:%S') ==="

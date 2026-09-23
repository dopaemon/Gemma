#!/bin/bash
# Measures peak memory and throughput for LoRA capacity settings, so rank and
# num_layers get picked from data rather than from mlx_lm's defaults.
# Writes to a scratch adapter dir - never touches ./adapters.
set -uo pipefail
cd "$(dirname "$0")"

SCRATCH="${TMPDIR:-/tmp}/lora_sweep"
ITERS=30

# "rank:num_layers". 42 is every text layer in this model; 16 is mlx_lm's default.
COMBOS=("8:16" "16:42" "32:42" "64:42")

printf '%-10s %-8s %-12s %-10s %s\n' RANK LAYERS TRAINABLE PEAK_GB IT_PER_SEC

for combo in "${COMBOS[@]}"; do
    IFS=':' read -r rank layers <<< "$combo"
    rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"

    cat > "$SCRATCH/cfg.yaml" <<EOF
model: ./gemma-mlx-4bit
train: true
data: ./re_data_all
adapter_path: $SCRATCH
fine_tune_type: lora
num_layers: $layers
lora_parameters:
  rank: $rank
  scale: 20.0
  dropout: 0.0
iters: $ITERS
batch_size: 2
max_seq_length: 2048
grad_checkpoint: true
mask_prompt: true
save_every: 1000
steps_per_eval: 1000
steps_per_report: 10
EOF

    log="$SCRATCH/run.log"
    .venv/bin/mlx_lm.lora -c "$SCRATCH/cfg.yaml" > "$log" 2>&1

    trainable=$(grep -o 'Trainable parameters: [0-9.]*%[^)]*)' "$log" | tail -1 | grep -o '[0-9.]*M/' | tr -d 'M/')
    # Skip iter 10: it carries one-time graph-build cost and understates steady state.
    peak=$(grep -o 'Peak mem [0-9.]*' "$log" | tail -1 | grep -o '[0-9.]*')
    itsec=$(grep -o 'It/sec [0-9.]*' "$log" | tail -2 | grep -o '[0-9.]*' | \
            awk '{s+=$1; n++} END {if (n) printf "%.3f", s/n}')

    if [ -z "${peak:-}" ]; then
        printf '%-10s %-8s %-12s %-10s %s\n' "$rank" "$layers" "-" "OOM/FAIL" "see $log"
        rtk proxy tail -3 "$log"
    else
        printf '%-10s %-8s %-12s %-10s %s\n' "$rank" "$layers" "${trainable}M" "$peak" "$itsec"
    fi
done

rm -rf "$SCRATCH"

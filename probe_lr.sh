#!/bin/bash
# mlx_lm's scale is a direct multiplier, not alpha/rank, so its 20.0 default is
# tied to rank 8. Raising rank without lowering scale multiplies the effective
# update size. This runs each candidate long enough to see the loss trend.
set -uo pipefail
cd "$(dirname "$0")"

SCRATCH="${TMPDIR:-/tmp}/lora_probe"
ITERS=150
RANK=64
LAYERS=42

# "scale:lr:label". 1e-4 was measured diverging at this capacity (val 2.51 -> 5.21),
# so the search moved down rather than up; 3e-4 and scale 20.0 were dropped as
# strictly hotter than a setting already known to blow up.
COMBOS=(
    "2.0:1e-5:mlx_lm's own default LR"
    "2.0:3e-5:between the default and the value that diverged"
)

for combo in "${COMBOS[@]}"; do
    IFS=':' read -r scale lr label <<< "$combo"
    rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"

    cat > "$SCRATCH/cfg.yaml" <<EOF
model: ./gemma-mlx-4bit
train: true
data: ./re_data_all
adapter_path: $SCRATCH
fine_tune_type: lora
num_layers: $LAYERS
lora_parameters:
  rank: $RANK
  scale: $scale
  dropout: 0.0
iters: $ITERS
batch_size: 2
learning_rate: $lr
max_seq_length: 2048
grad_checkpoint: true
mask_prompt: true
save_every: 1000
steps_per_eval: $ITERS
steps_per_report: 10
val_batches: 100
EOF

    echo "=== scale=$scale lr=$lr — $label"
    .venv/bin/mlx_lm.lora -c "$SCRATCH/cfg.yaml" > "$SCRATCH/run.log" 2>&1
    grep -oE 'Iter [0-9]+: (Train loss [0-9.]+|Val loss [0-9.]+)' "$SCRATCH/run.log" \
        | grep -E 'Iter (10|30|50|70|90|100):|Val'
    echo
done

rm -rf "$SCRATCH"

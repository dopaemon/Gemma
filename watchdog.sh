#!/bin/bash
# Restart training if it dies. train_re.sh already resumes from the newest
# checkpoint, so a restart costs at most the 500 iters since the last save.
#
# ponytail: caps at MAX_RESTARTS because a fault that kills training instantly
# (OOM, corrupt checkpoint) would otherwise spin here all night writing logs.
set -uo pipefail
cd "$(dirname "$0")"

MAX_RESTARTS=10
LOG=chain_logs/watchdog.log
restarts=0

while true; do
    sleep 60
    pgrep -f "mlx_lm.lora" >/dev/null && continue

    # A finished run also has no process. Without this the watchdog would
    # restart training the moment it succeeds, forever.
    if tail -c 4000 chain_logs/train_all.log | tr '\r' '\n' | grep -q "Saved final weights"; then
        echo "$(date '+%F %T') training finished cleanly" >> "$LOG"
        exit 0
    fi

    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "$(date '+%F %T') GIVING UP after $MAX_RESTARTS restarts" >> "$LOG"
        exit 1
    fi

    restarts=$((restarts + 1))
    ckpt=$(ls -t adapters/*_adapters.safetensors 2>/dev/null | head -1)
    echo "$(date '+%F %T') training gone; restart $restarts from ${ckpt:-scratch}" >> "$LOG"
    nohup ./train_re.sh >> chain_logs/train_all.log 2>&1 &
done

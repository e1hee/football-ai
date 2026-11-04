#!/bin/bash
# EXP-021: args copied from experiments/EXP-020-tackle/s0/config.yaml; changed: --exp, --shed-rate, --reengage-cooldown, --threads
# Made for the Mac (M1 Max, about 47 min for three seeds). On the 4-vCPU cloud VM use --threads 1 (about 81 min).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-3}
for s in 0 1 2; do
  uv run python -m train.selfplay --exp EXP-021-hold/s$s --phases 24 --phase-steps 2000000 --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $s --threads $THREADS \
    > experiments/EXP-021-hold/train_s$s.log 2>&1 &
done
wait

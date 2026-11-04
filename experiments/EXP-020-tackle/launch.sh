#!/bin/bash
# EXP-020: args copied from experiments/EXP-019-mix/s0/config.yaml; changed: --exp, --tackle-base, --engaged-tackle, --threads
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
for s in 0 1 2; do
  uv run python -m train.selfplay --exp EXP-020-tackle/s$s --phases 24 --phase-steps 2000000 --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $s --threads 1 \
    > experiments/EXP-020-tackle/train_s$s.log 2>&1 &
done
wait

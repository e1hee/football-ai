#!/bin/bash
# EXP-022: args copied from experiments/EXP-021-hold/launch.sh; changed: --exp, --phases (24 -> 72 = 36 generations)
# Made for the Mac (M1 Max, about 2 h 20 min for three seeds).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-3}
for s in 0 1 2; do
  uv run python -m train.selfplay --exp EXP-022-long/s$s --phases 72 --phase-steps 2000000 --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $s --threads $THREADS \
    > experiments/EXP-022-long/train_s$s.log 2>&1 &
done
wait

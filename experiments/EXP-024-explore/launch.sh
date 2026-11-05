#!/bin/bash
# EXP-024: args copied from experiments/EXP-022-long/launch.sh; changed: --exp, --phases (72 -> 48 = 24 generations),
# --threads, and per arm A: --min-std 0.8 / B: --min-std 0.8 --mirror-share 0.5
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-2}
O=experiments/EXP-024-explore
mkdir -p $O/A $O/B
run() { # arm seed extra-args
  uv run python -m train.selfplay --exp EXP-024-explore/$1/s$2 --phases 48 --phase-steps 2000000 --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $2 --threads $THREADS \
    "${@:3}" > $O/$1/train_s$2.log 2>&1 &
}
# ARMS="B" reruns one arm. Each run's exit code is checked: a plain `wait` reported success after B crashed.
for s in 0 1 2; do
  [[ " ${ARMS:-A B} " == *" A "* ]] && run A $s --min-std 0.8
  [[ " ${ARMS:-A B} " == *" B "* ]] && run B $s --min-std 0.8 --mirror-share 0.5
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

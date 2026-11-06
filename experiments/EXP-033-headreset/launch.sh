#!/bin/bash
# EXP-033: EXP-030 arm A (EXP-028 A rules, never replace, --push-bound 1.25, new-offense league 0.2, per-step noise) plus a new offense action layer and std 1 before generation 2, 2+N, ... (train/selfplay.py --reset-head-every). A: every generation, B: every 3rd (2, 5, 8, 11)
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-2}
O=experiments/EXP-033-headreset
mkdir -p $O/A $O/B
run() { # arm seed extra-args
  uv run python -m train.selfplay --exp EXP-033-headreset/$1/s$2 --phases ${PHASES:-24} --phase-steps ${PHASE_STEPS:-2000000} --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $2 --threads $THREADS \
    --min-std 0.8 --mirror-share 0.5 --blocker-shaping 0.03 --ol-run-cost 0.01 --lb-rush-cost 0.01 --replace-margin 1000 --push-bound 1.25 --new-offense-share 0.2 \
    "${@:3}" > $O/$1/train_s$2.log 2>&1 &
}
for s in 0 1 2; do
  run A $s --reset-head-every 1
  run B $s --reset-head-every 3
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

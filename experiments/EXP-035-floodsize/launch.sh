#!/bin/bash
# EXP-035: EXP-034 arm A (overload defense in 30% of the offense's plays) with a shorter flood: 1.0 / 0.5 / 0.25 s (--overload-commit). 2 generations: EXP-034 read the 2.5 s flood in generation 1
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-2}
O=experiments/EXP-035-floodsize
mkdir -p $O/c10 $O/c05 $O/c025
run() { # arm seed extra-args
  uv run python -m train.selfplay --exp EXP-035-floodsize/$1/s$2 --phases ${PHASES:-4} --phase-steps ${PHASE_STEPS:-2000000} --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $2 --threads $THREADS \
    --min-std 0.8 --mirror-share 0.5 --blocker-shaping 0.03 --ol-run-cost 0.01 --lb-rush-cost 0.01 --replace-margin 1000 --push-bound 1.25 --new-offense-share 0.2 \
    "${@:3}" > $O/$1/train_s$2.log 2>&1 &
}
for s in 0 1 2; do
  run c10 $s --overload-share 0.3 --overload-commit 1.0
  run c05 $s --overload-share 0.3 --overload-commit 0.5
  run c025 $s --overload-share 0.3 --overload-commit 0.25
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

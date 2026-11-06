#!/bin/bash
# EXP-036: EXP-034 arm A plus, in 30% of the defense's plays, the newest offense with a flow read: it cuts away from the defense when the defense has moved >= 0.5 yd (A) or 0.75 yd (B) by 1 s (--bait-share, --bait-min-lean). Can the defense learn to bait it? 6 generations
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-2}
O=experiments/EXP-036-bait
mkdir -p $O/A $O/B
run() { # arm seed extra-args
  uv run python -m train.selfplay --exp EXP-036-bait/$1/s$2 --phases ${PHASES:-12} --phase-steps ${PHASE_STEPS:-2000000} --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $2 --threads $THREADS \
    --min-std 0.8 --mirror-share 0.5 --blocker-shaping 0.03 --ol-run-cost 0.01 --lb-rush-cost 0.01 --replace-margin 1000 --push-bound 1.25 --new-offense-share 0.2 \
    "${@:3}" > $O/$1/train_s$2.log 2>&1 &
}
for s in 0 1 2; do
  run A $s --overload-share 0.3 --bait-share 0.3 --bait-min-lean 0.5
  run B $s --overload-share 0.3 --bait-share 0.3 --bait-min-lean 0.75
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

#!/bin/bash
# EXP-025: args copied from experiments/EXP-024-explore/launch.sh arm B; changed: --exp, --threads, --phases (48 -> 24 = 12 generations),
# added --blocker-shaping 0.03 --ol-run-cost 0.01 --lb-rush-cost 0.01
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-3}
O=experiments/EXP-025-roles
for s in 0 1 2; do
  uv run python -m train.selfplay --exp EXP-025-roles/s$s --phases ${PHASES:-24} --phase-steps ${PHASE_STEPS:-2000000} --first tackler \
    --scripted-share 0.3 --delay 2 --scripted-tackler angle --tackle-base 3.2 --engaged-tackle -2.4 \
    --shed-rate 0.15 --reengage-cooldown 0.3 \
    --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 --spread 8.0 --full-play-share 0.1 --yards-scale 0.1 \
    --sim box-mix --lane-width 5.0 --stumble 0.2 --square-ahead --timeout-ends --net 256 --seed $s --threads $THREADS \
    --min-std 0.8 --mirror-share 0.5 --blocker-shaping 0.03 --ol-run-cost 0.01 --lb-rush-cost 0.01 \
    > $O/train_s$s.log 2>&1 &
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

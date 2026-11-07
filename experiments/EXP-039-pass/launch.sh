#!/bin/bash
# EXP-039: pass step 1, QB (auto lead, decides when) + outside WR vs CB, self-play, 3 seeds, 12 generations
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
THREADS=${THREADS:-3}
O=experiments/EXP-039-pass
run() { # seed
  uv run python -m train.selfplay --exp EXP-039-pass/s$1 --phases ${PHASES:-24} --phase-steps ${PHASE_STEPS:-2000000} --first tackler \
    --scripted-share 0.3 --delay 2 --tackle-base 3.2 --overspeed 0.25 --sidestep 0.2 --facing --n-envs 512 \
    --full-play-share 0.1 --yards-scale 0.1 --sim pass --stumble 0.2 --square-ahead --timeout-ends --net 256 \
    --seed $1 --threads $THREADS --min-std 0.8 --push-bound 1.25 --brake 2 \
    > $O/train_s$1.log 2>&1 &
}
for s in 0 1 2; do run $s; done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

#!/bin/bash
# EXP-029: can the 24M-step EXP-028 A offense still move? Each offense trains against its own frozen
# generation-12 defense, 4M steps, saved every 0.5M. Same harness as the fresh offense in EXP-028's analysis
# (scripts/recovery_test.py, 512 envs, seed 0, no std floor). The variants are a ladder from "as is" to "new":
#   warm   old weights, old noise (std 0.8)
#   noise  old weights, noise back to std 1 as a new network starts
#   head   old hidden layers, new action layer, std 1
#   fresh  a new network (the reference; EXP-028 analysis ran it for 2M only)
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
X=experiments/EXP-028-older/A
O=experiments/EXP-029-unstick
STEPS=${STEPS:-4000000}
EVERY=${EVERY:-500000}
mkdir -p $O/curves $O/checkpoints
run() { # variant seed extra-args
  uv run python -m scripts.recovery_test --name $1_s$2 --learner runner --run $X/s$2 --opp-gen 12 \
    --steps $STEPS --every $EVERY --threads ${THREADS:-1} \
    --save $O/checkpoints/$1_s$2.zip --save-each --out $O/curves/$1_s$2.jsonl "${@:3}" > $O/curves/$1_s$2.log 2>&1 &
}
for s in 0 1 2; do
  run warm $s --init-gen 12
  run noise $s --init-gen 12 --reset-std
  run head $s --init-gen 12 --reset-std --reset-head
  run fresh $s
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

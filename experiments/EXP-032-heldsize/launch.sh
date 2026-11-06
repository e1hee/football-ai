#!/bin/bash
# EXP-032: how big can held (gSDE) noise be and still let a trained offense move? The EXP-030 A generation-12
# offense (always one side) keeps training against its own frozen generation-12 defense, 2M steps, saved every 0.5M,
# same harness as EXP-029 (scripts/recovery_test.py, 512 envs, seed 0, no std floor). Variants:
#   step     the per-step noise it trained with (the control)
#   held05   same weights, held noise std 0.5, new every 64 steps (--sde-freq -1)
#   held10   std 1.0
#   held15   std 1.5 (about where EXP-031 A's noise grew to)
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
X=experiments/EXP-030-bounded/A
O=experiments/EXP-032-heldsize
STEPS=${STEPS:-2000000}
EVERY=${EVERY:-500000}
mkdir -p $O/curves $O/checkpoints
run() { # variant seed extra-args
  uv run python -m scripts.recovery_test --name $1_s$2 --learner runner --run $X/s$2 --opp-gen 12 --init-gen 12 \
    --steps $STEPS --every $EVERY --threads ${THREADS:-1} --flip \
    --save $O/checkpoints/$1_s$2.zip --save-each --out $O/curves/$1_s$2.jsonl "${@:3}" > $O/curves/$1_s$2.log 2>&1 &
}
for s in 0 1 2; do
  run step $s
  run held05 $s --sde-freq -1 --sde-std 0.5
  run held10 $s --sde-freq -1 --sde-std 1.0
  run held15 $s --sde-freq -1 --sde-std 1.5
done
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

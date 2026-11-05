#!/bin/bash
# EXP-023 audit: can each side still learn against one fixed opponent from EXP-022?
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
O=experiments/EXP-023-audit
run() { n=$1; shift; uv run python -m scripts.recovery_test --name $n --out $O/$n.jsonl "$@" > $O/$n.log 2>&1 & }
S2=experiments/EXP-022-long/s2; S1=experiments/EXP-022-long/s1
run t_warm    --learner tackler --run $S2 --init-gen 36
run t_fresh   --learner tackler --run $S2
run t_shaped  --learner tackler --run $S2 --init-gen 36 --shaping 0.02
run r_warm    --learner runner  --run $S1 --init-gen 36
run r_fresh   --learner runner  --run $S1
wait
# Second batch, after the first results: keep the fresh policies to look at them, and
# split "no exploration" from "weights can't change" with the noise reset to std 1.
CK=$O/checkpoints; mkdir -p $CK
run t_fresh2  --learner tackler --run $S2 --steps 2000000 --save $CK/t_fresh2.zip
run r_fresh2  --learner runner  --run $S1 --steps 2000000 --save $CK/r_fresh2.zip
run r_std1    --learner runner  --run $S1 --init-gen 36 --reset-std --steps 8000000
run t_std1    --learner tackler --run $S2 --init-gen 36 --reset-std --steps 8000000
wait

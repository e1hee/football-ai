#!/bin/bash
# EXP-041: GAE lambda for the corner alone, 3M sim steps each
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
O=experiments/EXP-041-cblambda
run() { uv run python $O/cb_check.py "$@" > $O/$1.log 2>&1 & }
run lam99 --lam 0.99
run lam1 --lam 1.0
run rep3lam1 --repeat 3 --lam 1.0
run rep3detlam1 --repeat 3 --catch-sep 1.0 --lam 1.0
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

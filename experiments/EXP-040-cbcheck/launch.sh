#!/bin/bash
# EXP-040: five corner-alone variants in parallel, 3M sim steps each
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
O=experiments/EXP-040-cbcheck
run() { uv run python $O/cb_check.py "$@" > $O/$1.log 2>&1 & }
run base
run rep3 --repeat 3
run rep5 --repeat 5
run det --catch-sep 1.0
run rep3det --repeat 3 --catch-sep 1.0
rc=0
for pid in $(jobs -p); do wait $pid || rc=1; done
exit $rc

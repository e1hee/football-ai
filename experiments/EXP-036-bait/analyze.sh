#!/bin/bash
# Records for one arm at generation 6: ./analyze.sh A 0.5
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
R=experiments/EXP-036-bait
X=$R/$1
O=$R/results_$1_g06
mkdir -p $O/checkpoints
experiments/EXP-024-explore/exploit.sh $X 6 $O
uv run python -m scripts.selfplay_audit $X --gen 6 --json $O/audit.json > $O/audit.log 2>&1
uv run python -m scripts.exploit_summary $O > $O/summary.log 2>&1
uv run python -m scripts.flow_bait $X --gen 1 3 6 --min-lean $2 --json $O/flow_bait.json > $O/flow_bait.log 2>&1
uv run python -m scripts.overload_check $X --gen 6 --json $O/overload_check.json > $O/overload_check.log 2>&1

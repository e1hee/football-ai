#!/bin/bash
# Records for one arm at generation 12, in the order the notes ask. ./analyze.sh A
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
R=experiments/EXP-026-lane
X=$R/$1
O=$R/results_$1_g12
mkdir -p $O/checkpoints
experiments/EXP-024-explore/exploit.sh $X 12 $O
uv run python -m scripts.selfplay_audit $X --gen 12 --json $O/audit.json > $O/audit.log 2>&1
uv run python -m scripts.box_report $X --gen 12 --mix --json $O/box_report.json > $O/box_report.log 2>&1
uv run python -m scripts.exploit_summary $O > $O/summary.log 2>&1
experiments/EXP-024-explore/records.sh $X 12 $O
uv run python scripts/slot_value.py $X 12 > $O/slot_value.log 2>&1
uv run python -m scripts.role_cross $X experiments/EXP-025-roles --gen 12 --json $O/role_cross.json > $O/role_cross.log 2>&1

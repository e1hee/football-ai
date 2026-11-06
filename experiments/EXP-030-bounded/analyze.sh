#!/bin/bash
# Records for one arm at generation 12, in the order the notes ask. ./analyze.sh A
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
R=experiments/EXP-030-bounded
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
uv run python -m scripts.role_report $X --gen 6 12 --json $R/role_$1.json > $R/role_$1.log 2>&1
uv run python -m scripts.lb_chase $X --gen 12 --json $O/lb_chase.json > $O/lb_chase.log 2>&1
uv run python -m scripts.offense_mix $X $O --gen 12 --json $O/offense_mix.json > $O/offense_mix.log 2>&1
uv run python -m scripts.lane_check $X --gen 1 3 6 12 --json $O/lane_check.json > $O/lane_check.log 2>&1
uv run python -m scripts.read_value $X --gen 12 --json $O/read_value.json > $O/read_value.log 2>&1
uv run python -m scripts.push_size $X --gen 1 3 6 12 --json $O/push_size.json > $O/push_size.log 2>&1

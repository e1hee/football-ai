#!/bin/bash
# Records at a generation: ./analyze.sh 12
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
R=experiments/EXP-039-pass
G=${1:-12}
O=$R/results_g$(printf %02d $G)
mkdir -p $O
uv run python -m scripts.pass_report $R --gen $G --json $O/pass_report.json > $O/pass_report.log 2>&1

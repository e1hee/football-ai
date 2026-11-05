#!/bin/bash
# Records that don't decide the pass/fail: line play and tackles at generation G.
# ./records.sh <exp root holding s0 s1 s2> <gen> <output dir>
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
uv run python -m scripts.ol_reblock $1 --gen $2 --json $3/ol_reblock.json > $3/ol_reblock.log 2>&1 &
uv run python -m scripts.ol_jobs $1 --gen $2 --json $3/ol_jobs.json > $3/ol_jobs.log 2>&1 &
uv run python -m scripts.line_hold $1 --gen $2 --mix --json $3/line_hold.json > $3/line_hold.log 2>&1 &
uv run python -m scripts.tackle_stats $1 --gen $2 --json $3/tackle_stats.json > $3/tackle_stats.log 2>&1 &
wait

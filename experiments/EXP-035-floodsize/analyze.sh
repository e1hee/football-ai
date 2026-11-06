#!/bin/bash
# Records for one variant at generations 1 and 2: ./analyze.sh c05 0.5
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
X=experiments/EXP-035-floodsize/$1
uv run python -m scripts.overload_check $X --gen 1 2 --commit $2 --json $X/overload_check.json > $X/overload_check.log 2>&1
uv run python -m scripts.flood_cue $X --gen 1 2 --commit $2 --json $X/flood_cue.json > $X/flood_cue.log 2>&1

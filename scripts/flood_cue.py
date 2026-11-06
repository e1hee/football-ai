"""Which cue does an offense read against the overload defense: where the LBs stand at the snap,
or where the defense moves after it? (EXP-034, EXP-035)

The same offense against the overload defense flooding the LBs' side, and flooding the other side.
switch: share crossing left when the LBs are right minus when they are left. A runner who reads the
LBs keeps +1 in both; one who reads the flow goes to -1 when the flood is flipped.
cue: "flow" (flipped switch <= -0.5), "LB" (flipped switch >= 0.5), else "mixed".

    PYTHONPATH=. uv run python -m scripts.flood_cue experiments/EXP-034-overload/A --gen 12 --commit 2.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.overload_check import sides
from scripts.read_value import run
from sim.team_policies import overload_defense
from train.evaluate import ppo_runner
from train.run_config import exp_config


def cue(off, cfg, commit: float, n: int = 1024, seed: int = 11) -> dict:
    side = sides(cfg, n, seed)
    L, R = side < 0, side > 0
    out = {}
    for toward in ("lb", "away"):
        r = run(off, lambda s, o=None: overload_defense(s, commit=commit, toward=toward), cfg, n, seed)
        y = r["y"]
        got = ~np.isnan(y)
        out[toward] = {"yards": float(r["yards"].mean()),
                       "switch": float((y[got & R] < 0).mean() - (y[got & L] < 0).mean())}
    f = out["away"]["switch"]
    out["cue"] = "flow" if f <= -0.5 else "LB" if f >= 0.5 else "mixed"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--commit", type=float, default=2.5)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    rows = []
    for g in args.gen:
        for s in range(3):
            off = ppo_runner(args.exp / f"s{s}" / "checkpoints" / f"runner_g{g:02d}.zip")
            r = {"seed": s, "gen": g, "commit": args.commit, **cue(off, cfg, args.commit)}
            rows.append(r)
            print(f"s{s} g{g:02d} commit {args.commit}s | flood LB side: {r['lb']['yards']:.2f} yd, switch {r['lb']['switch']:+.2f} "
                  f"| flood away from LBs: {r['away']['yards']:.2f} yd, switch {r['away']['switch']:+.2f} | cue {r['cue']}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

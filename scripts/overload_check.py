"""Does the overload defense leave a side worth reading? And does an offense read it? (EXP-034)

`overload_defense` floods the side the LBs stand on at the snap for a while. Runners, each with
its team's own blockers, 1024 plays, same starts for every runner:
- dumb runners (`scripts.lane_check`) aimed at fixed points on the line, and a reader aimed at
  the same distance on the side away from the LBs (it reads the snap and nothing else);
- the AI offense as it is.
Per runner: yards when the LBs are left / right, and for the AI the share crossing on the side
away from the LBs (weak side).
read gain: reader yards minus the best fixed point's yards. switch: the AI's left-crossing share
when the LBs are right minus when they are left (1 = always goes away from them).

    PYTHONPATH=. uv run python -m scripts.overload_check experiments/EXP-033-headreset/B --gen 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.lane_check import dumb_runner
from scripts.read_value import at, run
from sim.team import TeamSim
from sim.team_policies import box_offense, overload_defense, strong_side
from train.evaluate import ppo_runner
from train.run_config import exp_config

POINTS = (-5.0, -3.2, 0.0, 3.2, 5.0)


def sides(cfg, n: int, seed: int) -> np.ndarray:
    sim = TeamSim(n, cfg, seed=seed)
    sim.reset()
    return strong_side(sim).copy()


def away(dist: float):
    return lambda sim: sim.cfg.field.width / 2 - strong_side(sim) * dist


def check(name: str, off, cfg, n: int = 1024, seed: int = 11, dfn=None, ai=True) -> dict:
    dfn = dfn or (lambda s, o=None: overload_defense(s))
    side = sides(cfg, n, seed)
    L, R = side < 0, side > 0
    out = {"name": name, "lb_right": float(R.mean()), "fixed": {}, "reader": {}}
    for y in POINTS:
        r = run(dumb_runner(off, at(y)), dfn, cfg, n, seed)["yards"]
        out["fixed"][y] = [float(r.mean()), float(r[L].mean()), float(r[R].mean())]
    for d in (3.2, 5.0):
        r = run(dumb_runner(off, away(d)), dfn, cfg, n, seed)["yards"]
        out["reader"][d] = [float(r.mean()), float(r[L].mean()), float(r[R].mean())]
    best = max(v[0] for v in out["fixed"].values())
    out["read_gain"] = max(v[0] for v in out["reader"].values()) - best
    if ai:
        r = run(off, dfn, cfg, n, seed)
        y, yd = r["y"], r["yards"]
        got = ~np.isnan(y)
        left_L = float((y[got & L] < 0).mean())
        left_R = float((y[got & R] < 0).mean())
        out["ai"] = {"yards": float(yd.mean()), "yards_lb_left": float(yd[L].mean()), "yards_lb_right": float(yd[R].mean()),
                     "left_when_lb_left": left_L, "left_when_lb_right": left_R, "switch": left_R - left_L}
    return out


def show(o: dict) -> None:
    print(f"\n== {o['name']} (LB right {o['lb_right']*100:.0f}%) read gain {o['read_gain']:+.2f} yd")
    for y, v in o["fixed"].items():
        print(f"   aim {y:+.1f}: {v[0]:5.2f} yd | LB left {v[1]:5.2f} | LB right {v[2]:5.2f}")
    for d, v in o["reader"].items():
        print(f"   away {d:.1f}: {v[0]:5.2f} yd | LB left {v[1]:5.2f} | LB right {v[2]:5.2f}")
    if "ai" in o:
        a = o["ai"]
        print(f"   AI: {a['yards']:.2f} yd (LB left {a['yards_lb_left']:.2f}, right {a['yards_lb_right']:.2f}) | "
              f"crosses left {a['left_when_lb_left']*100:.0f}% when LB left, {a['left_when_lb_right']*100:.0f}% when LB right "
              f"| switch {a['switch']:+.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--commit", type=float, default=2.5)
    p.add_argument("--spot", type=float, default=5.0)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    dfn = lambda s, o=None: overload_defense(s, commit=args.commit, spot=args.spot)
    print(f"overload commit {args.commit} s, spot {args.spot} yd")
    outs = [check("스크립트 블로킹", lambda s, o=None: box_offense(s), cfg, args.n, dfn=dfn, ai=False)]
    show(outs[-1])
    for g in args.gen:
        for s in range(3):
            off = ppo_runner(args.exp / f"s{s}" / "checkpoints" / f"runner_g{g:02d}.zip")
            outs.append(check(f"s{s} g{g}", off, cfg, args.n, dfn=dfn))
            show(outs[-1])
    if args.json:
        args.json.write_text(json.dumps(outs, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

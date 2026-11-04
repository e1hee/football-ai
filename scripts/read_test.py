"""Does the runner read the defense? Two tests; the scripted reading runner is
the control.

1. Defense stands still. Setups: LBs where they line up / both on the right or
   left edge just past the line / slid 3 yd either way (`--mix`: inside what
   training saw), and with `--mix` each fixed front too. Any change in his path
   is the runner's own doing. Reports the share crossing right of the middle,
   the median crossing spot (yards from the middle, + = right), tackled share.
2. `--mix`, live against his own defense: the share crossing right, split by
   where the LBs lined up and which way the DTs shade at the snap.

    uv run python -m scripts.read_test experiments/EXP-018-snap --gen 12
    uv run python -m scripts.read_test experiments/EXP-019-mix --gen 12 --mix
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TACKLED, FieldConfig, TackleConfig
from sim.team import CARRIER, FRONTS, TeamSim, box_config
from sim.team_policies import box_offense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config

SETUPS = {"원래 자리": None, "LB 둘 다 오른쪽 길목": 6.0, "LB 둘 다 왼쪽 길목": -6.0}
SLIDES = {"LB 오른쪽으로 3야드": 3.0, "LB 왼쪽으로 3야드": -3.0}


def read_test(offense, cfg, lb_y: float | None, n: int = 512, seed: int = 7, slide: float = 0.0) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    sim.reset()
    roles = np.array(sim.roles)
    lbs = sim.dfn[roles[sim.dfn] == "LB"]
    mid = cfg.field.width / 2
    if lb_y is not None:
        for k, lb in enumerate(lbs):
            sim.pos[:, lb, 0] = sim.los + 1.5 + k * 1.5
            sim.pos[:, lb, 1] = mid + lb_y + (k - 0.5) * 1.5
    sim.pos[:, lbs, 1] += slide
    sim._push_history(np.ones(n, bool))
    obs = sim.observe()
    still = np.zeros((n, sim.n_def * 3), np.float32)
    live = np.ones(n, bool)
    cross = np.full(n, np.nan)
    tackled = np.zeros(n, bool)
    while live.any():
        before = sim.pos[:, CARRIER, 0] < sim.los
        los = sim.los.copy()
        obs, _, done, info = sim.step(offense(sim, obs), still)
        p = info["r_pos"]
        hit = live & before & (p[:, 0] >= los)
        cross[hit] = p[hit, 1] - mid
        tackled[done & live] = (info["outcome"] == OUTCOME_TACKLED)[done & live]
        live &= ~done
    y = cross[~np.isnan(cross)]
    return {"right": float((y > 0).mean()) if y.size else float("nan"),
            "y": float(np.median(y)) if y.size else float("nan"), "tackled": float(tackled.mean())}


def live_split(offense, defense, cfg, n: int = 1024, seed: int = 7) -> dict:
    """Share crossing right by LB side and DT shade at the snap, against a live defense."""
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    roles = np.array(sim.roles)
    mid = cfg.field.width / 2
    lb_y = sim.pos[:, sim.dfn[roles[sim.dfn] == "LB"], 1].mean(1) - mid
    dt_y = sim.pos[:, sim.dfn[1:3], 1].mean(1) - mid
    live = np.ones(n, bool)
    cross = np.full(n, np.nan)
    while live.any():
        before = sim.pos[:, CARRIER, 0] < sim.los
        los = sim.los.copy()
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        p = info["r_pos"]
        hit = live & before & (p[:, 0] >= los)
        cross[hit] = p[hit, 1] - mid
        live &= ~done
    right = lambda m: float((cross[m & ~np.isnan(cross)] > 0).mean()) if (m & ~np.isnan(cross)).any() else float("nan")
    return {"LB 왼쪽": right(lb_y < -1.5), "LB 오른쪽": right(lb_y > 1.5),
            "DT 왼쪽으로": right(dt_y < -0.4), "DT 오른쪽으로": right(dt_y > 0.4)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--mix", action="store_true")
    args = p.parse_args()
    cfg = box_config(snap=True, mix=args.mix, stumble=0.2, carrier_square_ahead=True, timeout_ends=True,
                     field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    cfg = exp_config(args.exp, cfg)   # the runs' own rules (tackle odds changed in EXP-020)
    fixed = dataclasses.replace(cfg, formation=dataclasses.replace(cfg.formation, fronts=None, lb_shift=0.0))
    runners = {"스크립트 구멍 읽기": lambda s, o=None: box_offense(s)}
    defenses = {}
    for r in sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()):
        runners[f"AI {r.name} g{args.gen:02d}"] = ppo_runner(r / "checkpoints" / f"runner_g{args.gen:02d}.zip")
        defenses[f"AI {r.name} g{args.gen:02d}"] = ppo_tackler(r / "checkpoints" / f"tackler_g{args.gen:02d}.zip")
    for name, off in runners.items():
        print(name)
        rows = [(label, fixed, y, 0.0) for label, y in SETUPS.items()]
        if args.mix:
            rows += [(label, fixed, None, s) for label, s in SLIDES.items()]
            for front in FRONTS:
                one = dataclasses.replace(fixed, formation=dataclasses.replace(fixed.formation, dl_align=front))
                rows.append((f"정렬 {front}", one, None, 0.0))
        for label, c, y, slide in rows:
            s = read_test(off, c, y, n=args.n, slide=slide)
            print(f"  {label:26s} 오른쪽 {s['right']:5.0%}  넘는 지점 {s['y']:+5.1f}야드  태클 {s['tackled']:4.0%}", flush=True)
        if args.mix and name in defenses:
            s = live_split(off, defenses[name], cfg)
            print("  자기 수비 상대, 오른쪽 비율: " + "  ".join(f"{k} {v:.0%}" for k, v in s.items()), flush=True)


if __name__ == "__main__":
    main()

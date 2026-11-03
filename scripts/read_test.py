"""Does the runner read the defense? Move both LBs before the snap and see if
his path changes. The defense stands still, so any change is the runner's own
doing; the scripted reading runner is the control.

Setups: LBs where they line up / both on his usual path (right edge, just past
the line) / both on the left edge. Reports the share crossing right of the
middle, the median crossing spot (yards from the middle, + = right), and how
often he was tackled.

    uv run python -m scripts.read_test experiments/EXP-018-snap --gen 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TACKLED, FieldConfig, TackleConfig
from sim.team import CARRIER, TeamSim, box_config
from sim.team_policies import box_offense
from train.evaluate import ppo_runner

SETUPS = {"원래 자리": None, "LB 둘 다 오른쪽 길목": 6.0, "LB 둘 다 왼쪽 길목": -6.0}


def read_test(offense, cfg, lb_y: float | None, n: int = 512, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    sim.reset()
    roles = np.array(sim.roles)
    lbs = sim.dfn[roles[sim.dfn] == "LB"]
    mid = cfg.field.width / 2
    if lb_y is not None:
        for k, lb in enumerate(lbs):
            sim.pos[:, lb, 0] = sim.los + 1.5 + k * 1.5
            sim.pos[:, lb, 1] = mid + lb_y + (k - 0.5) * 1.5
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=512)
    args = p.parse_args()
    cfg = box_config(snap=True, stumble=0.2, carrier_square_ahead=True, timeout_ends=True,
                     field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    runners = {"스크립트 구멍 읽기": lambda s, o=None: box_offense(s)}
    for r in sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()):
        runners[f"AI {r.name} g{args.gen:02d}"] = ppo_runner(r / "checkpoints" / f"runner_g{args.gen:02d}.zip")
    for name, off in runners.items():
        print(name)
        for label, y in SETUPS.items():
            s = read_test(off, cfg, y, n=args.n)
            print(f"  {label:14s} 오른쪽 {s['right']:5.0%}  넘는 지점 {s['y']:+5.1f}야드  태클 {s['tackled']:4.0%}", flush=True)


if __name__ == "__main__":
    main()

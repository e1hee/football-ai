"""Which side does the runner take, generation by generation? For each run and
generation, offense vs its own defense: share crossing the line right of the
middle, median yards from the middle at the crossing, runner wins.

    uv run python -m scripts.side_by_gen experiments/EXP-018-snap --snap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TOUCHDOWN, FieldConfig, TackleConfig
from sim.team import CARRIER, TeamSim, box_config
from train.evaluate import ppo_runner, ppo_tackler


def crossings(offense, defense, cfg, n: int = 512, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    cross_y = np.full(n, np.nan)
    won = np.zeros(n, bool)
    mid = cfg.field.width / 2
    while live.any():
        before = sim.pos[:, CARRIER, 0] < sim.los
        los = sim.los.copy()
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        p = info["r_pos"]
        crossed = live & before & (p[:, 0] >= los)
        cross_y[crossed] = p[crossed, 1] - mid
        won[done & live] = (info["outcome"] == OUTCOME_TOUCHDOWN)[done & live]
        live &= ~done
    got = ~np.isnan(cross_y)
    y = cross_y[got]
    return {"crossed": float(got.mean()), "right": float((y > 0).mean()) if got.any() else float("nan"),
            "far_y": float(np.median(np.abs(y))) if got.any() else float("nan"), "win": float(won.mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--snap", action="store_true")
    p.add_argument("--mix", action="store_true")
    p.add_argument("--n", type=int, default=512)
    args = p.parse_args()
    cfg = box_config(snap=args.snap, mix=args.mix, stumble=0.2, carrier_square_ahead=True, timeout_ends=True,
                     field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir())
    for r in runs:
        print(r.name)
        for f in sorted((r / "checkpoints").glob("runner_g*.zip")):
            g = f.stem.split("_g")[1]
            t = r / "checkpoints" / f"tackler_g{g}.zip"
            if not t.exists():
                continue
            s = crossings(ppo_runner(f), ppo_tackler(t), cfg, n=args.n)
            print(f"  g{g}  오른쪽 {s['right']:5.0%}  중앙에서 {s['far_y']:4.1f}야드  승 {s['win']:4.0%}  라인 넘음 {s['crossed']:4.0%}", flush=True)


if __name__ == "__main__":
    main()

"""Describe *how* a runner plays, not just whether it scores.

For each generation: closest approach to the tackler, how far sideways the
runner drifts, runner and tackler speed at the closest moment, and whether
the tackler was braking (breaking down) then.

    uv run python -m scripts.behavior_stats experiments/EXP-002-ppo-runner-v1 --gens 0 1 2 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.core import FieldConfig, OneVOneSim, SimConfig
from sim.policies import angle_tackler, evasive_runner, straight_runner
from train.evaluate import ppo_runner


def stats(runner, cfg: SimConfig, n: int = 2048, seed: int = 12345) -> dict:
    sim = OneVOneSim(n, cfg, seed=seed)
    obs = sim.reset()
    y0 = sim.r_pos[:, 1].copy()
    live = np.ones(n, bool)
    min_d = np.full(n, np.inf)
    r_sp_at = np.zeros(n)
    t_sp_at = np.zeros(n)
    t_brk_at = np.zeros(n, bool)
    max_lat = np.zeros(n)
    while live.any():
        t_act = angle_tackler(sim)
        # Is the tackler pushing against its own motion (breaking down)?
        tv = sim.t_vel
        braking = np.sum(t_act * tv, axis=1) < -0.1 * np.linalg.norm(tv, axis=1)
        d = np.linalg.norm(sim.r_pos - sim.t_pos, axis=1)
        closer = live & (d < min_d)
        min_d[closer] = d[closer]
        r_sp_at[closer] = np.linalg.norm(sim.r_vel, axis=1)[closer]
        t_sp_at[closer] = np.linalg.norm(tv, axis=1)[closer]
        t_brk_at[closer] = braking[closer]
        max_lat = np.where(live, np.maximum(max_lat, np.abs(sim.r_pos[:, 1] - y0)), max_lat)
        obs, _, done, _ = sim.step(runner(sim, obs), t_act)
        live &= ~done
    return {
        "closest_yd": float(np.median(min_d)),
        "runner_speed_at_closest": float(np.median(r_sp_at)),
        "tackler_speed_at_closest": float(np.median(t_sp_at)),
        "tackler_braking_at_closest": float(t_brk_at.mean()),
        "max_sideways_yd": float(np.median(max_lat)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gens", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--spread", type=float, default=8.0)
    args = p.parse_args()
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=args.spread))
    rows = [("직진(스크립트)", lambda s, o: straight_runner(s)), ("회피(스크립트)", lambda s, o: evasive_runner(s))]
    rows += [(f"AI {g}세대", ppo_runner(args.exp / "checkpoints" / f"gen_{g:03d}.zip")) for g in args.gens]
    print(f"{'runner':>14} | closest yd | runner v | tackler v | tackler braking | max sideways yd")
    for name, pol in rows:
        r = stats(pol, cfg)
        print(f"{name:>14} | {r['closest_yd']:10.2f} | {r['runner_speed_at_closest']:8.2f} | "
              f"{r['tackler_speed_at_closest']:9.2f} | {r['tackler_braking_at_closest']:15.0%} | {r['max_sideways_yd']:15.2f}")


if __name__ == "__main__":
    main()

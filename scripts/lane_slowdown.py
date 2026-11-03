"""Where does the lane-drill runner lose speed before the line: while touching
someone (bumped by his blocker, the DL, a tackle try) or on his own (a brake)?

    uv run python -m scripts.lane_slowdown experiments/EXP-014-lane-lb --lb
"""
import argparse
from pathlib import Path

import numpy as np

from sim.core import FieldConfig, TackleConfig
from sim.team import CARRIER, TeamSim, lane_drill_config
from train.evaluate import ppo_runner, ppo_tackler


def slowdown(offense, defense, cfg, n=2048, seed=7):
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    lost_free = np.zeros(n)
    lost_touch = np.zeros(n)
    reached = np.zeros(n, bool)
    rad = sim.bodies.radius
    while live.any():
        o, d = offense(sim, obs), defense(sim, obs)
        v0 = np.linalg.norm(sim.vel[:, CARRIER], axis=1)
        before = sim.pos[:, CARRIER, 0] < sim.los
        obs, _, done, info = sim.step(o, d)
        v1 = np.linalg.norm(sim.vel[:, CARRIER], axis=1)
        pos = info["pos"]
        gap = np.linalg.norm(pos[:, None, CARRIER] - pos[:, 1:], axis=-1) - (rad[:, CARRIER, None] + rad[:, 1:])
        touching = (gap < 0.05).any(1) | info["wrapped"] | info["tackle_attempt"]
        ok = live & before & ~done
        drop = np.maximum(0.0, v0 - v1)
        lost_free += np.where(ok & ~touching, drop, 0.0)
        lost_touch += np.where(ok & touching, drop, 0.0)
        reached |= live & before & (pos[:, CARRIER, 0] >= sim.los)
        live &= ~done
    return {"free_loss_median": float(np.median(lost_free)), "touch_loss_median": float(np.median(lost_touch)),
            "free_loss_1plus": float((lost_free >= 1.0).mean()), "touch_loss_1plus": float((lost_touch >= 1.0).mean())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--lb", action="store_true")
    p.add_argument("--combo", action="store_true")
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--stumble", type=float, default=0.0)
    p.add_argument("--square-ahead", action="store_true")
    a = p.parse_args()
    cfg = lane_drill_config(width=a.width, linebacker=a.lb, combo=a.combo, field=FieldConfig(reaction_delay=2),
                            stumble=a.stumble, carrier_square_ahead=a.square_ahead,
                            tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    for s in sorted(d for d in a.exp.iterdir() if (d / "checkpoints").is_dir()):
        ck = s / "checkpoints"
        r = slowdown(ppo_runner(ck / f"runner_g{a.gen:02d}.zip"), ppo_tackler(ck / f"tackler_g{a.gen:02d}.zip"), cfg)
        print(f"{s.name}: speed lost before the line, on his own: median {r['free_loss_median']:.2f} yd/s "
              f"(1+ yd/s in {r['free_loss_1plus']:.0%}) | touching: median {r['touch_loss_median']:.2f} "
              f"(1+ in {r['touch_loss_1plus']:.0%})")

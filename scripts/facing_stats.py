"""Do learned players use their hips like NFL players? (EXP-008)

For a runner/tackler pair, while each player moves faster than 1.5 yd/s,
measure the angle between where the body faces and where it moves (0 = running
forward, 180 = backpedal), and the tackler's square-up action by distance to
the runner. NFL reference: `sim/orientation_params.json` (share of moving
frames with off-angle > 90 deg: DB 26%, WR 8%).

    uv run python -m scripts.facing_stats --exp experiments/EXP-008-facing --gen 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.core import RUNNER, TACKLER, FieldConfig, OneVOneSim, SimConfig, TackleConfig
from train.evaluate import ppo_runner, ppo_tackler

ORIENT = Path(__file__).resolve().parents[1] / "sim/orientation_params.json"


def off_angles(sim: OneVOneSim, who: int) -> np.ndarray:
    v = sim.vel[:, who]
    sp = np.linalg.norm(v, axis=1)
    cos = np.sum(v * sim.facing[:, who], axis=1) / np.maximum(sp, 1e-6)
    return np.where(sp > 1.5, np.degrees(np.arccos(np.clip(cos, -1, 1))), np.nan)


def run(runner, tackler_model, cfg: SimConfig, n: int = 2048, seed: int = 7) -> dict:
    sim = OneVOneSim(n, cfg, seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    offs = {RUNNER: [], TACKLER: []}
    squares, dists = [], []
    while live.any():
        t_obs = sim.observe_tackler()
        t_act, _ = tackler_model.predict(t_obs, deterministic=True)
        dist = np.linalg.norm(sim.r_pos - sim.t_pos, axis=1)
        squares.append(np.clip(t_act[live, 2], 0, 1))
        dists.append(dist[live])
        obs, _, done, _ = sim.step(runner(sim, obs), t_act)
        for who in (RUNNER, TACKLER):
            offs[who].append(off_angles(sim, who)[live])
        live &= ~done
    out = {}
    for who, name in ((RUNNER, "runner"), (TACKLER, "tackler")):
        o = np.concatenate(offs[who])
        o = o[~np.isnan(o)]
        out[name] = {"off_median": round(float(np.median(o)), 1),
                     "share_off_gt90": round(float((o > 90).mean()), 3),
                     "share_off_gt135": round(float((o > 135).mean()), 3)}
    sq, d = np.concatenate(squares), np.concatenate(dists)
    out["tackler_square_by_dist"] = {f"{lo}-{hi}yd": round(float(sq[(d >= lo) & (d < hi)].mean()), 2)
                                     for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 60)) if ((d >= lo) & (d < hi)).any()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--gen", type=int, nargs="+", default=[12])
    args = ap.parse_args()
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=8.0, reaction_delay=2),
                    tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2), facing=True)
    from stable_baselines3 import PPO
    nfl = json.loads(ORIENT.read_text())["groups"]
    print(f"NFL  DB off>90 {nfl['DB']['share_off_gt90']:.1%}  WR off>90 {nfl['WR']['share_off_gt90']:.1%}")
    ck = args.exp / "checkpoints"
    for g in args.gen:
        res = run(ppo_runner(ck / f"runner_g{g:02d}.zip"), PPO.load(str(ck / f"tackler_g{g:02d}.zip"), device="cpu"), cfg)
        print(f"G{g:02d}", json.dumps(res))


if __name__ == "__main__":
    main()

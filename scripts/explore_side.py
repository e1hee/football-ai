"""Does the offense's exploration ever try the other side for a whole play? (EXP-031)

EXP-030 kept the per-step noise turning the push by ~34 degrees, yet the offense never moved to
the other side even where that side was empty. Per-step noise is drawn fresh every 0.1 s and
averages out within a play; gSDE noise is held for a while. For each seed and generation: the
offense against its own generation's defense (defense on its mean), 1024 plays, once on the
offense's mean and `--draws` times with its exploration noise (gSDE: one noise per play).

flip: share of noisy plays that cross the line on the other side of the middle from the same
start on the mean (only starts whose mean crossing is more than 1 yd off the middle).

    PYTHONPATH=. uv run python -m scripts.explore_side experiments/EXP-031-sde/A --gen 1 6 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from scripts.read_value import run
from train.policies import resample_noise
from train.run_config import exp_config


def actor(model: PPO, side: str, deterministic: bool):
    def act(sim, obs):
        ob = sim.observe() if side == "off" else sim.observe_tackler()
        return model.predict(ob, deterministic=deterministic)[0].astype(np.float32)
    return act


def measure(off: PPO, dfn: PPO, cfg, n: int = 1024, draws: int = 4, seed: int = 7) -> dict:
    d = actor(dfn, "def", True)
    base = run(actor(off, "off", True), d, cfg, n, seed)
    far = np.abs(base["y"]) > 1.0
    flips, box, yards = [], [], []
    for k in range(draws):
        resample_noise(off.policy, n)
        noisy = run(actor(off, "off", False), d, cfg, n, seed)
        both = far & ~np.isnan(noisy["y"])
        flips.append(float((np.sign(noisy["y"][both]) != np.sign(base["y"][both])).mean()) if both.any() else np.nan)
        crossed = ~np.isnan(noisy["y"])
        box.append(float((np.abs(noisy["y"][crossed]) <= cfg.reward.lane_box).mean()))
        yards.append(float(noisy["yards"].mean()))
    crossed = ~np.isnan(base["y"])
    return {"flip": float(np.nanmean(flips)) if not np.isnan(flips).all() else float("nan"), "box_noisy": float(np.mean(box)), "yards_noisy": float(np.mean(yards)),
            "box_mean": float((np.abs(base["y"][crossed]) <= cfg.reward.lane_box).mean()),
            "yards_mean": float(base["yards"].mean()), "far_share": float(far.mean()),
            "left_mean": float((base["y"][crossed] < 0).mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--draws", type=int, default=4)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    rows = []
    for s in range(3):
        ck = args.exp / f"s{s}" / "checkpoints"
        for g in args.gen:
            off = PPO.load(str(ck / f"runner_g{g:02d}.zip"), device="cpu")
            dfn = PPO.load(str(ck / f"tackler_g{g:02d}.zip"), device="cpu")
            r = {"seed": s, "gen": g, **measure(off, dfn, cfg, draws=args.draws)}
            rows.append(r)
            print(f"s{s} g{g:02d} flip {r['flip']*100:4.1f}% | box mean {r['box_mean']*100:3.0f}% noisy {r['box_noisy']*100:3.0f}% "
                  f"| yd mean {r['yards_mean']:.2f} noisy {r['yards_noisy']:.2f} | starts off-middle {r['far_share']*100:.0f}% "
                  f"| left {r['left_mean']*100:.0f}%",
                  flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

"""How long the mean push is, and how much the exploration noise can still turn it (EXP-029, EXP-030).

The sim scales any push longer than 1 down to length 1, so past 1 only its direction counts and
the same noise turns it less. For each seed and generation: offense against its own generation's
defense, 1024 plays, the first 1.5 s (both sides act on their mean). Per side: mean push length
of the runner, OL, DL and LB, the share past 1, and the average angle by which the side's own
noise turns the push (fresh networks: about 60 degrees).

    PYTHONPATH=. uv run python -m scripts.push_size experiments/EXP-030-bounded/A --gen 1 3 6 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from sim.team import TeamSim
from train.policies import action_std
from train.run_config import exp_config


def _mean(model: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.mean.numpy()


def _turn(mu: np.ndarray, std: np.ndarray, rng) -> float:
    """`std` (same shape as `mu`): per-sample noise std, constant without gSDE (EXP-031)."""
    e = mu[..., None, :] + std[..., None, :] * rng.normal(size=mu.shape[:-1] + (64, 2))
    d = np.arctan2(e[..., 1], e[..., 0]) - np.arctan2(mu[..., 1], mu[..., 0])[..., None]
    return float(np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi)).mean())


def measure(off_path: Path, def_path: Path, cfg, n: int = 1024, steps: int = 15) -> dict:
    off, dfn = PPO.load(str(off_path), device="cpu"), PPO.load(str(def_path), device="cpu")
    sim = TeamSim(n, cfg, seed=7)
    obs = sim.reset()
    mo, md, so, sd = [], [], [], []
    for _ in range(steps):
        ot = sim.observe_tackler()
        a, b = _mean(off, obs), _mean(dfn, ot)
        mo.append(a), md.append(b), so.append(action_std(off.policy, obs)), sd.append(action_std(dfn.policy, ot))
        obs, _, _, _ = sim.step(a.astype(np.float32), b.astype(np.float32))
    rng = np.random.default_rng(0)
    out = {}
    for side, mus, sds, groups in (("offense", mo, so, {"RB": [0], "OL": [1, 2, 3, 4, 5]}),
                                   ("defense", md, sd, {"DL": [0, 1, 2, 3], "LB": [4, 5]})):
        mu = np.concatenate(mus).reshape(-1, 6, 3)[..., :2]
        std = np.concatenate(sds).reshape(-1, 6, 3)[..., :2]
        length = np.linalg.norm(mu, axis=-1)
        for g, idx in groups.items():
            out[f"{g}_length"] = float(length[:, idx].mean())
            out[f"{g}_past1"] = float((length[:, idx] > 1).mean())
            out[f"{g}_turn_deg"] = _turn(mu[:, idx], std[:, idx], rng)
        out[f"{side}_std"] = float(std.mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    rows = []
    for s in range(3):
        ck = args.exp / f"s{s}" / "checkpoints"
        for g in args.gen:
            r = {"seed": s, "gen": g, **measure(ck / f"runner_g{g:02d}.zip", ck / f"tackler_g{g:02d}.zip", cfg)}
            rows.append(r)
            print(f"s{s} g{g:02d} " + " | ".join(f"{k} {r[k+'_length']:.2f} ({r[k+'_past1']*100:3.0f}% >1, turn {r[k+'_turn_deg']:2.0f} deg)"
                                               for k in ("RB", "OL", "DL", "LB"))
                  + f" | std off {r['offense_std']:.2f} def {r['defense_std']:.2f}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

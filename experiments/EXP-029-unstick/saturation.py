"""How much the exploration noise can still turn each player (EXP-029).

The sim scales any push longer than 1 down to length 1 (`TeamSim.step`), so only its direction
counts once the policy's mean push is long. Noise of the same std then turns that direction less.
For each offense: mean push length of the runner and the OL over the first 1.5 s of 1024 plays
against its own generation-12 defense (and, for the
EXP-028 A offense at generations 1 to 12, against that generation's defense), the share that is past 1, and the average angle by which
the policy's own noise turns the push.

    PYTHONPATH=. uv run python experiments/EXP-029-unstick/saturation.py
"""
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from sim.team import TeamSim
from train.evaluate import ppo_tackler
from train.run_config import exp_config

X = Path("experiments/EXP-028-older/A")
O = Path("experiments/EXP-029-unstick/checkpoints")


def stats(path: Path, s: int, cfg, gen: int = 12) -> str:
    m = PPO.load(str(path), device="cpu")
    d = ppo_tackler(X / f"s{s}/checkpoints/tackler_g{gen:02d}.zip")
    sim = TeamSim(1024, cfg, seed=7)
    obs = sim.reset()
    mus = []
    for _ in range(15):
        with torch.no_grad():
            mu = m.policy.get_distribution(m.policy.obs_to_tensor(obs)[0]).distribution.mean.numpy()
        mus.append(mu)
        obs, _, _, _ = sim.step(mu.astype(np.float32), d(sim, obs))
    mu = np.concatenate(mus).reshape(-1, 6, 3)[..., :2]
    pn = np.linalg.norm(mu, axis=-1)
    std = float(m.policy.log_std.exp().detach().numpy().reshape(6, 3)[:, :2].mean())
    e = mu[..., None, :] + std * np.random.default_rng(0).normal(size=mu.shape[:2] + (64, 2))
    turn = np.abs((np.arctan2(e[..., 1], e[..., 0]) - np.arctan2(mu[..., 1], mu[..., 0])[..., None] + np.pi) % (2 * np.pi) - np.pi)
    return (f"push length RB {pn[:, 0].mean():.2f} OL {pn[:, 1:].mean():.2f} | past 1: RB {(pn[:, 0] > 1).mean()*100:3.0f}% "
            f"OL {(pn[:, 1:] > 1).mean()*100:3.0f}% | noise turns the push {np.degrees(turn).mean():3.0f} deg | std {std:.2f}")


def main() -> None:
    cfg = exp_config(X, None)
    for s in range(3):
        print(f"s{s} start (24M)  ", stats(X / f"s{s}/checkpoints/runner_g12.zip", s, cfg), flush=True)
        for v in ("warm", "noise", "head", "fresh"):
            print(f"s{s} {v:5} +2M     ", stats(O / f"{v}_s{s}_2000k.zip", s, cfg), flush=True)
    print("\nEXP-028 A offense through the generations, each against its own generation's defense")
    for s in range(3):
        for g in (1, 2, 3, 6, 9, 12):
            print(f"s{s} g{g:02d}          ", stats(X / f"s{s}/checkpoints/runner_g{g:02d}.zip", s, cfg, g), flush=True)


if __name__ == "__main__":
    main()

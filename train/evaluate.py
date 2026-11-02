"""Evaluate any runner policy over many plays against a fixed tackler.

A runner policy is `fn(sim, obs) -> (N, 2)`. `ppo_runner(path)` turns a saved
SB3 checkpoint into one, so learned and scripted runners are measured, and
rendered, the same way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from sim.core import OUTCOME_NAMES, OneVOneSim, SimConfig

RunnerPolicy = Callable[[OneVOneSim, np.ndarray], np.ndarray]


def ppo_runner(path: str | Path, deterministic: bool = True) -> RunnerPolicy:
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device="cpu")

    def act(sim: OneVOneSim, obs: np.ndarray) -> np.ndarray:
        a, _ = model.predict(obs, deterministic=deterministic)
        return a.astype(np.float32)
    return act


def ppo_tackler(path: str | Path, deterministic: bool = True):
    """Saved SB3 tackler as `fn(sim, *_) -> (N, 2)`; it builds its own observation."""
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device="cpu")

    def act(sim: OneVOneSim, *_) -> np.ndarray:
        a, _ = model.predict(sim.observe_tackler(), deterministic=deterministic)
        return a.astype(np.float32)
    return act


def evaluate(runner: RunnerPolicy, tackler: Callable[[OneVOneSim], np.ndarray], cfg: SimConfig,
             n_plays: int = 4096, seed: int = 12345) -> dict:
    """Play exactly one play in each of `n_plays` slots (fixed seed = same starts every time)."""
    sim = OneVOneSim(n_plays, cfg, seed=seed)
    obs = sim.reset()
    finished = np.zeros(n_plays, bool)
    outcome = np.zeros(n_plays, np.int8)
    yards = np.zeros(n_plays, np.float32)
    attempts = np.zeros(n_plays, np.int32)
    made = np.zeros(n_plays, np.int32)
    while not finished.all():
        obs, _, done, info = sim.step(runner(sim, obs), tackler(sim))
        live = ~finished
        attempts += (info["tackle_attempt"] & live)
        made += (info["tackle_made"] & live)
        newly = done & live
        outcome[newly] = info["outcome"][newly]
        yards[newly] = info["yards"][newly]
        finished |= done
    res = {f"rate/{k}": float((outcome == i).mean()) for i, k in enumerate(OUTCOME_NAMES) if k != "running"}
    res.update({
        "yards/mean": float(yards.mean()),
        "yards/median": float(np.median(yards)),
        "tackle/attempts_per_play": float(attempts.mean()),
        "tackle/broken_per_play": float((attempts - made).mean()),
    })
    return res

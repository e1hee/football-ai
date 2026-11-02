"""Scripted (non-learning) policies. Used as fixed opponents and baselines."""
from __future__ import annotations

import numpy as np

from .core import OneVOneSim


def random_policy(sim: OneVOneSim, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, (sim.n, 2)).astype(np.float32)


def straight_runner(sim: OneVOneSim) -> np.ndarray:
    """Baseline runner: sprint straight downfield, ignore the tackler."""
    a = np.zeros((sim.n, 2), np.float32)
    a[:, 0] = 1.0
    return a


def pursuit_tackler(sim: OneVOneSim, lead_time: float = 0.3) -> np.ndarray:
    """Chase the runner's predicted position `lead_time` seconds ahead.

    A plain pursuit curve is what a learner has to beat first; it is also the
    scripted opponent for the earliest runner-training experiments.
    """
    target = sim.r_pos + sim.r_vel * lead_time
    d = target - sim.t_pos
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return (d / np.maximum(norm, 1e-6)).astype(np.float32)

"""Scripted (non-learning) policies. Used as fixed opponents and baselines."""
from __future__ import annotations

import numpy as np

from .core import RUNNER, TACKLER, OneVOneSim


def random_policy(sim: OneVOneSim, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, (sim.n, 2)).astype(np.float32)


def straight_runner(sim: OneVOneSim) -> np.ndarray:
    """Baseline runner: sprint straight downfield, ignore the tackler."""
    a = np.zeros((sim.n, 2), np.float32)
    a[:, 0] = 1.0
    return a


def evasive_runner(sim: OneVOneSim, danger: float = 6.0) -> np.ndarray:
    """Hand-written 'smart' runner: sprint downfield, and when the tackler is
    within `danger` yards ahead, veer away from it (harder the closer it is),
    staying off the sideline. The first bar a learned runner has to clear."""
    t_pos, _ = sim.seen(TACKLER)
    rel = t_pos - sim.r_pos
    dist = np.linalg.norm(rel, axis=1)
    ahead = rel[:, 0] > -1.0
    side = -np.sign(rel[:, 1] + 1e-3)
    # Don't veer into the sideline: steer back toward the middle near the edge.
    to_mid = np.sign(sim.cfg.field.width / 2 - sim.r_pos[:, 1])
    near_edge = np.minimum(sim.r_pos[:, 1], sim.cfg.field.width - sim.r_pos[:, 1]) < 4.0
    side = np.where(near_edge, to_mid, side)
    strength = np.where(ahead & (dist < danger), 1.0 - dist / danger, 0.0)
    a = np.stack([np.ones(sim.n), side * strength * 1.5], axis=1)
    return a.astype(np.float32)


def pursuit_tackler(sim: OneVOneSim, lead_time: float = 0.3) -> np.ndarray:
    """Chase the runner's predicted position `lead_time` seconds ahead.

    A plain pursuit curve is what a learner has to beat first; it is also the
    scripted opponent for the earliest runner-training experiments.
    """
    r_pos, r_vel = sim.seen(RUNNER)
    target = r_pos + r_vel * lead_time
    d = target - sim.t_pos
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return (d / np.maximum(norm, 1e-6)).astype(np.float32)


def angle_tackler(sim: OneVOneSim, breakdown: float = 3.0, breakdown_speed: float = 4.5) -> np.ndarray:
    """What coaches teach: take a pursuit angle and break down before contact.

    - Aim at where the runner will be when you can get there (intercept),
      never behind him: the target is kept goal-side of the runner.
    - Inside `breakdown` yards, chop your feet: cap your speed so you can
      still change direction with him instead of flying past.
    """
    r_pos, r_vel = sim.seen(RUNNER)
    rel = r_pos - sim.t_pos
    dist = np.linalg.norm(rel, axis=1)
    t_speed = np.linalg.norm(sim.t_vel, axis=1)
    closing = np.maximum(sim.bodies.max_speed[:, 1] * 0.8, 1.0)
    t_hit = np.clip(dist / closing, 0.0, 1.5)
    target = r_pos + r_vel * t_hit[:, None]
    # Stay between the runner and the goal line.
    target[:, 0] = np.maximum(target[:, 0], r_pos[:, 0] + 0.5)
    d = target - sim.t_pos
    a = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    # Break down: if close and too fast, push against own velocity.
    too_fast = (dist < breakdown) & (t_speed > breakdown_speed)
    brake = -sim.t_vel / np.maximum(t_speed, 1e-6)[:, None]
    a = np.where(too_fast[:, None], 0.5 * a + 0.5 * brake, a)
    return a.astype(np.float32)

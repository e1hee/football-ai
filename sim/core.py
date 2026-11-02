"""Vectorized 1v1 sim: one ball carrier (runner) vs one defender (tackler).

Everything is batched over N parallel plays so training can step thousands
of plays per second without a Python loop. No physics engine: players are
circles with velocity caps, and a tackle is a proximity check.

Coordinates: x runs downfield (0 = runner's start side, `length` = goal
line), y runs across the field (0 and `width` are the sidelines).
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np


@dataclass
class FieldConfig:
    length: float = 60.0        # yards from the runner's start to the goal line
    width: float = 53.3         # NFL field width in yards
    runner_start_x: float = 5.0
    tackler_gap: float = 10.0   # tackler starts this many yards downfield of the runner
    dt: float = 0.1             # seconds per step (10 Hz)
    max_steps: int = 100        # 10 s play clock
    tackle_radius: float = 1.0  # yards; closer than this = tackled


@dataclass
class PlayerAttrs:
    """Per-player physical attributes. Arrays of shape (N,) so each play can
    use different players (roster experiments change these)."""
    max_speed: np.ndarray   # yd/s. NFL ball carriers top out around 9-10 yd/s
    accel: np.ndarray       # yd/s^2

    @classmethod
    def uniform(cls, n: int, max_speed: float, accel: float) -> "PlayerAttrs":
        return cls(np.full(n, max_speed, np.float32), np.full(n, accel, np.float32))


@dataclass
class RewardConfig:
    """Runner's reward. Start minimal; every extra term is an experiment."""
    yards_scale: float = 1.0        # reward per yard gained this step
    touchdown_bonus: float = 0.0
    tackle_penalty: float = 0.0
    out_of_bounds_penalty: float = 0.0
    time_penalty: float = 0.0       # per step


@dataclass
class SimConfig:
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    reward: RewardConfig = dc_field(default_factory=RewardConfig)
    runner_speed: float = 9.0
    runner_accel: float = 8.0
    tackler_speed: float = 9.0
    tackler_accel: float = 8.0


# Observation layout for the runner (egocentric where it matters).
OBS_DIM = 10
RUNNER_OBS_NAMES = [
    "x_norm", "y_norm",                     # own position / field size
    "vx_norm", "vy_norm",                   # own velocity / max speed
    "rel_tackler_x", "rel_tackler_y",       # tackler offset / 20 yd
    "tackler_vx_norm", "tackler_vy_norm",   # tackler velocity / max speed
    "dist_goal_norm",                       # yards to goal line / length
    "dist_sideline_norm",                   # yards to nearest sideline / half width
]

# Episode outcome codes written to `info`.
OUTCOME_RUNNING, OUTCOME_TOUCHDOWN, OUTCOME_TACKLED, OUTCOME_OUT_OF_BOUNDS, OUTCOME_TIMEOUT = 0, 1, 2, 3, 4
OUTCOME_NAMES = ["running", "touchdown", "tackled", "out_of_bounds", "timeout"]


class OneVOneSim:
    """N parallel runner-vs-tackler plays.

    `step(runner_action, tackler_action)` advances every play one tick and
    auto-resets finished plays, returning the observation of the *new* play
    for those slots (the standard vectorized-env convention).
    """

    def __init__(self, n: int, cfg: SimConfig | None = None, seed: int = 0):
        self.n = n
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        f = self.cfg.field
        self.runner = PlayerAttrs.uniform(n, self.cfg.runner_speed, self.cfg.runner_accel)
        self.tackler = PlayerAttrs.uniform(n, self.cfg.tackler_speed, self.cfg.tackler_accel)

        self.r_pos = np.zeros((n, 2), np.float32)
        self.r_vel = np.zeros((n, 2), np.float32)
        self.t_pos = np.zeros((n, 2), np.float32)
        self.t_vel = np.zeros((n, 2), np.float32)
        self.t = np.zeros(n, np.int32)
        self.yards = np.zeros(n, np.float32)  # yards gained so far in this play
        self._half_w = f.width / 2.0

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Reset all plays (mask=None) or only those where mask is True."""
        if mask is None:
            mask = np.ones(self.n, bool)
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        f = self.cfg.field
        # Runner starts centered-ish with a little jitter; tackler lines up
        # `tackler_gap` yards downfield at a random lateral spot.
        self.r_pos[mask, 0] = f.runner_start_x
        self.r_pos[mask, 1] = self._half_w + self.rng.uniform(-5, 5, k)
        self.t_pos[mask, 0] = f.runner_start_x + f.tackler_gap
        self.t_pos[mask, 1] = self.rng.uniform(5, f.width - 5, k)
        self.r_vel[mask] = 0
        self.t_vel[mask] = 0
        self.t[mask] = 0
        self.yards[mask] = 0
        return self.observe()

    # ---------------------------------------------------------------- physics
    @staticmethod
    def _integrate(pos, vel, action, attrs: PlayerAttrs, dt: float):
        """action: (N,2) desired acceleration direction in [-1,1]^2."""
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.where(norm > 1.0, a / np.maximum(norm, 1e-6), a)   # keep inside unit disk
        vel += a * attrs.accel[:, None] * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        cap = attrs.max_speed[:, None]
        vel *= np.where(speed > cap, cap / np.maximum(speed, 1e-6), 1.0)
        pos += vel * dt

    # ------------------------------------------------------------------- step
    def step(self, runner_action: np.ndarray, tackler_action: np.ndarray):
        """Returns (obs, reward, done, info). Reward is the runner's; the
        tackler's is its negation (zero-sum) and is exposed in info."""
        f, rw = self.cfg.field, self.cfg.reward
        prev_x = self.r_pos[:, 0].copy()

        self._integrate(self.r_pos, self.r_vel, runner_action, self.runner, f.dt)
        self._integrate(self.t_pos, self.t_vel, tackler_action, self.tackler, f.dt)
        self.t += 1

        # Field edges: x is clamped (you can't run backwards out of the sim),
        # crossing a sideline in y ends the play.
        self.r_pos[:, 0] = np.clip(self.r_pos[:, 0], 0.0, f.length)
        self.t_pos[:, 0] = np.clip(self.t_pos[:, 0], 0.0, f.length)
        self.t_pos[:, 1] = np.clip(self.t_pos[:, 1], 0.0, f.width)

        gained = self.r_pos[:, 0] - prev_x
        self.yards += gained

        dist = np.linalg.norm(self.r_pos - self.t_pos, axis=1)
        touchdown = self.r_pos[:, 0] >= f.length
        out_of_bounds = (self.r_pos[:, 1] < 0.0) | (self.r_pos[:, 1] > f.width)
        tackled = (dist < f.tackle_radius) & ~touchdown & ~out_of_bounds
        timeout = (self.t >= f.max_steps) & ~touchdown & ~out_of_bounds & ~tackled
        done = touchdown | tackled | out_of_bounds | timeout

        outcome = np.zeros(self.n, np.int8)
        outcome[touchdown] = OUTCOME_TOUCHDOWN
        outcome[tackled] = OUTCOME_TACKLED
        outcome[out_of_bounds] = OUTCOME_OUT_OF_BOUNDS
        outcome[timeout] = OUTCOME_TIMEOUT

        reward = (
            rw.yards_scale * gained
            + rw.touchdown_bonus * touchdown
            - rw.tackle_penalty * tackled
            - rw.out_of_bounds_penalty * out_of_bounds
            - rw.time_penalty
        ).astype(np.float32)

        info = {
            "outcome": outcome,
            "yards": self.yards.copy(),
            "steps": self.t.copy(),
            "tackler_reward": -reward,
            # Final positions before auto-reset, for rendering/analysis.
            "r_pos": self.r_pos.copy(), "t_pos": self.t_pos.copy(),
        }
        if done.any():
            self.reset(done)
        return self.observe(), reward, done, info

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        f = self.cfg.field
        obs = np.empty((self.n, OBS_DIM), np.float32)
        obs[:, 0] = self.r_pos[:, 0] / f.length
        obs[:, 1] = self.r_pos[:, 1] / f.width
        obs[:, 2:4] = self.r_vel / self.runner.max_speed[:, None]
        obs[:, 4:6] = (self.t_pos - self.r_pos) / 20.0
        obs[:, 6:8] = self.t_vel / self.tackler.max_speed[:, None]
        obs[:, 8] = (f.length - self.r_pos[:, 0]) / f.length
        obs[:, 9] = np.minimum(self.r_pos[:, 1], f.width - self.r_pos[:, 1]) / self._half_w
        return obs

    def observe_tackler(self) -> np.ndarray:
        """Mirror observation for the tackler (used once it learns too)."""
        f = self.cfg.field
        obs = np.empty((self.n, OBS_DIM), np.float32)
        obs[:, 0] = self.t_pos[:, 0] / f.length
        obs[:, 1] = self.t_pos[:, 1] / f.width
        obs[:, 2:4] = self.t_vel / self.tackler.max_speed[:, None]
        obs[:, 4:6] = (self.r_pos - self.t_pos) / 20.0
        obs[:, 6:8] = self.r_vel / self.runner.max_speed[:, None]
        obs[:, 8] = (f.length - self.r_pos[:, 0]) / f.length
        obs[:, 9] = np.minimum(self.t_pos[:, 1], f.width - self.t_pos[:, 1]) / self._half_w
        return obs

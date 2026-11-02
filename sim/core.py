"""Vectorized 1v1 sim: one ball carrier (runner) vs one defender (tackler).

Everything is batched over N parallel plays. Players are discs with
NFL-fitted movement limits (`physics.py`): momentum, speed-dependent
acceleration, limited cutting at speed, and body collisions.

Tackling is a contest, not a switch. When the tackler gets within arm's
reach, it attempts a tackle whose success depends on
  - angle: meeting the runner head-on beats chasing from behind, and
  - momentum: a runner carrying more momentum than the tackler can break it.
A made tackle wraps the runner up for a moment (they keep sliding forward
together: yards after contact). A missed tackle leaves the tackler stumbling.

Coordinates: x runs downfield (0 = runner's start side, `length` = goal
line), y runs across the field (0 and `width` are the sidelines).
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from .physics import PROFILES, Bodies, Profile, apply_controls, resolve_collisions

RUNNER, TACKLER = 0, 1


@dataclass
class FieldConfig:
    length: float = 60.0        # yards from the runner's start to the goal line
    width: float = 53.3         # NFL field width in yards
    runner_start_x: float = 5.0
    tackler_gap: float = 10.0   # tackler starts this many yards downfield of the runner
    # None: tackler anywhere across the field (EXP-000). A number: within that
    # many yards of the runner's lane.
    tackler_lateral_spread: float | None = None
    dt: float = 0.1             # seconds per step (10 Hz, same as NFL tracking)
    max_steps: int = 100        # 10 s play clock


@dataclass
class TackleConfig:
    reach: float = 0.5          # yd of arm reach beyond body contact
    base_logit: float = 1.2     # side-on, equal momentum -> ~77% success
    headon_weight: float = 1.0  # head-on (+1) vs from behind (-1)
    momentum_weight: float = 3.0
    wrap_time: float = 0.4      # s the pile keeps moving after a made tackle
    wrap_decel: float = 6.0     # yd/s^2 while wrapped up
    miss_stun: float = 0.6      # s the tackler has no control after a miss
    miss_cooldown: float = 0.9  # s before the same tackler can try again
    miss_slowdown: float = 0.85 # runner keeps this share of velocity after breaking a tackle


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
    tackle: TackleConfig = dc_field(default_factory=TackleConfig)
    runner: Profile = dc_field(default_factory=lambda: PROFILES["WR"])
    tackler: Profile = dc_field(default_factory=lambda: PROFILES["DB"])


OBS_DIM = 12
RUNNER_OBS_NAMES = [
    "x_norm", "y_norm",                     # own position / field size
    "vx_norm", "vy_norm",                   # own velocity / max speed
    "rel_tackler_x", "rel_tackler_y",       # tackler offset / 20 yd
    "tackler_vx_norm", "tackler_vy_norm",   # tackler velocity / max speed
    "dist_goal_norm",                       # yards to goal line / length
    "dist_sideline_norm",                   # yards to nearest sideline / half width
    "tackler_stunned",                      # remaining stumble / miss_stun (0 = recovered)
    "wrapped",                              # 1 while the runner is wrapped up
]

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
        self.bodies = Bodies(n, [self.cfg.runner, self.cfg.tackler])

        self.pos = np.zeros((n, 2, 2), np.float32)   # (play, player, xy)
        self.vel = np.zeros((n, 2, 2), np.float32)
        self.t = np.zeros(n, np.int32)
        self.yards = np.zeros(n, np.float32)
        self.wrap = np.zeros(n, np.float32)          # >0: runner wrapped up, play ending
        self.stun = np.zeros((n, 2), np.float32)     # >0: no control
        self.cooldown = np.zeros(n, np.float32)      # >0: tackler can't attempt
        self._half_w = self.cfg.field.width / 2.0

    # Views kept for renderers/tests written against the original API.
    @property
    def r_pos(self): return self.pos[:, RUNNER]
    @property
    def t_pos(self): return self.pos[:, TACKLER]
    @property
    def r_vel(self): return self.vel[:, RUNNER]
    @property
    def t_vel(self): return self.vel[:, TACKLER]

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(self.n, bool)
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        f = self.cfg.field
        ry = self._half_w + self.rng.uniform(-5, 5, k)
        self.pos[mask, RUNNER, 0] = f.runner_start_x
        self.pos[mask, RUNNER, 1] = ry
        self.pos[mask, TACKLER, 0] = f.runner_start_x + f.tackler_gap
        if f.tackler_lateral_spread is None:
            self.pos[mask, TACKLER, 1] = self.rng.uniform(5, f.width - 5, k)
        else:
            s = f.tackler_lateral_spread
            self.pos[mask, TACKLER, 1] = np.clip(ry + self.rng.uniform(-s, s, k), 1, f.width - 1)
        self.vel[mask] = 0
        self.t[mask] = 0
        self.yards[mask] = 0
        self.wrap[mask] = 0
        self.stun[mask] = 0
        self.cooldown[mask] = 0
        return self.observe()

    # --------------------------------------------------------------- tackling
    def _tackle_attempts(self, pre_vel: np.ndarray):
        """Decide new tackle attempts. Returns (attempt, made) bool arrays."""
        tc, b = self.cfg.tackle, self.bodies
        d = self.pos[:, RUNNER] - self.pos[:, TACKLER]               # tackler -> runner
        dist = np.linalg.norm(d, axis=1)
        reach = b.radius[:, RUNNER] + b.radius[:, TACKLER] + tc.reach
        attempt = (dist < reach) & (self.wrap <= 0) & (self.cooldown <= 0) & (self.stun[:, TACKLER] <= 0)
        made = np.zeros(self.n, bool)
        if not attempt.any():
            return attempt, made

        u = d / np.maximum(dist, 1e-6)[:, None]
        vr, vt = pre_vel[:, RUNNER], pre_vel[:, TACKLER]
        r_speed = np.linalg.norm(vr, axis=1)
        # Head-on: runner moving toward the tackler. From behind: moving away.
        headon = np.clip(np.sum(vr * -u, axis=1) / np.maximum(r_speed, 1.0), -1, 1)
        p_r = b.mass_lb[:, RUNNER] * r_speed
        p_t = b.mass_lb[:, TACKLER] * np.maximum(0.0, np.sum(vt * u, axis=1))
        ratio = p_r / (p_r + p_t + 1e-6)
        logit = tc.base_logit + tc.headon_weight * headon - tc.momentum_weight * (ratio - 0.5)
        p = 1.0 / (1.0 + np.exp(-logit))
        made = attempt & (self.rng.random(self.n) < p)
        return attempt, made

    # ------------------------------------------------------------------- step
    def step(self, runner_action: np.ndarray, tackler_action: np.ndarray):
        """Returns (obs, reward, done, info). Reward is the runner's; the
        tackler's is its negation (zero-sum) and is exposed in info."""
        f, rw, tc = self.cfg.field, self.cfg.reward, self.cfg.tackle
        dt = f.dt
        prev_x = self.pos[:, RUNNER, 0].copy()
        wrapped = self.wrap > 0

        action = np.stack([runner_action, tackler_action], axis=1).astype(np.float32)
        apply_controls(self.pos, self.vel, action, self.bodies, dt, stunned=self.stun > 0)

        # Wrapped-up pile: both move together at their shared momentum, slowing down.
        if wrapped.any():
            m = self.bodies.mass_lb[wrapped]
            v_pile = (self.vel[wrapped] * m[..., None]).sum(1) / m.sum(1)[:, None]
            sp = np.linalg.norm(v_pile, axis=1, keepdims=True)
            v_pile *= np.maximum(0.0, sp - tc.wrap_decel * dt) / np.maximum(sp, 1e-6)
            prev = self.pos[wrapped] - self.vel[wrapped] * dt     # undo free movement
            self.vel[wrapped] = v_pile[:, None, :]
            self.pos[wrapped] = prev + v_pile[:, None, :] * dt

        pre_vel = self.vel.copy()
        resolve_collisions(self.pos, self.vel, self.bodies)
        attempt, made = self._tackle_attempts(pre_vel)
        missed = attempt & ~made
        self.wrap[made] = tc.wrap_time
        self.stun[missed, TACKLER] = tc.miss_stun
        self.cooldown[missed] = tc.miss_cooldown
        self.vel[missed, RUNNER] *= tc.miss_slowdown

        self.t += 1
        self.stun = np.maximum(0.0, self.stun - dt)
        self.cooldown = np.maximum(0.0, self.cooldown - dt)
        self.wrap[wrapped] -= dt                     # plays wrapped before this step
        tackled_now = wrapped & (self.wrap <= 1e-6)

        # Field edges: x is clamped, crossing a sideline in y ends the play.
        self.pos[:, :, 0] = np.clip(self.pos[:, :, 0], 0.0, f.length)
        self.pos[:, TACKLER, 1] = np.clip(self.pos[:, TACKLER, 1], 0.0, f.width)

        gained = self.pos[:, RUNNER, 0] - prev_x
        self.yards += gained

        ry = self.pos[:, RUNNER, 1]
        touchdown = self.pos[:, RUNNER, 0] >= f.length
        out_of_bounds = ((ry < 0.0) | (ry > f.width)) & ~touchdown
        tackled = tackled_now & ~touchdown & ~out_of_bounds
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
            "tackle_attempt": attempt, "tackle_made": made, "tackle_missed": missed,
            "wrapped": self.wrap > 0,
            # Final positions before auto-reset, for rendering/analysis.
            "r_pos": self.pos[:, RUNNER].copy(), "t_pos": self.pos[:, TACKLER].copy(),
            # Last observation of finished plays (RL bootstrapping on time-outs).
            "terminal_obs": self.observe() if done.any() else None,
        }
        if done.any():
            self.reset(done)
        return self.observe(), reward, done, info

    # ------------------------------------------------------------ observation
    def _obs(self, me: int, other: int) -> np.ndarray:
        f, b = self.cfg.field, self.bodies
        obs = np.empty((self.n, OBS_DIM), np.float32)
        obs[:, 0] = self.pos[:, me, 0] / f.length
        obs[:, 1] = self.pos[:, me, 1] / f.width
        obs[:, 2:4] = self.vel[:, me] / b.max_speed[:, me, None]
        obs[:, 4:6] = (self.pos[:, other] - self.pos[:, me]) / 20.0
        obs[:, 6:8] = self.vel[:, other] / b.max_speed[:, other, None]
        obs[:, 8] = (f.length - self.pos[:, RUNNER, 0]) / f.length
        obs[:, 9] = np.minimum(self.pos[:, me, 1], f.width - self.pos[:, me, 1]) / self._half_w
        obs[:, 10] = self.stun[:, other] / max(self.cfg.tackle.miss_stun, 1e-6)
        obs[:, 11] = (self.wrap > 0).astype(np.float32)
        return obs

    def observe(self) -> np.ndarray:
        return self._obs(RUNNER, TACKLER)

    def observe_tackler(self) -> np.ndarray:
        """Mirror observation for the tackler (used once it learns too)."""
        return self._obs(TACKLER, RUNNER)

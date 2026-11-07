"""Pass play, step 1 (EXP-039): QB + one outside receiver vs one cornerback.

No pass rush yet: the QB stands 6 yd behind the line and only decides *when* to
throw (the user's choice, 2026-11-06). The ball is aimed automatically where
the receiver will be if he keeps his current velocity (a QB's lead), and flies
straight at a fixed speed. So a receiver who changes direction after the throw
runs away from his own ball, and the timing of the break and the throw is
what the offense has to learn.

Numbers from real NFL tracking (`analysis/pass_from_bdb.py` -> `sim/pass_params.json`,
outside WR in man coverage, BDB 2023-2025): ball ~23 yd/s, throw at ~2.5 s
(p90 3.7-4.1 s), CB cushion at the snap 2.7 yd median (p75 6.6, p90 9),
catch 52%, interception 2.8%, separation to the CB at arrival 1.5 yd median.

At arrival (positions interpolated back to the exact arrival time):
  - interception: the CB within reach of the ball, int_max * sigmoid((int_at - d_cb) / int_width)
  - catch: only if the receiver is within `catch_radius` of the ball, in bounds, with
    catch_base + catch_gain * (1 - exp(-separation / catch_scale)) (~45% at 0.5 yd, ~75% open;
    BDB shows 47% at 0.5-1 yd from the CB across, n=36; with NFL separations this gives ~3% picks)
  - otherwise incomplete.
After the catch the receiver is the ball carrier and the CB tackles with the
team sim's tackle model. The receiver (slot 0 = CARRIER) can't be tackled
before he has the ball; bodies still collide (no pass interference rule yet).

Offense action (N, 6): receiver push x, y, square-up; QB throw (> threshold
throws), unused, unused. Defense (N, 3): the CB's push and square-up.
Reward: the offense earns the yards the ball moves past the line (air yards at
the catch, then yards after catch); a sack costs `sack_yards`, an interception
`int_yards`. `info["yards"]` uses the same numbers, so yards/mean matches the reward.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .core import (OUTCOME_INCOMPLETE, OUTCOME_INTERCEPTION, OUTCOME_OUT_OF_BOUNDS, OUTCOME_SACK, OUTCOME_TACKLED,
                   OUTCOME_TIMEOUT, OUTCOME_TOUCHDOWN, FieldConfig)
from .physics import apply_controls, resolve_collisions
from .team import PER_PLAYER_OBS, TeamConfig, TeamSim

WR, QB, CB = 0, 1, 2
PASS_GLOBAL_OBS = 11
END_ZONE = 10.0     # yd past the goal line (field.length) where a catch is still a touchdown


@dataclass
class PassConfig(TeamConfig):
    offense: tuple[str, ...] = ("WR", "QB")
    defense: tuple[str, ...] = ("DB",)
    los: float = 10.0                    # x of the line of scrimmage
    qb_depth: float = 6.0
    wr_from_sideline: tuple[float, float] = (8.0, 14.0)
    press_share: float = 0.5             # press: CB 1.5-3 yd off; off: 5-9 yd (NFL p10-p90 1.9-9.0)
    press_cushion: tuple[float, float] = (1.5, 3.0)
    off_cushion: tuple[float, float] = (5.0, 9.0)
    cb_shade: float = 1.0                # CB y offset from the receiver, uniform +-
    ball_speed: float = 23.0             # yd/s
    min_throw: float = 0.5               # s after the snap before the QB can throw
    throw_by: float = 4.0                # not thrown by then: sack
    sack_yards: float = 7.0
    catch_radius: float = 1.0
    catch_base: float = 0.30
    catch_gain: float = 0.45
    catch_scale: float = 1.2
    int_max: float = 0.20
    int_at: float = 0.54
    int_width: float = 0.25
    int_yards: float = 15.0
    throw_threshold: float = 1.0        # exploration noise (std >= 0.8, mean <= 1.25) rarely crosses it by accident


def pass_config(field: FieldConfig | None = None, **kw) -> PassConfig:
    field = field or FieldConfig()
    los = kw.get("los", 10.0)
    field = FieldConfig(**{**field.__dict__, "width": 53.3, "length": los + kw.pop("past_los", 30.0), "runner_start_x": los,
                           "breakaway_lead": None, "max_steps": 80})
    return PassConfig(field=field, **kw)


class PassSim(TeamSim):
    """N parallel pass plays: WR (carrier slot 0), QB, CB."""

    def __init__(self, n: int, cfg: PassConfig | None = None, seed: int = 0):
        cfg = cfg or pass_config()
        super().__init__(n, cfg, seed)
        self.blk = np.arange(0)                      # nobody blocks
        self.n_global = PASS_GLOBAL_OBS
        self.obs_dim = self.n_global + PER_PLAYER_OBS * self.p
        self.thrown = np.zeros(n, bool)
        self.resolved = np.zeros(n, bool)            # the ball has arrived (caught or not)
        self.caught = np.zeros(n, bool)
        self.throw_time = np.zeros(n, np.float32)
        self.arrive_time = np.full(n, np.inf, np.float32)
        self.throw_from = np.zeros((n, 2), np.float32)
        self.target = np.zeros((n, 2), np.float32)
        self.qb_spot = np.zeros((n, 2), np.float32)
        self.wr_inside = np.ones(n, np.float32)      # +1: the middle of the field is at larger y
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(self.n, bool)
        k = int(mask.sum())
        if k == 0 or not hasattr(self, "thrown"):     # TeamSim.__init__ has no ball yet
            return np.zeros((self.n, self.obs_dim), np.float32) if not hasattr(self, "thrown") else self.observe()
        c, f, rng = self.cfg, self.cfg.field, self.rng
        low = rng.random(k) < 0.5                    # receiver on the low-y side
        d = rng.uniform(*c.wr_from_sideline, k)
        wy = np.where(low, d, f.width - d)
        pos = np.zeros((k, self.p, 2), np.float32)
        pos[:, WR] = np.stack([np.full(k, c.los - 0.5), wy], 1)
        pos[:, QB] = np.stack([np.full(k, c.los - c.qb_depth), np.full(k, f.width / 2)], 1)
        press = rng.random(k) < c.press_share
        cushion = np.where(press, rng.uniform(*c.press_cushion, k), rng.uniform(*c.off_cushion, k))
        pos[:, CB] = np.stack([c.los + cushion, wy + rng.uniform(-c.cb_shade, c.cb_shade, k)], 1)
        self.thrown[mask] = False
        self.resolved[mask] = False
        self.caught[mask] = False
        self.arrive_time[mask] = np.inf
        self.target[mask] = 0
        self.qb_spot[mask] = pos[:, QB]
        self.wr_inside[mask] = np.where(low, 1.0, -1.0)
        self._place(mask, pos)
        self.los[mask] = c.los
        return self.observe()

    # ------------------------------------------------------------------- ball
    def lead(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where and when a ball thrown now from the QB meets the receiver if he keeps his velocity."""
        p, v, q = self.pos[rows, WR], self.vel[rows, WR], self.pos[rows, QB]
        s = self.cfg.ball_speed
        r = p - q
        a = np.sum(v * v, -1) - s * s                 # < 0: the ball is faster than any receiver
        b = 2 * np.sum(r * v, -1)
        cc = np.sum(r * r, -1)
        disc = np.maximum(b * b - 4 * a * cc, 0.0)
        t = (-b - np.sqrt(disc)) / (2 * a)
        t = np.maximum(t, 0.05)
        return p + v * t[:, None], t

    def ball_pos(self) -> np.ndarray:
        """(N, 2) where the ball is: with the QB, in the air, with the receiver, or where it landed."""
        clock = self.t * self.cfg.field.dt
        out = self.pos[:, QB].copy()
        fly = self.thrown & ~self.resolved
        if fly.any():
            a = np.clip((clock[fly] - self.throw_time[fly]) / np.maximum(self.arrive_time[fly] - self.throw_time[fly], 1e-6), 0, 1)
            out[fly] = self.throw_from[fly] + (self.target[fly] - self.throw_from[fly]) * a[:, None]
        out[self.caught] = self.pos[self.caught, WR]
        dead = self.resolved & ~self.caught
        out[dead] = self.target[dead]
        return out

    # ------------------------------------------------------------------- step
    def step(self, offense_action: np.ndarray, defense_action: np.ndarray):
        c, f, rw, tc = self.cfg, self.cfg.field, self.cfg.reward, self.cfg.tackle
        dt = f.dt
        yards_before = self.yards.copy()
        wrapped = self.wrap > 0

        action = np.concatenate([self._actions(offense_action, self.n_off), self._actions(defense_action, self.n_def)], 1)
        throw_now = (~self.thrown) & (action[:, QB, 0] > c.throw_threshold) & (self.t * dt >= c.min_throw - 1e-6)
        push = action[..., :2].copy()
        push[:, QB] = 0.0
        if c.facing:
            self._turn_hips(push, np.clip(action[..., 2], 0.0, 1.0))
        n_ = np.linalg.norm(push, axis=-1, keepdims=True)
        push = np.where(n_ > 1.0, push / np.maximum(n_, 1e-6), push)
        apply_controls(self.pos, self.vel, push, self.bodies, dt, stunned=self.stun > 0,
                       facing=self.facing if c.facing else None)
        if wrapped.any():   # carrier and the wrapping CB slide together (as in TeamSim)
            rows = np.nonzero(wrapped)[0]
            pair = np.stack([np.zeros_like(rows), self.wrapper[rows]], 1)
            m = self.bodies.mass_lb[rows[:, None], pair]
            v = self.vel[rows[:, None], pair]
            v_pile = (v * m[..., None]).sum(1) / m.sum(1)[:, None]
            sp = np.linalg.norm(v_pile, axis=1, keepdims=True)
            v_pile *= np.maximum(0.0, sp - tc.wrap_decel * dt) / np.maximum(sp, 1e-6)
            prev = self.pos[rows[:, None], pair] - v * dt
            self.vel[rows[:, None], pair] = v_pile[:, None]
            self.pos[rows[:, None], pair] = prev + v_pile[:, None] * dt
        self._pin_qb()
        pre_vel = self.vel.copy()
        resolve_collisions(self.pos, self.vel, self.bodies)
        self._pin_qb()
        if c.stumble:
            self._stumble(pre_vel)

        self.t += 1
        clock = self.t * dt
        if throw_now.any():
            rows = np.nonzero(throw_now)[0]
            tgt, flight = self.lead(rows)
            self.thrown[rows] = True
            self.throw_time[rows] = clock[rows]
            self.arrive_time[rows] = clock[rows] + flight
            self.throw_from[rows] = self.pos[rows, QB]
            self.target[rows] = tgt

        # The ball arrives: who is there?
        arrive = self.thrown & ~self.resolved & (clock >= self.arrive_time - 1e-6)
        picked = np.zeros(self.n, bool)
        catch = np.zeros(self.n, bool)
        incomplete = np.zeros(self.n, bool)
        if arrive.any():
            rows = np.nonzero(arrive)[0]
            back = (clock[rows] - self.arrive_time[rows])[:, None]
            wr_at = self.pos[rows, WR] - self.vel[rows, WR] * back
            cb_at = self.pos[rows, CB] - self.vel[rows, CB] * back
            tgt = self.target[rows]
            d_wr = np.linalg.norm(wr_at - tgt, axis=-1)
            d_cb = np.linalg.norm(cb_at - tgt, axis=-1)
            sep = np.linalg.norm(cb_at - wr_at, axis=-1)
            inb = (tgt[:, 1] >= 0.5) & (tgt[:, 1] <= f.width - 0.5) & (tgt[:, 0] <= f.length + END_ZONE - 0.5)
            p_int = c.int_max / (1.0 + np.exp(-(c.int_at - d_cb) / c.int_width)) * inb
            p_catch = (c.catch_base + c.catch_gain * (1.0 - np.exp(-sep / c.catch_scale))) * ((d_wr <= c.catch_radius) & inb)
            u = self.rng.random(rows.size)
            pk = u < p_int
            ct = ~pk & (u < p_int + (1.0 - p_int) * p_catch)
            picked[rows], catch[rows] = pk, ct
            incomplete[rows] = ~pk & ~ct
            self.resolved[rows] = True
            self.caught[rows[ct]] = True
            self.cooldown[rows[ct]] = 0.0

        # Tackles only once the receiver has the ball.
        self.cooldown[~self.caught] = np.maximum(self.cooldown[~self.caught], 1.0)
        attempt, made, missed, wrapper = self._tackles(pre_vel)
        self.wrap[made] = tc.wrap_time
        self.wrapper[made] = wrapper[made]
        self.stun[:, self.dfn] = np.where(missed, tc.miss_stun, self.stun[:, self.dfn])
        self.cooldown = np.where(missed, tc.miss_cooldown, self.cooldown)
        self.vel[:, WR] *= (tc.miss_slowdown ** missed.sum(1))[:, None]
        self.stun = np.maximum(0.0, self.stun - dt)
        self.cooldown = np.maximum(0.0, self.cooldown - dt)
        self.wrap[wrapped] -= dt
        tackled_now = wrapped & (self.wrap <= 1e-6)

        self.pos[..., 0] = np.clip(self.pos[..., 0], 0.0, f.length + END_ZONE)
        self.pos[:, 1:, 1] = np.clip(self.pos[:, 1:, 1], 0.0, f.width)
        self.pos[~self.caught, WR, 1] = np.clip(self.pos[~self.caught, WR, 1], 0.0, f.width)   # can't leave before the catch
        self._push_history()

        # The ball's yards past the line once caught (air yards, then after the catch), capped at the goal line.
        self.yards[self.caught] = np.minimum(self.pos[self.caught, WR, 0], f.length) - c.los
        cy = self.pos[:, WR, 1]
        touchdown = self.caught & (self.pos[:, WR, 0] >= f.length)
        out_of_bounds = self.caught & ((cy <= 0.0) | (cy >= f.width)) & ~touchdown
        sack = ~self.thrown & (clock >= c.throw_by - 1e-6)
        tackled = tackled_now & ~touchdown & ~out_of_bounds
        over = touchdown | out_of_bounds | tackled | incomplete | picked | sack
        timeout = (self.t >= f.max_steps) & ~over
        done = over | timeout
        self.yards[sack] = -c.sack_yards
        self.yards[picked] = -c.int_yards
        self.yards[incomplete] = 0.0

        outcome = np.zeros(self.n, np.int8)
        outcome[touchdown] = OUTCOME_TOUCHDOWN
        outcome[tackled] = OUTCOME_TACKLED
        outcome[out_of_bounds] = OUTCOME_OUT_OF_BOUNDS
        outcome[timeout] = OUTCOME_TIMEOUT
        outcome[incomplete] = OUTCOME_INCOMPLETE
        outcome[picked] = OUTCOME_INTERCEPTION
        outcome[sack] = OUTCOME_SACK
        gained = self.yards - yards_before
        base = (rw.yards_scale * gained + rw.touchdown_bonus * touchdown - rw.tackle_penalty * tackled
                - rw.time_penalty).astype(np.float32)
        info = {
            "outcome": outcome, "yards": self.yards.copy(), "steps": self.t.copy(),
            "tackler_reward": (-base).astype(np.float32),
            "tackle_attempt": attempt.any(axis=1), "tackle_made": made, "tackle_missed": missed.any(axis=1),
            "attempts": attempt.sum(1), "misses": missed.sum(1),
            "tackle_by_def": attempt, "tackle_missed_by_def": missed, "tackle_p": self.last_tackle_p,
            "tackle_engaged": self.last_tackle_engaged, "wrapped": self.wrap > 0, "breakaway": np.zeros(self.n, bool),
            "thrown_now": throw_now, "caught_now": catch, "arrived": arrive,
            "pos": self.pos.copy(), "facing": self.facing.copy(), "partner": self.partner.copy(),
            "r_pos": self.pos[:, WR].copy(), "ball": self.ball_pos(),
            "terminal_obs": self.observe() if done.any() else None,
            "terminal_obs_tackler": self.observe_tackler() if done.any() else None,
        }
        if done.any():
            self.reset(done)
        return self.observe(), base, done, info

    def _pin_qb(self) -> None:
        self.pos[:, QB] = self.qb_spot
        self.vel[:, QB] = 0.0

    # ------------------------------------------------------------ observation
    def _obs(self, team: str) -> np.ndarray:
        """Globals (11): receiver x, y, to sideline, wrapped, ball thrown, caught, time until the
        ball arrives, ball target relative to the receiver (x, y), play clock / throw_by, which
        way is inside. Then per player as in TeamSim, own team first."""
        c, f, b = self.cfg, self.cfg.field, self.bodies
        pos, vel, face = self.perceived(team)
        order = np.r_[self.off, self.dfn] if team == "offense" else np.r_[self.dfn, self.off]
        ref = pos[:, WR]
        clock = self.t * f.dt
        fly = self.thrown & ~self.resolved
        obs = np.zeros((self.n, self.obs_dim), np.float32)
        obs[:, 0] = ref[:, 0] / f.length
        obs[:, 1] = ref[:, 1] / f.width
        obs[:, 2] = np.minimum(ref[:, 1], f.width - ref[:, 1]) / self._half_w
        obs[:, 3] = self.wrap > 0
        obs[:, 4] = self.thrown
        obs[:, 5] = self.caught
        obs[:, 6] = np.where(fly, np.minimum((self.arrive_time - clock) / 2.0, 1.0), 0.0)
        obs[:, 7:9] = np.where(fly[:, None], (self.target - ref) / 20.0, 0.0)
        obs[:, 9] = np.minimum(clock / c.throw_by, 1.5)
        obs[:, 10] = self.wr_inside
        per = obs[:, self.n_global:].reshape(self.n, self.p, PER_PLAYER_OBS)
        per[..., 0:2] = (pos[:, order] - ref[:, None]) / 20.0
        per[..., 2:4] = vel[:, order] / b.max_speed[:, order, None]
        per[..., 4:6] = face[:, order]
        per[..., 7] = self.stun[:, order] / max(self.cfg.tackle.miss_stun, 1e-6)
        return obs

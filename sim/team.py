"""Vectorized team sim: one ball carrier plus blockers vs a group of defenders.

Same physics, tackling and reward as the 1v1 sim (`core.py`), plus:

- Rosters. `offense[0]` carries the ball; the other offensive players block.
  The first setup is an RB screen: RB + 2 OL vs LB + 2 DB (3v3).
- Blocking. A blocker who reaches a free defender latches on (engaged). While
  engaged the defender keeps only part of his push, and the pair is held
  together on a short leash, so a 315 lb lineman anchors a 194 lb DB while a
  linebacker drags him further. Defenders shed blocks at random, faster when
  they are heavier relative to the blocker; a shed defender can't be
  re-blocked right away. Engaged defenders can't tackle, unless
  `BlockConfig.engaged_tackle_logit` is set: then one can reach out for a
  carrier passing on a side his blocker isn't covering (lane drill).
- Bodies. Every pair of players collides (`resolve_collisions`), so a line of
  blockers is a wall the carrier has to find a way around.
- Formations. "screen" (above) or "lane": the Oklahoma drill, OL and DL
  engaged nose to nose at the snap, the RB 5 yd behind them, in a lane a few
  yards wide (`lane_drill_config`).
- Tackles. Any free defender in reach of the carrier attempts; one made
  tackle wraps him up, each miss stuns that defender.
- Breakaway. The carrier must lead *every* defender and be outrunning all.

Interface matches `OneVOneSim` (`observe()` = offense, `observe_tackler()` =
defense, `step(offense_action, defense_action)`), so the vec env, self-play
and evaluation code drive either sim. Each side is one "team" policy that
moves all its players: actions are (N, players * 3) = push x, push y,
square-up per player.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from .core import (OUTCOME_OUT_OF_BOUNDS, OUTCOME_TACKLED, OUTCOME_TIMEOUT, OUTCOME_TOUCHDOWN, FieldConfig,
                   OneVOneSim, RewardConfig, SimConfig, TackleConfig)
from .physics import PROFILES, Bodies, apply_controls, resolve_collisions, turn_facing

CARRIER = 0
PER_PLAYER_OBS = 8       # rel x, rel y, vx, vy, facing x, facing y, engaged, stunned
GLOBAL_OBS = 4           # carrier x, carrier y, carrier to sideline, wrapped


@dataclass
class BlockConfig:
    reach: float = 0.3              # yd beyond body contact a blocker can latch on
    engaged_control: float = 0.35   # share of his push an engaged defender keeps
    leash: float = 0.3              # yd of slack before an engaged pair is pulled together
    shed_rate: float = 0.6          # per second between equal weights (~1.7 s average block)
    shed_mass_power: float = 1.5    # rate x (defender / blocker weight)^power
    reengage_cooldown: float = 1.0  # s before a shed defender can be blocked again
    # None: an engaged defender can't tackle (screen, EXP-009..012). A number:
    # he can, at this logit offset, when the carrier is in reach and not
    # behind his blocker (more than `shield_angle` degrees off the line to him).
    engaged_tackle_logit: float | None = None
    shield_angle: float = 60.0


@dataclass
class Formation:
    """Random starting spots, yards relative to the carrier (x downfield)."""
    kind: str = "screen"            # "screen" or "lane" (Oklahoma drill)
    carrier_y_jitter: float = 5.0
    # Lane: line of scrimmage this far ahead of the carrier; each DL lines up
    # on a blocker's nose, up to `shade` yd to either side; any defender past
    # the blockers' count is an unblocked LB this deep off the line.
    los_ahead: float = 5.0
    shade: float = 0.5
    lb_depth: tuple[float, float] = (4.0, 5.0)
    # Screen timing: the linemen have already released when the RB has the
    # ball. Starting them 1-3 yd ahead, blocking was worth ~nothing (42 vs 42%
    # TD, scripted); 3-5 yd ahead it is worth ~20 points (see EXP-009 notes).
    blocker_ahead: tuple[float, float] = (3.0, 5.0)
    blocker_side: tuple[float, float] = (1.5, 4.0)
    defender_ahead: tuple[float, float] = (7.0, 13.0)
    defender_spread: float = 10.0


@dataclass
class TeamConfig:
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    reward: RewardConfig = dc_field(default_factory=RewardConfig)
    tackle: TackleConfig = dc_field(default_factory=TackleConfig)
    block: BlockConfig = dc_field(default_factory=BlockConfig)
    formation: Formation = dc_field(default_factory=Formation)
    offense: tuple[str, ...] = ("RB", "OL", "OL")
    defense: tuple[str, ...] = ("LB", "DB", "DB")
    facing: bool = True


def lane_drill_config(width: float = 5.0, linebacker: bool = False, past_los: float = 10.0,
                      **kw) -> TeamConfig:
    """Oklahoma drill: RB + OL vs DL (+ an unblocked LB) between two bags
    `width` yd apart. Leaving the lane ends the play; reaching `past_los`
    yd beyond the line wins it (scored as a touchdown).

    `kw` passes field/tackle/reward/block settings; field length, width, start
    and play clock are set here."""
    fm = Formation(kind="lane", carrier_y_jitter=0.3)
    field = kw.pop("field", FieldConfig())
    field = FieldConfig(**{**field.__dict__, "width": width, "runner_start_x": 5.0, "breakaway_lead": None,
                           "length": 5.0 + fm.los_ahead + past_los, "max_steps": 40})
    # A DL is near an OL's weight (285 vs 315 lb), so he keeps more of his push than
    # a DB (0.35, screen): at 0.35 waiting for the block never paid (EXP-013 notes).
    block = kw.pop("block", BlockConfig(engaged_control=0.7, engaged_tackle_logit=-1.5))
    return TeamConfig(field=field, block=block, formation=fm, offense=("RB", "OL"),
                      defense=("DL", "LB") if linebacker else ("DL",), **kw)


SIMS = ("1v1", "3v3", "lane", "lane-lb")


def config_for(sim: str, base: SimConfig, lane_width: float = 5.0):
    """The config a CLI `--sim` name means, built on the shared 1v1-style settings."""
    if sim == "1v1":
        return base
    if sim == "3v3":
        return TeamConfig(field=base.field, tackle=base.tackle, reward=base.reward, facing=base.facing)
    return lane_drill_config(width=lane_width, linebacker=sim == "lane-lb", field=base.field,
                             tackle=base.tackle, reward=base.reward, facing=base.facing)


def make_sim(n: int, cfg, seed: int = 0):
    """The sim that matches a config: TeamConfig -> TeamSim, SimConfig -> OneVOneSim."""
    return TeamSim(n, cfg, seed=seed) if isinstance(cfg, TeamConfig) else OneVOneSim(n, cfg, seed=seed)


class TeamSim:
    """N parallel plays of len(offense) vs len(defense). Auto-resets finished plays."""

    def __init__(self, n: int, cfg: TeamConfig | None = None, seed: int = 0):
        self.n = n
        self.cfg = cfg or TeamConfig()
        self.rng = np.random.default_rng(seed)
        self.n_off, self.n_def = len(self.cfg.offense), len(self.cfg.defense)
        self.p = self.n_off + self.n_def
        self.off = np.arange(self.n_off)
        self.dfn = np.arange(self.n_off, self.p)
        self.blk = np.arange(1, self.n_off)
        self.bodies = Bodies(n, [PROFILES[r] for r in (*self.cfg.offense, *self.cfg.defense)])
        self.roles = (*self.cfg.offense, *self.cfg.defense)

        p = self.p
        self.pos = np.zeros((n, p, 2), np.float32)
        self.vel = np.zeros((n, p, 2), np.float32)
        self.facing = np.zeros((n, p, 2), np.float32)
        self.stun = np.zeros((n, p), np.float32)
        self.cooldown = np.zeros((n, self.n_def), np.float32)      # tackle re-try
        self.shed_cd = np.zeros((n, self.n_def), np.float32)       # can't be re-blocked yet
        self.partner = np.full((n, p), -1, np.int64)               # engaged with (player index) or -1
        self.wrap = np.zeros(n, np.float32)
        self.wrapper = np.full(n, len(self.cfg.offense), np.int64)  # defender index doing the wrap
        self.t = np.zeros(n, np.int32)
        self.yards = np.zeros(n, np.float32)
        self.los = np.full(n, self.cfg.field.runner_start_x, np.float32)   # lane drill line of scrimmage
        self.full_run = np.ones(n, bool)
        self._half_w = self.cfg.field.width / 2.0
        self.obs_dim = GLOBAL_OBS + PER_PLAYER_OBS * p
        d = self.cfg.field.reaction_delay
        self._hist_pos = np.zeros((d + 1, n, p, 2), np.float32) if d else None
        self._hist_vel = np.zeros((d + 1, n, p, 2), np.float32) if d else None
        self._hist_face = np.zeros((d + 1, n, p, 2), np.float32) if d else None

    def act_dim_for(self, side: str) -> int:
        return 3 * (self.n_off if side == "runner" else self.n_def)

    # ------------------------------------------------------------- perception
    def perceived(self, team: str):
        """(pos, vel, facing) of every player as `team` ("offense"/"defense")
        sees them: own players now, opponents `reaction_delay` steps ago."""
        if self._hist_pos is None:
            return self.pos, self.vel, self.facing
        opp = self.dfn if team == "offense" else self.off
        pos, vel, face = self.pos.copy(), self.vel.copy(), self.facing.copy()
        pos[:, opp], vel[:, opp], face[:, opp] = (self._hist_pos[0][:, opp], self._hist_vel[0][:, opp],
                                                  self._hist_face[0][:, opp])
        return pos, vel, face

    def _push_history(self, mask: np.ndarray | None = None) -> None:
        if self._hist_pos is None:
            return
        if mask is None:
            for h, cur in ((self._hist_pos, self.pos), (self._hist_vel, self.vel), (self._hist_face, self.facing)):
                h[:-1] = h[1:]
                h[-1] = cur
        else:
            for h, cur in ((self._hist_pos, self.pos), (self._hist_vel, self.vel), (self._hist_face, self.facing)):
                h[:, mask] = cur[mask]

    @property
    def engaged(self) -> np.ndarray:
        return self.partner >= 0

    # ------------------------------------------------------------------ reset
    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = np.ones(self.n, bool)
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        f, fm, rng = self.cfg.field, self.cfg.formation, self.rng
        cx = np.full(k, f.runner_start_x)
        cy = self._half_w + rng.uniform(-fm.carrier_y_jitter, fm.carrier_y_jitter, k)
        pos = np.zeros((k, self.p, 2), np.float32)
        pos[:, CARRIER] = np.stack([cx, cy], 1)
        if fm.kind == "lane":
            return self._reset_lane(mask, pos, cx)
        for j, b in enumerate(self.blk):
            side = 1.0 if j % 2 == 0 else -1.0           # alternate sides of the carrier
            pos[:, b, 0] = cx + rng.uniform(*fm.blocker_ahead, k)
            pos[:, b, 1] = cy + side * rng.uniform(*fm.blocker_side, k)
        # Defenders spread across the carrier's lane, at least 2 yd apart in y.
        lanes = np.sort(rng.uniform(-fm.defender_spread, fm.defender_spread, (k, self.n_def)), axis=1)
        lanes += np.arange(self.n_def) * 2.0 - (self.n_def - 1)
        for j, d in enumerate(self.dfn):
            pos[:, d, 0] = cx + rng.uniform(*fm.defender_ahead, k)
            pos[:, d, 1] = cy + lanes[:, j]
        pos[..., 1] = np.clip(pos[..., 1], 1.0, f.width - 1.0)
        return self._place(mask, pos)

    def _reset_lane(self, mask: np.ndarray, pos: np.ndarray, cx: np.ndarray) -> np.ndarray:
        """Oklahoma drill: blockers side by side on the line (1 yd apart), a DL
        engaged nose to nose with each, the rest unblocked LBs off the line."""
        fm, rng, b = self.cfg.formation, self.rng, self.bodies
        k = len(cx)
        los = cx + fm.los_ahead
        n_dl = min(len(self.blk), self.n_def)
        for j, blk in enumerate(self.blk):
            pos[:, blk, 0] = los - b.radius[0, blk]
            pos[:, blk, 1] = self._half_w + (j - (len(self.blk) - 1) / 2)
        for j, d in enumerate(self.dfn):
            if j < n_dl:
                pos[:, d, 0] = los + b.radius[0, d]
                pos[:, d, 1] = pos[:, self.blk[j], 1] + rng.uniform(-fm.shade, fm.shade, k)
            else:
                pos[:, d, 0] = los + rng.uniform(*fm.lb_depth, k)
                pos[:, d, 1] = self._half_w + rng.uniform(-1.0, 1.0, k)
        self._place(mask, pos)
        rows = np.nonzero(mask)[0]
        for j in range(n_dl):
            self.partner[rows, self.blk[j]] = self.dfn[j]
            self.partner[rows, self.dfn[j]] = self.blk[j]
        self.los[mask] = los
        self.yards[mask] = cx - los       # yards count from the line of scrimmage
        return self.observe()

    def _place(self, mask: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Write starting spots for the masked plays and clear their state."""
        f, rng = self.cfg.field, self.rng
        k = len(pos)
        self.pos[mask] = pos
        self.vel[mask] = 0
        face = np.zeros((k, self.p, 2), np.float32)
        face[:, : self.n_off, 0] = 1.0                    # offense faces downfield
        face[:, self.n_off:, 0] = -1.0                    # defense faces the offense
        self.facing[mask] = face
        self.stun[mask] = 0
        self.cooldown[mask] = 0
        self.shed_cd[mask] = 0
        self.partner[mask] = -1
        self.wrap[mask] = 0
        self.t[mask] = 0
        self.yards[mask] = 0
        if f.breakaway_lead is not None:
            self.full_run[mask] = rng.random(k) < f.full_play_share
        self._push_history(mask)
        return self.observe()

    # ------------------------------------------------------------------ parts
    def _actions(self, a: np.ndarray, players: int) -> np.ndarray:
        """(N, players*k) or (N, players, k) with k = 2 or 3 -> (N, players, 3)."""
        a = np.asarray(a, np.float32).reshape(self.n, players, -1)
        if a.shape[2] >= 3:
            return a[:, :, :3]
        return np.concatenate([a, np.zeros((self.n, players, 1), np.float32)], axis=2)

    def _turn_hips(self, push: np.ndarray, square: np.ndarray) -> None:
        """Square-up target: the carrier for defenders, the nearest defender for offense."""
        p_norm = np.linalg.norm(push, axis=-1, keepdims=True)
        push_dir = push / np.maximum(p_norm, 1e-6) * (p_norm > 0.05)
        o_pos, _, _ = self.perceived("offense")
        d_pos, _, _ = self.perceived("defense")
        focus = np.empty_like(self.pos)
        focus[:, self.dfn] = d_pos[:, CARRIER, None]
        rel = o_pos[:, None, self.dfn] - self.pos[:, self.off, None]        # (N, off, def, 2)
        near = np.argmin(np.linalg.norm(rel, axis=-1), axis=2)              # (N, off)
        focus[:, self.off] = np.take_along_axis(o_pos[:, self.dfn], near[..., None], axis=1)
        to_f = focus - self.pos
        to_f /= np.maximum(np.linalg.norm(to_f, axis=-1, keepdims=True), 1e-6)
        target = (1.0 - square[..., None]) * push_dir + square[..., None] * to_f
        speed = np.linalg.norm(self.vel, axis=-1)
        turn_facing(self.facing, target, speed, self.bodies, self.cfg.field.dt, stunned=self.stun > 0)

    def _leash(self) -> None:
        """Hold engaged pairs together: pull back by inverse mass, kill separating speed."""
        b = self.bodies
        rows, blks = np.nonzero(self.partner[:, self.blk] >= 0)
        if rows.size == 0:
            return
        blks = self.blk[blks]
        dfs = self.partner[rows, blks]
        d = self.pos[rows, dfs] - self.pos[rows, blks]
        dist = np.linalg.norm(d, axis=1)
        nrm = d / np.maximum(dist, 1e-6)[:, None]
        limit = b.radius[rows, blks] + b.radius[rows, dfs] + self.cfg.block.leash
        over = np.maximum(0.0, dist - limit)
        wb, wd = 1.0 / b.mass_lb[rows, blks], 1.0 / b.mass_lb[rows, dfs]
        w = wb + wd
        self.pos[rows, blks] += nrm * (over * wb / w)[:, None]
        self.pos[rows, dfs] -= nrm * (over * wd / w)[:, None]
        v_sep = np.sum((self.vel[rows, dfs] - self.vel[rows, blks]) * nrm, axis=1)
        j = np.where((over > 0) & (v_sep > 0), v_sep / w, 0.0)
        self.vel[rows, blks] += nrm * (j * wb)[:, None]
        self.vel[rows, dfs] -= nrm * (j * wd)[:, None]

    def _blocking(self) -> tuple[np.ndarray, np.ndarray]:
        """Shed existing blocks, then start new ones. Returns (new_engaged, shed) counts per play."""
        bc, b, dt = self.cfg.block, self.bodies, self.cfg.field.dt
        shed = np.zeros(self.n, np.int32)
        new = np.zeros(self.n, np.int32)
        for blk in self.blk:
            dfs = self.partner[:, blk]
            on = dfs >= 0
            if on.any():
                rows = np.nonzero(on)[0]
                ratio = b.mass_lb[rows, dfs[rows]] / b.mass_lb[rows, blk]
                rate = bc.shed_rate * ratio ** bc.shed_mass_power
                go = self.rng.random(rows.size) < 1.0 - np.exp(-rate * dt)
                r, d = rows[go], dfs[rows[go]]
                self.partner[r, blk] = -1
                self.partner[r, d] = -1
                self.shed_cd[r, d - self.n_off] = bc.reengage_cooldown
                shed[r] += 1
        free_def = (self.partner[:, self.dfn] < 0) & (self.shed_cd <= 0)
        if self.wrap.any():   # the defender wrapping the carrier can't be blocked
            free_def[np.arange(self.n), self.wrapper - self.n_off] &= self.wrap <= 0
        for blk in self.blk:
            can = (self.partner[:, blk] < 0) & (self.stun[:, blk] <= 0)
            d = self.pos[:, self.dfn] - self.pos[:, blk, None]
            dist = np.linalg.norm(d, axis=-1)
            reach = b.radius[:, blk, None] + b.radius[:, self.dfn] + bc.reach
            ok = (dist < reach) & free_def & can[:, None]
            if not ok.any():
                continue
            pick = np.argmin(np.where(ok, dist, np.inf), axis=1)
            rows = np.nonzero(ok.any(axis=1))[0]
            di = pick[rows]
            self.partner[rows, blk] = self.n_off + di
            self.partner[rows, self.n_off + di] = blk
            free_def[rows, di] = False
            new[rows] += 1
        return new, shed

    def _tackles(self, pre_vel: np.ndarray):
        """Every free defender in reach tries. Returns (attempts, made_any, missed_count, wrapper)."""
        tc, b, bc = self.cfg.tackle, self.bodies, self.cfg.block
        d = self.pos[:, CARRIER, None] - self.pos[:, self.dfn]            # defender -> carrier
        dist = np.linalg.norm(d, axis=-1)
        reach = b.radius[:, CARRIER, None] + b.radius[:, self.dfn] + tc.reach
        engaged = self.partner[:, self.dfn] >= 0
        attempt = ((dist < reach) & (self.wrap[:, None] <= 0) & (self.cooldown <= 0)
                   & (self.stun[:, self.dfn] <= 0))
        if bc.engaged_tackle_logit is None:
            attempt &= ~engaged
        else:
            # Shielded: the carrier is within shield_angle of the line to his blocker.
            idx = np.maximum(self.partner[:, self.dfn], 0)
            blk = self.pos[np.arange(self.n)[:, None], idx] - self.pos[:, self.dfn]
            cos = np.sum(d * blk, -1) / np.maximum(dist * np.linalg.norm(blk, axis=-1), 1e-6)
            attempt &= ~engaged | (cos < np.cos(np.radians(bc.shield_angle)))
        made = np.zeros_like(attempt)
        if attempt.any():
            u = d / np.maximum(dist, 1e-6)[..., None]
            vr, vt = pre_vel[:, CARRIER, None], pre_vel[:, self.dfn]
            r_speed = np.linalg.norm(vr, axis=-1)
            headon = np.clip(np.sum(vr * -u, axis=-1) / np.maximum(r_speed, 1.0), -1, 1)
            p_r = b.mass_lb[:, CARRIER, None] * r_speed
            p_t = b.mass_lb[:, self.dfn] * np.maximum(0.0, np.sum(vt * u, axis=-1))
            ratio = p_r / (p_r + p_t + 1e-6)
            logit = tc.base_logit + tc.headon_weight * headon - tc.momentum_weight * (ratio - 0.5)
            if tc.overspeed_weight:
                logit -= tc.overspeed_weight * np.maximum(0.0, np.linalg.norm(vt, axis=-1) - tc.control_speed)
            if tc.sidestep_weight:
                logit -= tc.sidestep_weight * np.abs(vr[..., 0] * u[..., 1] - vr[..., 1] * u[..., 0])
            if bc.engaged_tackle_logit is not None:
                logit += bc.engaged_tackle_logit * engaged
            made = attempt & (self.rng.random(attempt.shape) < 1.0 / (1.0 + np.exp(-logit)))
        missed = attempt & ~made
        made_any = made.any(axis=1)
        wrapper = self.n_off + np.argmax(made, axis=1)
        return attempt, made_any, missed, wrapper

    def blocker_potential(self) -> np.ndarray:
        """Phi(s) <= 0 per play: minus the yards each free blocker would have to
        move to stand on the line between some free defender and the carrier
        (the defender he could best cut off). Engaged blockers score 0, and so
        does a play with no free defender left. Uses true positions."""
        c = self.pos[:, CARRIER, None]                         # (N, 1, 2)
        d = self.pos[:, self.dfn]                              # (N, D, 2)
        seg = c - d
        free_d = (self.partner[:, self.dfn] < 0) & (self.stun[:, self.dfn] <= 0)
        phi = np.zeros(self.n, np.float32)
        for b in self.blk:
            p = self.pos[:, b, None]                           # (N, 1, 2)
            t = np.clip(np.sum((p - d) * seg, -1) / np.maximum(np.sum(seg * seg, -1), 1e-6), 0.0, 1.0)
            dist = np.linalg.norm(p - (d + seg * t[..., None]), axis=-1)
            best = np.where(free_d, dist, np.inf).min(axis=1)
            miss = np.where(np.isfinite(best) & (self.partner[:, b] < 0), best, 0.0)
            phi -= miss
        return phi

    # ------------------------------------------------------------------- step
    def step(self, offense_action: np.ndarray, defense_action: np.ndarray):
        f, rw, tc = self.cfg.field, self.cfg.reward, self.cfg.tackle
        dt = f.dt
        prev_x = self.pos[:, CARRIER, 0].copy()
        shaping = rw.blocker_shaping
        phi_before = self.blocker_potential() if shaping else None
        wrapped = self.wrap > 0

        action = np.concatenate([self._actions(offense_action, self.n_off),
                                 self._actions(defense_action, self.n_def)], axis=1)
        push = action[..., :2].copy()
        if self.cfg.facing:
            self._turn_hips(push, np.clip(action[..., 2], 0.0, 1.0))
        # An engaged defender is fighting the block: only part of his push counts.
        eng_def = self.partner[:, self.dfn] >= 0
        n_ = np.linalg.norm(push, axis=-1, keepdims=True)
        push = np.where(n_ > 1.0, push / np.maximum(n_, 1e-6), push)
        push[:, self.dfn] *= np.where(eng_def, self.cfg.block.engaged_control, 1.0)[..., None]
        apply_controls(self.pos, self.vel, push, self.bodies, dt, stunned=self.stun > 0,
                       facing=self.facing if self.cfg.facing else None)

        if wrapped.any():   # carrier and the wrapping defender slide together
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

        self._leash()
        pre_vel = self.vel.copy()
        resolve_collisions(self.pos, self.vel, self.bodies)
        new_blocks, sheds = self._blocking()
        attempt, made, missed, wrapper = self._tackles(pre_vel)
        self.wrap[made] = tc.wrap_time
        self.wrapper[made] = wrapper[made]
        if made.any():      # a defender tackling out of a block leaves the block
            rows = np.nonzero(made)[0]
            blk = self.partner[rows, wrapper[rows]]
            rows, blk = rows[blk >= 0], blk[blk >= 0]
            self.partner[rows, blk] = -1
            self.partner[rows, wrapper[rows]] = -1
        self.stun[:, self.dfn] = np.where(missed, tc.miss_stun, self.stun[:, self.dfn])
        self.cooldown = np.where(missed, tc.miss_cooldown, self.cooldown)
        self.vel[:, CARRIER] *= (tc.miss_slowdown ** missed.sum(1))[:, None]

        self.t += 1
        self.stun = np.maximum(0.0, self.stun - dt)
        self.cooldown = np.maximum(0.0, self.cooldown - dt)
        self.shed_cd = np.maximum(0.0, self.shed_cd - dt)
        self.wrap[wrapped] -= dt
        tackled_now = wrapped & (self.wrap <= 1e-6)

        self.pos[..., 0] = np.clip(self.pos[..., 0], 0.0, f.length)
        others = np.arange(1, self.p)
        self.pos[:, others, 1] = np.clip(self.pos[:, others, 1], 0.0, f.width)
        self._push_history()

        gained = self.pos[:, CARRIER, 0] - prev_x
        self.yards += gained
        cy = self.pos[:, CARRIER, 1]
        touchdown = self.pos[:, CARRIER, 0] >= f.length
        out_of_bounds = ((cy < 0.0) | (cy > f.width)) & ~touchdown
        breakaway = np.zeros(self.n, bool)
        if f.breakaway_lead is not None:
            lead = self.pos[:, CARRIER, 0] - self.pos[:, self.dfn, 0].max(axis=1)
            r_vx = self.vel[:, CARRIER, 0]
            outrunning = (r_vx >= f.breakaway_min_speed) & (r_vx >= self.vel[:, self.dfn, 0].max(axis=1))
            breakaway = ((lead >= f.breakaway_lead) & outrunning & ~self.full_run & (self.wrap <= 0)
                         & ~touchdown & ~out_of_bounds)
            if breakaway.any():
                rest = f.length - self.pos[breakaway, CARRIER, 0]
                gained[breakaway] += rest
                self.yards[breakaway] += rest
                touchdown = touchdown | breakaway
        tackled = tackled_now & ~touchdown & ~out_of_bounds
        timeout = (self.t >= f.max_steps) & ~touchdown & ~out_of_bounds & ~tackled
        done = touchdown | tackled | out_of_bounds | timeout

        outcome = np.zeros(self.n, np.int8)
        outcome[touchdown] = OUTCOME_TOUCHDOWN
        outcome[tackled] = OUTCOME_TACKLED
        outcome[out_of_bounds] = OUTCOME_OUT_OF_BOUNDS
        outcome[timeout] = OUTCOME_TIMEOUT
        base = (rw.yards_scale * gained + rw.touchdown_bonus * touchdown - rw.tackle_penalty * tackled
                  - rw.out_of_bounds_penalty * out_of_bounds - rw.time_penalty).astype(np.float32)
        reward = base
        if shaping:
            # F = gamma * Phi(s') - Phi(s), with Phi = 0 once the play is really over.
            # A timeout is a cut-off, not an end: the learner bootstraps the value of
            # s' there, so Phi(s') must stay. Zeroing it paid an "end bonus" on top of
            # the bootstrapped value, and EXP-010's offense learned to send its blockers
            # far away and run out the clock (71% timeouts) to collect it.
            phi_after = np.where(done & ~timeout, 0.0, self.blocker_potential())
            reward = (base + shaping * (rw.shaping_gamma * phi_after - phi_before)).astype(np.float32)

        info = {
            "outcome": outcome, "yards": self.yards.copy(), "steps": self.t.copy(),
            "tackler_reward": -base,
            "tackle_attempt": attempt.any(axis=1), "tackle_made": made, "tackle_missed": missed.any(axis=1),
            "attempts": attempt.sum(1), "misses": missed.sum(1),
            "wrapped": self.wrap > 0, "breakaway": breakaway,
            "blocks_started": new_blocks, "blocks_shed": sheds,
            "pos": self.pos.copy(), "facing": self.facing.copy(), "partner": self.partner.copy(),
            "r_pos": self.pos[:, CARRIER].copy(),
            "terminal_obs": self.observe() if done.any() else None,
            "terminal_obs_tackler": self.observe_tackler() if done.any() else None,
        }
        if done.any():
            self.reset(done)
        return self.observe(), reward, done, info

    # ------------------------------------------------------------ observation
    def _obs(self, team: str) -> np.ndarray:
        """Team view: own players first (carrier/first defender first), then opponents.
        Positions relative to the carrier as this team perceives him."""
        f, b = self.cfg.field, self.bodies
        pos, vel, face = self.perceived(team)
        order = np.r_[self.off, self.dfn] if team == "offense" else np.r_[self.dfn, self.off]
        ref = pos[:, CARRIER]
        obs = np.empty((self.n, self.obs_dim), np.float32)
        obs[:, 0] = ref[:, 0] / f.length
        obs[:, 1] = ref[:, 1] / f.width
        obs[:, 2] = np.minimum(ref[:, 1], f.width - ref[:, 1]) / self._half_w
        obs[:, 3] = (self.wrap > 0).astype(np.float32)
        per = obs[:, GLOBAL_OBS:].reshape(self.n, self.p, PER_PLAYER_OBS)
        per[..., 0:2] = (pos[:, order] - ref[:, None]) / 20.0
        per[..., 2:4] = vel[:, order] / b.max_speed[:, order, None]
        per[..., 4:6] = face[:, order]
        per[..., 6] = (self.partner[:, order] >= 0).astype(np.float32)
        per[..., 7] = self.stun[:, order] / max(self.cfg.tackle.miss_stun, 1e-6)
        return obs

    def observe(self) -> np.ndarray:
        return self._obs("offense")

    def observe_tackler(self) -> np.ndarray:
        return self._obs("defense")

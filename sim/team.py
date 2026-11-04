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
  yards wide (`lane_drill_config`). Its `combo` variant has two OL, a DL over
  the gap between them and an LB: double team, then one OL climbs.
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
PER_PLAYER_OBS = 8       # rel x, rel y, vx, vy, facing x, facing y, blockers on him (0-2), stunned
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
    # Combo blocks (EXP-016; both off before). `double_team`: a second blocker
    # can latch onto a defender who is already blocked; each blocker on him
    # takes another `engaged_control` share off his push, and he sheds by
    # their combined weight. `release_below`: a blocker whose square-up action
    # drops under this lets go of his man (turns his hips upfield to climb to
    # the next level), and can't latch on again for `release_cooldown` s.
    double_team: bool = False
    release_below: float | None = None
    release_cooldown: float = 0.5
    # Holding the line (EXP-018; all off before). `blocker_control`: share of his
    # push an engaged blocker keeps, so a one-on-one block is a stalemate and it
    # takes a double team to move a DL. `engaged_speed`: top speed (yd/s) of an
    # engaged player. `shed_leverage`: shed rate x exp(this * cos), cos of the
    # angle at the blocker between his man and the carrier: -1 with the blocker
    # between them, +1 once the defender is on the carrier's side.
    # `release_double_only`: only one of a double team can let go, not a lone
    # blocker (at 0.2 square-up EXP-017's s0 dropped 54% of its blocks that way,
    # median 0.2 s after the snap).
    blocker_control: float = 1.0
    engaged_speed: float | None = None
    shed_leverage: float = 0.0
    release_double_only: bool = False


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
    # Lane: how many defenders are DL (None = one per blocker). With fewer DL
    # than blockers, DL j lines up over the gap between blockers j and j+1,
    # engaged with the nearer one.
    dl_count: int | None = None
    # Box (6v6): blockers `ol_spacing` yd apart (centre to centre); with
    # `dl_align` each DL lines up at that y offset from the middle of the line
    # (+/- `shade`), engaged at the snap with the nearest blocker nobody else
    # has; `lb_align` gives each LB's y offset (+/- 1 yd).
    ol_spacing: float = 1.0
    dl_align: tuple[float, ...] | None = None
    lb_align: tuple[float, ...] | None = None
    # Mixed fronts (EXP-019): each play draws one of `fronts` (dl_align tuples)
    # and slides both LBs by up to `lb_shift` yd, so which side is open changes
    # snap to snap. With one fixed front the runner took one path (EXP-018),
    # even with both LBs standing on it.
    fronts: tuple[tuple[float, ...], ...] | None = None
    lb_shift: float = 0.0
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
    # Balance (EXP-015; 0 = off). A collision that changes a free player's
    # velocity by more than `stumble_free` x what his own brakes could in one
    # step knocks him off balance: `stumble` s of no control per yd/s beyond
    # that, up to `stumble_max`. Without it, running into a teammate's back was
    # a free ~40 yd/s^2 brake (EXP-014). Engaged and wrapped players are exempt.
    stumble: float = 0.0
    stumble_free: float = 2.0
    stumble_max: float = 0.5
    # The carrier squares up only to defenders ahead of him; with none ahead his
    # square-up faces downfield (EXP-015). Off: nearest defender anywhere, so a
    # runner who broke free kept his hips on the chaser behind him (EXP-013).
    carrier_square_ahead: bool = False
    # The play clock is the drill's real end (EXP-016+), not a cut-off: no
    # bootstrapping past it. As a cut-off, a combo-drill runner learned to
    # hover 2 yd short of the end line until time ran out (critic credit for
    # the score he "would" get) while the DL camped on that line.
    timeout_ends: bool = False
    # Seconds the snap takes to reach the carrier (EXP-018; 0 = he has the ball at
    # the start). Until then he stands still, and everyone else is already playing.
    snap_time: float = 0.0


def lane_drill_config(width: float = 5.0, linebacker: bool = False, past_los: float = 10.0,
                      combo: bool = False, **kw) -> TeamConfig:
    """Oklahoma drill: RB + OL vs DL (+ an unblocked LB) between two bags
    `width` yd apart. Leaving the lane ends the play; reaching `past_los`
    yd beyond the line wins it (scored as a touchdown).

    `combo`: the inside-zone combo block instead, RB + 2 OL vs a DL over the
    gap between them + an LB (double team, then one OL climbs to the LB).

    `kw` passes field/tackle/reward/block settings; field length, width, start
    and play clock are set here."""
    fm = Formation(kind="lane", carrier_y_jitter=0.3, dl_count=1 if combo else None, shade=0.7 if combo else 0.5)
    field = kw.pop("field", FieldConfig())
    field = FieldConfig(**{**field.__dict__, "width": width, "runner_start_x": 5.0, "breakaway_lead": None,
                           "length": 5.0 + fm.los_ahead + past_los, "max_steps": 40})
    # A DL is near an OL's weight (285 vs 315 lb), so he keeps more of his push than
    # a DB (0.35, screen): at 0.35 waiting for the block never paid (EXP-013 notes).
    block = kw.pop("block", BlockConfig(engaged_control=0.7, engaged_tackle_logit=-1.5,
                                        double_team=combo, release_below=0.2 if combo else None))
    return TeamConfig(field=field, block=block, formation=fm, offense=("RB", "OL", "OL") if combo else ("RB", "OL"),
                      defense=("DL", "LB") if (linebacker or combo) else ("DL",), **kw)


FRONTS = ((-3.2, -0.6, 1.9, 3.2),        # 1-technique left, 3-technique right
          (-3.2, -1.9, 0.6, 3.2),        # the same, mirrored
          (-3.2, -1.0, 1.0, 3.2),        # even
          (-2.4, -0.6, 0.6, 2.4))        # tight, ends head up on the tackles


def box_config(width: float = 20.0, past_los: float = 10.0, snap: bool = False, mix: bool = False, **kw) -> TeamConfig:
    """Inside run, 6v6: RB + 5 OL vs a 4-3 front's 4 DL + 2 LB. Five blockers
    for six defenders, so one defender is always unblocked at the snap, as in
    a real run against a six-man box.

    OL 1.3 yd apart (LT LG C RG RT at -2.6 ... +2.6). DL: left end outside the
    LT (5-technique), a 1-technique shading the centre, a 3-technique outside
    the RG, right end outside the RT; each starts engaged with the nearest OL,
    leaving one OL free to double team or climb. LBs at +/-1.5 yd, 4-5 yd deep.
    Reaching `past_los` yd beyond the line ends the play as a win (no
    secondary is modelled). `kw` as in `lane_drill_config`.

    `snap` (EXP-018): the centre snaps the ball to an RB 7 yd deep, 0.6 s in the
    air, with blocks that hold (see BlockConfig). NFL backs take about 2.5-3.1 s
    to the line (Next Gen Stats); without the snap ours took 1.5 s.

    `mix` (EXP-019): a different front and LB slide each play (see Formation)."""
    snap = snap or mix
    fm = Formation(kind="lane", carrier_y_jitter=0.3, ol_spacing=1.3, shade=0.3, los_ahead=7.0 if snap else 5.0,
                   dl_align=(-3.2, -0.6, 1.9, 3.2), lb_align=(-1.5, 1.5),
                   fronts=FRONTS if mix else None, lb_shift=3.0 if mix else 0.0)
    field = kw.pop("field", FieldConfig())
    field = FieldConfig(**{**field.__dict__, "width": width, "runner_start_x": 5.0, "breakaway_lead": None,
                           "length": 5.0 + fm.los_ahead + past_los, "max_steps": 70 if snap else 50})
    hold = dict(blocker_control=0.7, engaged_speed=1.0, shed_rate=0.25, shed_leverage=1.0,
                release_double_only=True) if snap else {}
    block = kw.pop("block", BlockConfig(engaged_control=0.7, engaged_tackle_logit=-1.5,
                                        double_team=True, release_below=0.2, **hold))
    if snap:
        kw.setdefault("snap_time", 0.6)
    return TeamConfig(field=field, block=block, formation=fm, offense=("RB",) + ("OL",) * 5,
                      defense=("DL",) * 4 + ("LB",) * 2, **kw)


SIMS = ("1v1", "3v3", "lane", "lane-lb", "combo", "box", "box-snap", "box-mix")


def config_for(sim: str, base: SimConfig, lane_width: float = 5.0, **team):
    """The config a CLI `--sim` name means, built on the shared 1v1-style settings.
    `team` sets TeamConfig-only fields (stumble, carrier_square_ahead)."""
    if sim == "1v1":
        return base
    if sim == "3v3":
        return TeamConfig(field=base.field, tackle=base.tackle, reward=base.reward, facing=base.facing, **team)
    if sim in ("box", "box-snap", "box-mix"):
        return box_config(snap=sim == "box-snap", mix=sim == "box-mix", field=base.field, tackle=base.tackle, reward=base.reward, facing=base.facing, **team)
    return lane_drill_config(width=lane_width, linebacker=sim == "lane-lb", combo=sim == "combo", field=base.field,
                             tackle=base.tackle, reward=base.reward, facing=base.facing, **team)


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
        # Engaged with (player index) or -1. A blocker points at his man; a
        # double-teamed defender points at one of his blockers.
        self.partner = np.full((n, p), -1, np.int64)
        self.release_cd = np.zeros((n, len(self.blk)), np.float32)  # released blocker can't re-latch yet
        self.wrap = np.zeros(n, np.float32)
        self.wrapper = np.full(n, len(self.cfg.offense), np.int64)  # defender index doing the wrap
        self.t = np.zeros(n, np.int32)
        self.yards = np.zeros(n, np.float32)
        self.los = np.full(n, self.cfg.field.runner_start_x, np.float32)   # lane drill line of scrimmage
        self.full_run = np.ones(n, bool)
        self._half_w = self.cfg.field.width / 2.0
        self.n_global = GLOBAL_OBS + (self.cfg.snap_time > 0)      # + time until the ball arrives
        self.obs_dim = self.n_global + PER_PLAYER_OBS * p
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
        nb = len(self.blk)
        dl_y = None                                  # (k, n_dl) y offset of each DL this play
        if fm.fronts is not None:
            dl_y = np.array(fm.fronts)[rng.integers(len(fm.fronts), size=k)]
        elif fm.dl_align is not None:
            dl_y = np.tile(fm.dl_align, (k, 1))
        if dl_y is not None:
            n_dl = dl_y.shape[1]
        else:
            n_dl = min(nb, self.n_def) if fm.dl_count is None else fm.dl_count
        lb_slide = rng.uniform(-fm.lb_shift, fm.lb_shift, k) if fm.lb_shift else 0.0
        over_gap = n_dl < nb
        for j, blk in enumerate(self.blk):
            pos[:, blk, 0] = los - b.radius[0, blk]
            pos[:, blk, 1] = self._half_w + (j - (nb - 1) / 2) * fm.ol_spacing
        engaged_with = []
        taken = np.zeros((k, nb), bool)
        for j, d in enumerate(self.dfn):
            if j < n_dl:
                pos[:, d, 0] = los + b.radius[0, d]
                if dl_y is not None:
                    pos[:, d, 1] = self._half_w + dl_y[:, j] + rng.uniform(-fm.shade, fm.shade, k)
                    near = None                      # assigned below, ends first
                elif over_gap:
                    gap = (pos[:, self.blk[j], 1] + pos[:, self.blk[j + 1], 1]) / 2
                    pos[:, d, 1] = gap + rng.uniform(-fm.shade, fm.shade, k)
                    near = np.where(pos[:, d, 1] < gap, self.blk[j], self.blk[j + 1])
                else:
                    pos[:, d, 1] = pos[:, self.blk[j], 1] + rng.uniform(-fm.shade, fm.shade, k)
                    near = np.full(k, self.blk[j])
                engaged_with.append(near)
            else:
                off = fm.lb_align[j - n_dl] if fm.lb_align is not None else 0.0
                pos[:, d, 0] = los + rng.uniform(*fm.lb_depth, k)
                pos[:, d, 1] = self._half_w + off + lb_slide + rng.uniform(-1.0, 1.0, k)
        if dl_y is not None:
            # Outside-most DL pick first, so an end never takes an inside OL
            # because the tackle went to a DT shading toward him.
            rows = np.arange(k)
            order = np.argsort(-np.abs(dl_y), axis=1)
            picks = np.zeros((k, n_dl), np.int64)
            for rank in range(n_dl):
                j = order[:, rank]
                gap_y = np.abs(pos[:, self.blk, 1] - pos[rows, self.dfn[j], 1, None])
                pick = np.argmin(np.where(taken, np.inf, gap_y), axis=1)
                taken[rows, pick] = True
                picks[rows, j] = self.blk[pick]
            engaged_with = list(picks.T)
        self._place(mask, pos)
        rows = np.nonzero(mask)[0]
        for j in range(n_dl):
            self.partner[rows, engaged_with[j]] = self.dfn[j]
            self.partner[rows, self.dfn[j]] = engaged_with[j]
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
        self.release_cd[mask] = 0
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
        if self.cfg.carrier_square_ahead:
            me = self.pos[:, CARRIER]
            dist = np.linalg.norm(o_pos[:, self.dfn] - me[:, None], axis=-1)
            ahead = o_pos[:, self.dfn, 0] > me[:, None, 0] - 0.5
            k = np.argmin(np.where(ahead, dist, np.inf), axis=1)
            focus[:, CARRIER] = np.where(ahead.any(1)[:, None], o_pos[np.arange(self.n), self.dfn[k]],
                                         me + np.array([5.0, 0.0], np.float32))
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

    def _stumble(self, pre_vel: np.ndarray) -> None:
        """Knock players off balance whose velocity a collision changed by more
        than they could have braked themselves (see TeamConfig.stumble)."""
        c = self.cfg
        dv = np.linalg.norm(self.vel - pre_vel, axis=-1)
        own = self.bodies.brake * self.cfg.field.dt * c.stumble_free
        exempt = self.partner >= 0
        if self.wrap.any():
            rows = np.nonzero(self.wrap > 0)[0]
            exempt[rows, CARRIER] = True
            exempt[rows, self.wrapper[rows]] = True
        hit = np.clip((dv - own) * c.stumble, 0.0, c.stumble_max) * ~exempt
        self.stun = np.maximum(self.stun, hit.astype(np.float32))

    def ball_at(self, center_start: np.ndarray) -> np.ndarray:
        """(N, 2) ball spot: in the air from the centre's starting spot, then on the carrier."""
        a = 1.0 if self.cfg.snap_time <= 0 else np.minimum(1.0, self.t * self.cfg.field.dt / self.cfg.snap_time)
        return center_start + (self.pos[:, CARRIER] - center_start) * np.reshape(a, (-1, 1))

    def blockers_on(self) -> np.ndarray:
        """(N, P) how many blockers each player is tied up with (blockers: 0/1)."""
        on = self.partner[:, self.blk]                                         # (N, B)
        count = np.zeros((self.n, self.p), np.int64)
        count[:, self.blk] = on >= 0
        count[:, self.dfn] = (on[:, :, None] == self.dfn[None, None, :]).sum(1)
        return count

    def _unlink(self, rows: np.ndarray, blk: int) -> None:
        """Blocker `blk` lets go of his man in `rows`; a defender still held by
        another blocker stays engaged with him."""
        d = self.partner[rows, blk]
        rows, d = rows[d >= 0], d[d >= 0]
        self.partner[rows, blk] = -1
        mine = self.partner[rows, d] == blk
        rows, d = rows[mine], d[mine]
        other = np.full(rows.size, -1, np.int64)
        for b2 in self.blk:
            if b2 != blk:
                other = np.where((other < 0) & (self.partner[rows, b2] == d), b2, other)
        self.partner[rows, d] = other

    def _release(self, square: np.ndarray) -> None:
        """Blockers turning their hips off their man (square-up under the bar) let go."""
        bc = self.cfg.block
        for j, blk in enumerate(self.blk):
            go = (self.partner[:, blk] >= 0) & (square[:, blk] < bc.release_below)
            if bc.release_double_only:
                go &= np.take_along_axis(self.blockers_on(), np.maximum(self.partner[:, blk, None], 0), 1)[:, 0] >= 2
            if go.any():
                rows = np.nonzero(go)[0]
                self._unlink(rows, blk)
                self.release_cd[rows, j] = bc.release_cooldown

    def _blocking(self) -> tuple[np.ndarray, np.ndarray]:
        """Shed existing blocks, then start new ones. Returns (new_engaged, shed) counts per play."""
        bc, b, dt = self.cfg.block, self.bodies, self.cfg.field.dt
        shed = np.zeros(self.n, np.int32)
        new = np.zeros(self.n, np.int32)
        for di, d in enumerate(self.dfn):
            on = self.partner[:, self.blk] == d                                  # (N, B)
            rows = np.nonzero(on.any(1))[0]
            if rows.size == 0:
                continue
            m_blk = (b.mass_lb[rows][:, self.blk] * on[rows]).sum(1)
            rate = bc.shed_rate * (b.mass_lb[rows, d] / m_blk) ** bc.shed_mass_power
            if bc.shed_leverage:
                at = (self.pos[rows][:, self.blk] * on[rows][..., None]).sum(1) / on[rows].sum(1, keepdims=True)
                to_d, to_c = self.pos[rows, d] - at, self.pos[rows, CARRIER] - at
                cos = np.sum(to_d * to_c, -1) / np.maximum(np.linalg.norm(to_d, axis=-1) * np.linalg.norm(to_c, axis=-1), 1e-6)
                rate = rate * np.exp(bc.shed_leverage * cos)
            go = self.rng.random(rows.size) < 1.0 - np.exp(-rate * dt)
            r = rows[go]
            for j, blk in enumerate(self.blk):
                self.partner[r[on[r, j]], blk] = -1
            self.partner[r, d] = -1
            self.shed_cd[r, di] = bc.reengage_cooldown
            shed[r] += 1
        engaged = self.partner[:, self.dfn] >= 0
        free_def = (~engaged | bc.double_team) & (self.shed_cd <= 0)
        if self.wrap.any():   # the defender wrapping the carrier can't be blocked
            free_def[np.arange(self.n), self.wrapper - self.n_off] &= self.wrap <= 0
        for j, blk in enumerate(self.blk):
            can = (self.partner[:, blk] < 0) & (self.stun[:, blk] <= 0) & (self.release_cd[:, j] <= 0)
            d = self.pos[:, self.dfn] - self.pos[:, blk, None]
            dist = np.linalg.norm(d, axis=-1)
            reach = b.radius[:, blk, None] + b.radius[:, self.dfn] + bc.reach
            ok = (dist < reach) & free_def & can[:, None]
            if not ok.any():
                continue
            pick = np.argmin(np.where(ok, dist, np.inf), axis=1)
            rows = np.nonzero(ok.any(axis=1))[0]
            di = pick[rows]
            first = self.partner[rows, self.n_off + di] < 0
            self.partner[rows, blk] = self.n_off + di
            self.partner[rows[first], self.n_off + di[first]] = blk
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
            p = 1.0 / (1.0 + np.exp(-logit))
            made = attempt & (self.rng.random(attempt.shape) < p)
            # Kept for reports: the odds of each attempt and whether the defender was engaged.
            self.last_tackle_p = np.where(attempt, p, np.nan)
        else:
            self.last_tackle_p = np.full(attempt.shape, np.nan)
        self.last_tackle_engaged = engaged & attempt
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
        waiting = self.t * dt < self.cfg.snap_time - 1e-6     # the ball hasn't reached the carrier
        if waiting.any():
            push[waiting, CARRIER] = 0.0
        if self.cfg.facing:
            self._turn_hips(push, np.clip(action[..., 2], 0.0, 1.0))
        if self.cfg.block.release_below is not None:
            self._release(np.clip(action[..., 2], 0.0, 1.0))
        # An engaged defender is fighting the block: only part of his push counts.
        n_on = self.blockers_on()[:, self.dfn]
        n_ = np.linalg.norm(push, axis=-1, keepdims=True)
        push = np.where(n_ > 1.0, push / np.maximum(n_, 1e-6), push)
        push[:, self.dfn] *= (self.cfg.block.engaged_control ** n_on)[..., None]
        bc = self.cfg.block
        if bc.blocker_control != 1.0:
            push[:, self.blk] *= np.where(self.partner[:, self.blk, None] >= 0, bc.blocker_control, 1.0)
        apply_controls(self.pos, self.vel, push, self.bodies, dt, stunned=self.stun > 0,
                       facing=self.facing if self.cfg.facing else None)
        if bc.engaged_speed is not None:
            sp = np.linalg.norm(self.vel, axis=-1, keepdims=True)
            over = (self.partner >= 0)[..., None] & (sp > bc.engaged_speed)
            capped = np.where(over, self.vel * bc.engaged_speed / np.maximum(sp, 1e-6), self.vel)
            self.pos += (capped - self.vel) * dt
            self.vel = capped

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
        if self.cfg.stumble:
            self._stumble(pre_vel)
        new_blocks, sheds = self._blocking()
        attempt, made, missed, wrapper = self._tackles(pre_vel)
        self.wrap[made] = tc.wrap_time
        self.wrapper[made] = wrapper[made]
        if made.any():      # a defender tackling out of a block leaves the block
            rows = np.nonzero(made)[0]
            for blk in self.blk:
                self.partner[rows[self.partner[rows, blk] == wrapper[rows]], blk] = -1
            self.partner[rows, wrapper[rows]] = -1
        self.stun[:, self.dfn] = np.where(missed, tc.miss_stun, self.stun[:, self.dfn])
        self.cooldown = np.where(missed, tc.miss_cooldown, self.cooldown)
        self.vel[:, CARRIER] *= (tc.miss_slowdown ** missed.sum(1))[:, None]

        self.t += 1
        self.stun = np.maximum(0.0, self.stun - dt)
        self.cooldown = np.maximum(0.0, self.cooldown - dt)
        self.shed_cd = np.maximum(0.0, self.shed_cd - dt)
        self.release_cd = np.maximum(0.0, self.release_cd - dt)
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
            "tackle_by_def": attempt, "tackle_missed_by_def": missed, "tackle_p": self.last_tackle_p, "tackle_engaged": self.last_tackle_engaged,
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
        if self.cfg.snap_time > 0:
            obs[:, GLOBAL_OBS] = np.maximum(0.0, 1.0 - self.t * f.dt / self.cfg.snap_time)
        per = obs[:, self.n_global:].reshape(self.n, self.p, PER_PLAYER_OBS)
        per[..., 0:2] = (pos[:, order] - ref[:, None]) / 20.0
        per[..., 2:4] = vel[:, order] / b.max_speed[:, order, None]
        per[..., 4:6] = face[:, order]
        per[..., 6] = self.blockers_on()[:, order]       # 0/1, or 2 for a double-teamed defender
        per[..., 7] = self.stun[:, order] / max(self.cfg.tackle.miss_stun, 1e-6)
        return obs

    def observe(self) -> np.ndarray:
        return self._obs("offense")

    def observe_tackler(self) -> np.ndarray:
        return self._obs("defense")

"""Scripted team policies for `TeamSim` (baselines and self-play pool members).

Each returns a team action (N, players * 3): push x, push y, square-up.
Everyone acts on what their team perceives (`sim.perceived`), so the
reaction delay applies to scripted players too.
"""
from __future__ import annotations

import numpy as np

from .team import CARRIER, TeamSim


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def _free_defenders(sim: TeamSim) -> np.ndarray:
    """(N, D) bool: defenders who can still make a play (not blocked, not stumbling)."""
    return (sim.partner[:, sim.dfn] < 0) & (sim.stun[:, sim.dfn] <= 0)


def carrier_evade(sim: TeamSim, pos: np.ndarray, danger: float = 6.0) -> np.ndarray:
    """Like the 1v1 `evasive_runner`, against the nearest free defender ahead. (N, 2)"""
    me = sim.pos[:, CARRIER]
    rel = pos[:, sim.dfn] - me[:, None]
    dist = np.linalg.norm(rel, axis=-1)
    threat = _free_defenders(sim) & (rel[..., 0] > -1.0)
    k = np.argmin(np.where(threat, dist, np.inf), axis=1)
    r = rel[np.arange(sim.n), k]
    d = dist[np.arange(sim.n), k]
    any_threat = threat.any(axis=1)
    side = -np.sign(r[:, 1] + 1e-3)
    to_mid = np.sign(sim.cfg.field.width / 2 - me[:, 1])
    near_edge = np.minimum(me[:, 1], sim.cfg.field.width - me[:, 1]) < 4.0
    side = np.where(near_edge, to_mid, side)
    strength = np.where(any_threat & (d < danger), 1.0 - d / danger, 0.0)
    return np.stack([np.ones(sim.n), side * strength * 1.5], axis=1)


def lead_blockers(sim: TeamSim, pos: np.ndarray) -> np.ndarray:
    """Each blocker picks a different free defender (closest to him, weighted by
    how close that defender is to the carrier), gets between him and the
    carrier, and once engaged drives him away from the carrier. (N, B, 2)"""
    n, nb = sim.n, len(sim.blk)
    carrier = sim.pos[:, CARRIER]
    d_pos = pos[:, sim.dfn]
    taken = ~(sim.partner[:, sim.dfn] < 0)        # engaged defenders are someone's already
    out = np.zeros((n, nb, 2), np.float32)
    for j, b in enumerate(sim.blk):
        me = sim.pos[:, b]
        partner = sim.partner[:, b]
        engaged = partner >= 0
        cost = (np.linalg.norm(d_pos - me[:, None], axis=-1)
                + 0.5 * np.linalg.norm(d_pos - carrier[:, None], axis=-1))
        k = np.argmin(np.where(taken, np.inf, cost), axis=1)
        has = ~taken.all(axis=1)
        taken[np.arange(n)[has & ~engaged], k[has & ~engaged]] = True
        tgt = d_pos[np.arange(n), k]
        # A yard in front of him on his line to the carrier.
        fit = tgt + _unit(carrier - tgt) * 1.0
        go = _unit(fit - me)
        p_pos = sim.pos[np.arange(n), np.maximum(partner, 0)]
        drive = _unit(p_pos - carrier)
        out[:, j] = np.where(engaged[:, None], drive, np.where(has[:, None], go, _unit(carrier - me) * 0.3))
    return out


def screen_offense(sim: TeamSim) -> np.ndarray:
    pos, _, _ = sim.perceived("offense")
    a = np.zeros((sim.n, sim.n_off, 3), np.float32)
    a[:, CARRIER, :2] = carrier_evade(sim, pos)
    a[:, sim.blk, :2] = lead_blockers(sim, pos)
    a[:, sim.blk, 2] = 1.0                         # blockers keep their hips on their man
    return a.reshape(sim.n, -1)


def carrier_only_offense(sim: TeamSim) -> np.ndarray:
    """Blockers stand still: what is the block worth?"""
    pos, _, _ = sim.perceived("offense")
    a = np.zeros((sim.n, sim.n_off, 3), np.float32)
    a[:, CARRIER, :2] = carrier_evade(sim, pos)
    return a.reshape(sim.n, -1)


def lane_runner(sim: TeamSim, read_until: float = 3.5, press_speed: float | None = None,
                hole: float = 1.6) -> np.ndarray:
    """Oklahoma-drill RB, the press-and-cut read from the coaching notes: head
    at the block, pick the side the blocker is covering (away from the side the
    DL is leaning to, or away from a DL who got free), cut there, then run.

    The side is re-read every step until he is `read_until` yd from the line
    (his own blocker's back is ~1.35 yd from it), then locked: a large value
    picks at the snap (no patience). With `press_speed` he holds that speed
    while reading, so he can still make the cut. (N, 2)"""
    pos, _, _ = sim.perceived("offense")
    n, w = sim.n, sim.cfg.field.width
    if getattr(sim, "_lane_side", None) is None or sim._lane_side.shape != (n,):
        sim._lane_side = np.zeros(n, np.float32)
    side_lock = sim._lane_side
    side_lock[sim.t == 0] = 0.0
    me, my_v = sim.pos[:, CARRIER], sim.vel[:, CARRIER]
    ol, dl = pos[:, sim.blk[0]], pos[:, sim.dfn[0]]
    engaged = sim.partner[:, sim.dfn[0]] >= 0
    lean = np.where(engaged, dl[:, 1] - ol[:, 1], dl[:, 1] - me[:, 1])
    side_now = -np.sign(lean + 1e-3)
    reading = (side_lock == 0) & (sim.los - me[:, 0] > read_until)
    lock = (side_lock == 0) & ~reading
    side_lock[lock] = side_now[lock]
    mid = (ol[:, 1] + dl[:, 1]) / 2
    hole_y = np.clip(mid + side_lock * hole, 1.0, w - 1.0)
    press = _unit(np.stack([sim.los - read_until + 0.5 - me[:, 0], mid - me[:, 1]], 1))
    if press_speed is not None:
        steer = press * press_speed - my_v
        press = steer / np.maximum(np.linalg.norm(steer, axis=-1, keepdims=True), 1.0)
    cut = _unit(np.stack([sim.los + 1.5, hole_y], 1) - me)
    # Through the line: straight downfield, damping sideways drift toward the middle.
    run = np.stack([np.full(n, 10.0), (w / 2 - me[:, 1]) * 1.0], 1) - my_v
    run /= np.maximum(np.linalg.norm(run, axis=-1, keepdims=True), 1.0)
    through = me[:, 0] > dl[:, 0] + 0.5
    push = np.where(reading[:, None], press, cut)
    return np.where(through[:, None], run, push).astype(np.float32)


def lane_offense(sim: TeamSim, **kw) -> np.ndarray:
    pos, _, _ = sim.perceived("offense")
    a = np.zeros((sim.n, sim.n_off, 3), np.float32)
    a[:, CARRIER, :2] = lane_runner(sim, **kw)
    a[:, sim.blk, :2] = lead_blockers(sim, pos)
    a[:, sim.blk, 2] = 1.0
    return a.reshape(sim.n, -1)


def pursuit_defense(sim: TeamSim, breakdown: float = 3.0, breakdown_speed: float = 4.5,
                    read_dist: float | None = None) -> np.ndarray:
    """Every defender takes an intercept angle to the carrier (goal-side) and
    breaks down near him, steering around free blockers in the way. An engaged
    defender fights toward the carrier. With `read_dist`, a defender farther
    than that squares up and mirrors instead of attacking (read-and-react)."""
    pos, vel, _ = sim.perceived("defense")
    n, nd = sim.n, sim.n_def
    c_pos, c_vel = pos[:, CARRIER, None], vel[:, CARRIER, None]
    me, my_v = sim.pos[:, sim.dfn], sim.vel[:, sim.dfn]
    rel = c_pos - me
    dist = np.linalg.norm(rel, axis=-1)
    closing = np.maximum(sim.bodies.max_speed[:, sim.dfn] * 0.8, 1.0)
    t_hit = np.clip(dist / closing, 0.0, 1.5)
    target = c_pos + c_vel * t_hit[..., None]
    target[..., 0] = np.maximum(target[..., 0], c_pos[..., 0] + 0.5)
    push = _unit(target - me)
    # Steer around a free blocker standing within 3 yd on the way.
    for b in sim.blk:
        bl = pos[:, b, None] - me
        bd = np.linalg.norm(bl, axis=-1)
        in_way = (sim.partner[:, b, None] < 0) & (bd < 3.0) & (np.sum(_unit(bl) * push, axis=-1) > 0.5)
        side = np.sign(push[..., 0] * bl[..., 1] - push[..., 1] * bl[..., 0] + 1e-3)   # blocker left/right of path
        around = np.stack([push[..., 1] * side, -push[..., 0] * side], axis=-1)
        push = np.where(in_way[..., None], _unit(push + around * 0.8), push)
    speed = np.linalg.norm(my_v, axis=-1)
    too_fast = (dist < breakdown) & (speed > breakdown_speed)
    push = np.where(too_fast[..., None], 0.5 * push + 0.5 * -_unit(my_v), push)
    square = np.zeros((n, nd), np.float32)
    if read_dist is not None:
        hold = dist > read_dist
        mid = sim.cfg.field.width / 2
        spot = np.concatenate([c_pos[..., :1] + 5.0, c_pos[..., 1:] + np.sign(mid - c_pos[..., 1:])], axis=-1)
        want = (spot - me) / 0.5
        want *= np.minimum(1.0, 4.5 / np.maximum(np.linalg.norm(want, axis=-1, keepdims=True), 1e-6))
        mirror = want - my_v
        mirror /= np.maximum(np.linalg.norm(mirror, axis=-1, keepdims=True), 1.0)
        push = np.where(hold[..., None], mirror, push)
        square = hold.astype(np.float32)
    engaged = sim.partner[:, sim.dfn] >= 0
    push = np.where(engaged[..., None], _unit(rel), push)
    return np.concatenate([push, square[..., None]], axis=-1).astype(np.float32).reshape(n, -1)


def read_defense(sim: TeamSim) -> np.ndarray:
    return pursuit_defense(sim, read_dist=8.0)


def gap_defense(sim: TeamSim, lead: float = 0.3) -> np.ndarray:
    """Pursuit, but an engaged defender plays his gap: he fights sideways to
    where the carrier is heading (`lead` s ahead) instead of straight at him
    through the blocker (how a DL two-gaps in the Oklahoma drill)."""
    a = pursuit_defense(sim).reshape(sim.n, sim.n_def, 3)
    pos, vel, _ = sim.perceived("defense")
    me = sim.pos[:, sim.dfn]
    aim_y = np.broadcast_to(pos[:, CARRIER, None, 1] + vel[:, CARRIER, None, 1] * lead, me[..., 1].shape)
    fight = np.stack([np.full_like(aim_y, -0.3), np.clip((aim_y - me[..., 1]) * 2.0, -1.0, 1.0)], -1)
    engaged = sim.partner[:, sim.dfn] >= 0
    a[..., :2] = np.where(engaged[..., None], _unit(fight), a[..., :2])
    return a.reshape(sim.n, -1)


OFFENSE = {"screen": screen_offense, "carrier_only": carrier_only_offense, "lane": lane_offense}
DEFENSE = {"pursuit": pursuit_defense, "read": read_defense, "gap": gap_defense}

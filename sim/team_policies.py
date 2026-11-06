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
                hole: float = 1.6, wait_lb: float = 0.0) -> np.ndarray:
    """Oklahoma-drill RB, the press-and-cut read from the coaching notes: head
    at the block, pick the side the blocker is covering (away from the side the
    DL is leaning to, or away from a DL who got free), cut there, then run.

    The side is re-read every step until he is `read_until` yd from the line
    (his own blocker's back is ~1.35 yd from it), then locked: a large value
    picks at the snap (no patience). With `press_speed` he holds that speed
    while reading, so he can still make the cut. With `wait_lb` > 0 he also
    holds at the press point, up to that many seconds, until the LB is
    blocked (the combo block's climb). (N, 2)"""
    pos, _, _ = sim.perceived("offense")
    n, w = sim.n, sim.cfg.field.width
    if getattr(sim, "_lane_side", None) is None or sim._lane_side.shape != (n,):
        sim._lane_side = np.zeros(n, np.float32)
    side_lock = sim._lane_side
    side_lock[sim.t == 0] = 0.0
    me, my_v = sim.pos[:, CARRIER], sim.vel[:, CARRIER]
    d0 = sim.dfn[0]
    dl = pos[:, d0]
    on_dl = sim.partner[:, sim.blk] == d0                                  # (N, B)
    blk_y = np.where(on_dl.any(1), (pos[:, sim.blk, 1] * on_dl).sum(1) / np.maximum(on_dl.sum(1), 1),
                     pos[:, sim.blk[0], 1])
    engaged = on_dl.any(1)
    lean = np.where(engaged, dl[:, 1] - blk_y, dl[:, 1] - me[:, 1])
    side_now = -np.sign(lean + 1e-3)
    reading = (side_lock == 0) & (sim.los - me[:, 0] > read_until)
    if wait_lb > 0 and sim.n_def > 1:
        lb_free = sim.partner[:, sim.dfn[1]] < 0
        reading |= (side_lock == 0) & lb_free & (sim.t * sim.cfg.field.dt < wait_lb)
    lock = (side_lock == 0) & ~reading
    side_lock[lock] = side_now[lock]
    mid = (blk_y + dl[:, 1]) / 2
    hole_y = np.clip(mid + side_lock * hole, 1.0, w - 1.0)
    spot = np.stack([sim.los - read_until + 0.5 - me[:, 0], mid - me[:, 1]], 1)
    press = _unit(spot)
    if press_speed is not None or wait_lb > 0:
        # Arrive at the press point and hold there while reading.
        top = press_speed if press_speed is not None else 10.0
        want = press * np.minimum(top, 2.0 * np.linalg.norm(spot, axis=1))[:, None]
        steer = want - my_v
        press = steer / np.maximum(np.linalg.norm(steer, axis=-1, keepdims=True), 1.0)
    cut = _unit(np.stack([sim.los + 1.5, hole_y], 1) - me)
    # Through the line: straight downfield, damping sideways drift toward the middle.
    run = np.stack([np.full(n, 10.0), (w / 2 - me[:, 1]) * 1.0], 1) - my_v
    run /= np.maximum(np.linalg.norm(run, axis=-1, keepdims=True), 1.0)
    through = me[:, 0] > np.minimum(dl[:, 0] + 0.5, sim.los + 1.0)   # past the DL, or a DL driven off the line
    push = np.where(reading[:, None], press, cut)
    return np.where(through[:, None], run, push).astype(np.float32)


def combo_blockers(sim: TeamSim, pos: np.ndarray, climb_depth: float | None = 2.5) -> np.ndarray:
    """Inside-zone combo: both OL take the DL; once the LB comes within
    `climb_depth` yd of the line (or he is not coming: None = climb at once),
    the OL nearer the LB turns his hips off the DL (square-up 0 = release)
    and climbs to him. (N, B, 3)"""
    n = sim.n
    a = np.zeros((n, len(sim.blk), 3), np.float32)
    a[..., :2] = lead_blockers(sim, pos)
    a[..., 2] = 1.0
    dl, lb = sim.dfn[0], sim.dfn[1]
    lb_free = sim.partner[:, lb] < 0
    on_dl = sim.partner[:, sim.blk] == dl
    climber = np.argmin(np.abs(pos[:, sim.blk, 1] - pos[:, lb, None, 1]), axis=1)
    come = np.ones(n, bool) if climb_depth is None else (pos[:, lb, 0] - sim.los < climb_depth)
    for j, b in enumerate(sim.blk):
        me = sim.pos[:, b]
        double_up = lb_free & ~come & ~on_dl[:, j] & (sim.partner[:, b] < 0)
        a[:, j, :2] = np.where(double_up[:, None], _unit(pos[:, dl] - me), a[:, j, :2])
        climb = lb_free & come & (climber == j) & on_dl.all(1)
        a[:, j, :2] = np.where(climb[:, None], _unit(pos[:, lb] - me), a[:, j, :2])
        a[:, j, 2] = np.where(climb, 0.0, a[:, j, 2])
    return a


def combo_offense(sim: TeamSim, climb_depth: float | None = 2.5, **runner) -> np.ndarray:
    pos, _, _ = sim.perceived("offense")
    a = np.zeros((sim.n, sim.n_off, 3), np.float32)
    a[:, CARRIER, :2] = lane_runner(sim, **runner)
    a[:, sim.blk] = combo_blockers(sim, pos, climb_depth)
    return a.reshape(sim.n, -1)


def box_runner(sim: TeamSim, read_until: float = 3.0) -> np.ndarray:
    """6v6 inside-run RB: pick the gap (between OL, or outside a tackle) whose
    spot just past the line is farthest from the defense (blocked defenders
    count as 1.5x farther), re-reading until `read_until` yd from the line,
    then hit it and run straight. (N, 2)"""
    pos, _, _ = sim.perceived("offense")
    n, w = sim.n, sim.cfg.field.width
    if getattr(sim, "_box_gap", None) is None or sim._box_gap.shape != (n,):
        sim._box_gap = np.full(n, np.nan, np.float32)
    lock = sim._box_gap
    lock[sim.t == 0] = np.nan
    me, my_v = sim.pos[:, CARRIER], sim.vel[:, CARRIER]
    ol_y = np.sort(pos[:, sim.blk, 1], axis=1)
    gaps = np.concatenate([ol_y[:, :1] - 1.3, (ol_y[:, 1:] + ol_y[:, :-1]) / 2, ol_y[:, -1:] + 1.3], axis=1)
    gaps = np.clip(gaps, 1.0, w - 1.0)                                       # (N, G)
    spot = np.stack([np.broadcast_to(sim.los[:, None] + 1.0, gaps.shape), gaps], -1)
    d = np.linalg.norm(spot[:, :, None] - pos[:, None, sim.dfn], axis=-1)    # (N, G, D)
    blocked = (sim.partner[:, sim.dfn] >= 0)[:, None]
    stunned = (sim.stun[:, sim.dfn] > 0)[:, None]
    d = np.where(stunned, np.inf, np.where(blocked, d * 1.5, d))
    best = gaps[np.arange(n), np.argmax(d.min(-1), axis=1)]
    reading = np.isnan(lock) & (sim.los - me[:, 0] > read_until)
    lock[np.isnan(lock) & ~reading] = best[np.isnan(lock) & ~reading]
    gap_y = np.where(reading, best, lock)
    press = _unit(np.stack([sim.los - read_until + 0.5 - me[:, 0], gap_y - me[:, 1]], 1))
    cut = _unit(np.stack([sim.los + 1.5, gap_y], 1) - me)
    run = np.stack([np.full(n, 10.0), np.zeros(n)], 1) - my_v
    run /= np.maximum(np.linalg.norm(run, axis=-1, keepdims=True), 1.0)
    through = me[:, 0] > sim.los + 1.0
    return np.where(through[:, None], run, np.where(reading[:, None], press, cut)).astype(np.float32)


def box_blockers(sim: TeamSim, pos: np.ndarray, climb_depth: float = 2.5) -> np.ndarray:
    """Zone-style combos for any number of OL: a free OL doubles the nearest
    singly-blocked DL while the LBs are deep; when an LB comes within
    `climb_depth` yd of the line, the double-teamer nearest him releases
    (square-up 0) and climbs to him. Otherwise `lead_blockers`. (N, B, 3)"""
    n = sim.n
    a = np.zeros((n, len(sim.blk), 3), np.float32)
    a[..., :2] = lead_blockers(sim, pos)
    a[..., 2] = 1.0
    roles = np.array(sim.roles)
    dls = sim.dfn[roles[sim.dfn] == "DL"]
    lbs = sim.dfn[roles[sim.dfn] == "LB"]
    on = sim.blockers_on()
    lb_deep = ((pos[:, lbs, 0] - sim.los[:, None]) >= climb_depth).all(1) if lbs.size else np.ones(n, bool)
    for j, b in enumerate(sim.blk):
        me = sim.pos[:, b]
        free = sim.partner[:, b] < 0
        single = on[:, dls] == 1
        dist = np.where(single, np.linalg.norm(pos[:, dls] - me[:, None], axis=-1), np.inf)
        k = np.argmin(dist, axis=1)
        go = free & lb_deep & np.isfinite(dist.min(1))
        a[:, j, :2] = np.where(go[:, None], _unit(pos[np.arange(n), dls[k]] - me), a[:, j, :2])
    for lb in lbs:
        come = (pos[:, lb, 0] - sim.los < climb_depth) & (sim.partner[:, lb] < 0)
        tgt = sim.partner[:, sim.blk]                                        # (N, B) whose man
        in_double = (tgt >= 0) & (on[np.arange(n)[:, None], np.maximum(tgt, 0)] >= 2)
        dist = np.where(in_double, np.linalg.norm(pos[:, sim.blk] - pos[:, lb, None], axis=-1), np.inf)
        j = np.argmin(dist, axis=1)
        climb = come & np.isfinite(dist.min(1))
        rows = np.nonzero(climb)[0]
        a[rows, j[rows], :2] = _unit(pos[rows, lb] - sim.pos[rows, sim.blk[j[rows]]])
        a[rows, j[rows], 2] = 0.0
    return a


def box_offense(sim: TeamSim, read_until: float = 3.0, climb_depth: float = 2.5) -> np.ndarray:
    pos, _, _ = sim.perceived("offense")
    a = np.zeros((sim.n, sim.n_off, 3), np.float32)
    a[:, CARRIER, :2] = box_runner(sim, read_until)
    a[:, sim.blk] = box_blockers(sim, pos, climb_depth)
    return a.reshape(sim.n, -1)


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


def strong_side(sim: TeamSim) -> np.ndarray:
    """+1 / -1 per play: the side of the middle the LBs stand on at the snap (`box-mix` slides
    both LBs up to 3 yd each play). Read on the play's first step and kept for the play."""
    lb = np.asarray(sim.dfn)[sim.n_def - 2:]
    side = getattr(sim, "_strong_side", None)
    if side is None or side.shape[0] != sim.n:
        side = sim._strong_side = np.ones(sim.n, np.float32)
    new = sim.t == 0
    if new.any():
        side[new] = np.where(sim.pos[new][:, lb, 1].mean(axis=1) >= sim.cfg.field.width / 2, 1.0, -1.0)
    return side


def overload_defense(sim: TeamSim, commit: float = 2.5, spot: float = 5.0, who: str = "all",
                     toward: str = "lb") -> np.ndarray:
    """A defense that guesses the run goes to the LBs' side (EXP-034). Until `commit` s, or until
    the carrier reaches the line, every free defender runs to a spot on the line `spot` yd to
    that side and every engaged DL fights toward it; then it plays `gap_defense`. The other
    side is left thin, and a runner who sees where the LBs stand can go there. With these
    defaults a runner aimed 5 yd to one side gains ~10 yd away from the LBs and ~0-3 yd toward
    them; 1.5 s or a 3 yd spot let a wide run get around the flood (`scripts/overload_check.py`).
    `who="lb"`: only the two LBs flood, the DL play `gap_defense` (a smaller flow, EXP-035).
    `toward="away"`: flood the side away from the LBs, to tell a runner who reads where the
    LBs stand from one who reads where the defense moves (`scripts/flood_cue.py`)."""
    a = gap_defense(sim).reshape(sim.n, sim.n_def, 3)
    side = strong_side(sim) * (-1.0 if toward == "away" else 1.0)
    early = (sim.t * sim.cfg.field.dt < commit) & (sim.pos[:, CARRIER, 0] < sim.los)
    if not early.any():
        return a.reshape(sim.n, -1)
    me = sim.pos[:, sim.dfn]
    goal = np.stack([np.broadcast_to(sim.los[:, None] + 0.5, me[..., 0].shape),
                     np.broadcast_to(sim.cfg.field.width / 2 + side[:, None] * spot, me[..., 1].shape)], -1)
    run_to = _unit(goal - me)
    fight = _unit(np.stack([np.full(me.shape[:2], -0.3), np.broadcast_to(side[:, None], me.shape[:2])], -1))
    engaged = sim.partner[:, sim.dfn] >= 0
    push = np.where(engaged[..., None], fight, run_to)
    flood = early[:, None] & (np.arange(sim.n_def) >= sim.n_def - 2 if who == "lb" else True)
    a[..., :2] = np.where(flood[..., None], push, a[..., :2])
    a[..., 2] = np.where(flood, 1.0, a[..., 2])
    return a.reshape(sim.n, -1)


def flow_side(sim: TeamSim, read_at: float = 1.0, min_lean: float = 0.0) -> np.ndarray:
    """+1 / -1 / 0 per play: the side away from where the defense has moved since the snap
    (mean y of all defenders, as the offense perceives them), decided once at `read_at` s and
    kept; 0 before that, and for good if the defense had moved less than `min_lean` yd."""
    pos, _, _ = sim.perceived("offense")
    y = pos[:, sim.dfn, 1].mean(axis=1)
    if getattr(sim, "_flow_y0", None) is None or sim._flow_y0.shape != (sim.n,):
        sim._flow_y0 = y.copy()
        sim._flow_side = np.zeros(sim.n, np.float32)
        sim._flow_done = np.zeros(sim.n, bool)
    new = sim.t == 0
    sim._flow_y0[new] = y[new]
    sim._flow_side[new] = 0.0
    sim._flow_done[new] = False
    decide = ~sim._flow_done & (sim.t * sim.cfg.field.dt >= read_at)
    dy = y - sim._flow_y0
    sim._flow_side[decide] = np.where(np.abs(dy[decide]) >= min_lean, np.where(dy[decide] >= 0, -1.0, 1.0), 0.0)
    sim._flow_done |= decide
    return sim._flow_side


def flow_cut(sim: TeamSim, side: np.ndarray, aim: float = 5.0) -> np.ndarray:
    """The carrier's push (N, 2): toward the line `aim` yd to `side` of the middle, straight
    upfield once past it."""
    n, mid = sim.n, sim.cfg.field.width / 2
    me, my_v = sim.pos[:, CARRIER], sim.vel[:, CARRIER]
    go = _unit(np.stack([sim.los + 1.0 - me[:, 0], mid + side * aim - me[:, 1]], 1))
    run = np.stack([np.full(n, 10.0), np.zeros(n)], 1) - my_v
    run /= np.maximum(np.linalg.norm(run, axis=-1, keepdims=True), 1.0)
    return np.where((me[:, 0] >= sim.los)[:, None], run, go).astype(np.float32)


def flow_offense(sim: TeamSim, read_at: float = 1.0, aim: float = 5.0, climb_depth: float = 2.5,
                 with_flow: bool = False) -> np.ndarray:
    """Reads the defense's flow (EXP-036): the RB drifts toward the middle of the line until
    `read_at` s, then aims `aim` yd to the side away from where the defense has moved since the
    snap, and runs straight once past the line. Blockers as `box_offense`. A defense that floods
    one side loses to it; one that leans one way and comes back fools it. `with_flow`: go the
    side the defense moved to instead (the opposite read, to tell what reading is worth)."""
    pos, _, _ = sim.perceived("offense")
    n, mid = sim.n, sim.cfg.field.width / 2
    side = flow_side(sim, read_at) * (-1.0 if with_flow else 1.0)
    me = sim.pos[:, CARRIER]
    wait = _unit(np.stack([sim.los - 2.0 - me[:, 0], mid - me[:, 1]], 1)) * 0.5
    a = np.zeros((n, sim.n_off, 3), np.float32)
    a[:, CARRIER, :2] = np.where((side == 0)[:, None] & (me[:, 0] < sim.los)[:, None], wait, flow_cut(sim, side, aim))
    a[:, sim.blk] = box_blockers(sim, pos, climb_depth)
    return a.reshape(n, -1)


def with_flow_read(offense, read_at: float = 1.0, min_lean: float = 0.5, aim: float = 5.0, with_flow: bool = False):
    """Any offense `fn(sim) -> (N, act)` whose RB cuts away from the defense's flow when the
    defense has moved at least `min_lean` yd by `read_at` s, and otherwise runs as it would
    (EXP-036). A defense can steer it by leaning on purpose: a bait."""
    def act(sim: TeamSim) -> np.ndarray:
        a = np.asarray(offense(sim), np.float32).reshape(sim.n, sim.n_off, 3).copy()
        side = flow_side(sim, read_at, min_lean) * (-1.0 if with_flow else 1.0)
        cut = (side != 0) & (sim.pos[:, CARRIER, 0] < sim.los + 30.0)
        a[cut, CARRIER, :2] = flow_cut(sim, side, aim)[cut]
        a[cut, CARRIER, 2] = 0.0
        return a.reshape(sim.n, -1)
    return act


OFFENSE = {"screen": screen_offense, "carrier_only": carrier_only_offense, "lane": lane_offense,
           "combo": combo_offense, "box": box_offense, "flow": flow_offense}
DEFENSE = {"pursuit": pursuit_defense, "read": read_defense, "gap": gap_defense, "overload": overload_defense}

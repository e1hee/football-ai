"""Scripted players for the pass sim (sim/passing.py): baselines and pool opponents.

`route_offense`: the receiver runs one route from a small NFL route tree, picked
at random each play, and the QB throws a fixed time after the break. The route
shares follow outside WRs in man coverage in BDB 2023-2025 (GO 23%, SLANT 15%,
HITCH 14%, OUT 12%, IN 7%, POST 5%, CORNER 4%; CROSS and SCREEN left out).
`man_coverage`: the CB keeps a cushion with inside leverage, turns and runs
when the receiver gets close, plays the ball in the air, then chases the catch.
"""
from __future__ import annotations

import numpy as np

from .passing import CB, QB, WR

# name: (break depth range yd, direction after the break in (downfield, inside) units, or None = stop)
ROUTES = {
    "go": ((99.0, 99.0), (1.0, 0.0)),
    "slant": ((2.0, 4.0), (1.0, 1.0)),
    "hitch": ((5.0, 7.0), None),
    "out": ((5.0, 10.0), (0.0, -1.0)),
    "in": ((5.0, 10.0), (0.0, 1.0)),
    "post": ((10.0, 13.0), (1.0, 0.8)),
    "corner": ((10.0, 13.0), (1.0, -0.8)),
}
ROUTE_SHARE = np.array([0.23, 0.15, 0.14, 0.12, 0.07, 0.05, 0.04])
NAMES = list(ROUTES)


def _route_state(sim) -> None:
    new = sim.t == 0
    if not hasattr(sim, "_route") or len(sim._route) != sim.n:
        sim._route = np.zeros(sim.n, np.int64)
        sim._depth = np.zeros(sim.n, np.float32)
        sim._broke_at = np.full(sim.n, np.inf, np.float32)
        new = np.ones(sim.n, bool)
    if new.any():
        k = int(new.sum())
        r = sim.rng.choice(len(NAMES), size=k, p=ROUTE_SHARE / ROUTE_SHARE.sum())
        sim._route[new] = r
        lo = np.array([ROUTES[NAMES[i]][0][0] for i in r])
        hi = np.array([ROUTES[NAMES[i]][0][1] for i in r])
        sim._depth[new] = sim.rng.uniform(lo, hi)
        sim._broke_at[new] = np.inf


def _arrive(spot, pos, vel, time_left, gain: float = 1.0) -> np.ndarray:
    """Push that turns the velocity toward the one that reaches `spot` in `time_left` s."""
    want = (spot - pos) / time_left[:, None]
    return (want - vel) * gain


def _track(aim, pos, vel, ref_vel, k: float = 1.5, gain: float = 1.0) -> np.ndarray:
    """Push toward velocity `ref_vel + k * (aim - pos)`: hold a spot that moves with someone."""
    return (ref_vel + k * (aim - pos) - vel) * gain


def route_offense(sim, throw_after: float = 0.5, go_throw: float = 2.6, latest: float = 3.4,
                  cut_speed: float = 4.0, run_speed: float = 8.0) -> np.ndarray:
    """(N, 6): run the play's route; throw `throw_after` s after the break (GO at `go_throw` s)."""
    _route_state(sim)
    n, dt = sim.n, sim.cfg.field.dt
    clock = sim.t * dt
    a = np.zeros((n, 2, 3), np.float32)
    depth_now = sim.pos[:, WR, 0] - sim.los
    broke = depth_now >= sim._depth
    sim._broke_at = np.where(broke & ~np.isfinite(sim._broke_at), clock, sim._broke_at)
    inside = sim.wr_inside
    for i, name in enumerate(NAMES):
        m = sim._route == i
        if not m.any():
            continue
        d = ROUTES[name][1]
        pre = m & ~broke
        a[pre, WR, 0] = 1.0
        post = m & broke
        if d is None:                            # hitch: stop and turn back to the QB
            a[post, WR, 0] = np.where(sim.vel[post, WR, 0] > 0.5, -1.0, 0.0)
        else:
            v = np.array(d, np.float32) / np.linalg.norm(d)
            sink = (clock[post] - sim._broke_at[post]) < 0.4           # slow down to cut, then go
            want = np.stack([np.full(int(post.sum()), v[0]), v[1] * inside[post]], 1) \
                * np.where(sink, cut_speed, run_speed)[:, None]
            a[post, WR, :2] = (want - sim.vel[post, WR]) * 2.0     # plant and cut: brake into the new line
    fly = sim.thrown & ~sim.resolved
    if fly.any():                                # ball in the air: get to the spot on time
        a[fly, WR, :2] = _arrive(sim.target[fly], sim.pos[fly, WR], sim.vel[fly, WR],
                                 np.maximum(sim.arrive_time[fly] - clock[fly], 0.1))
    if sim.caught.any():                         # after the catch: straight upfield
        a[sim.caught, WR, :2] = (1.0, 0.0)
    is_go = sim._route == NAMES.index("go")
    when = np.where(is_go, go_throw, sim._broke_at + throw_after)
    a[:, QB, 0] = np.where((clock >= np.minimum(when, latest)) & ~sim.thrown, 2.0, -2.0)
    return a.reshape(n, 6)


def man_coverage(sim, cushion: float = 1.5, leverage: float = 0.8, turn_at: float = 4.0) -> np.ndarray:
    """(N, 3): hold a spot `cushion` yd downfield and `leverage` yd inside of the receiver,
    moving with him (as seen with the reaction delay); go to the ball; chase the catch."""
    n = sim.n
    pos, vel, _ = sim.perceived("defense")
    wr, wv = pos[:, WR], vel[:, WR]
    me, mv = sim.pos[:, CB], sim.vel[:, CB]
    aim = wr + np.stack([np.full(n, cushion), leverage * sim.wr_inside], 1)
    aim[:, 0] = np.maximum(aim[:, 0], me[:, 0])           # off coverage: keep the cushion, never step up
    push = _track(aim, me, mv, wv)
    fly = sim.thrown & ~sim.resolved
    clock = sim.t * sim.cfg.field.dt
    if fly.any():
        push[fly] = _arrive(sim.target[fly], me[fly], mv[fly], np.maximum(sim.arrive_time[fly] - clock[fly], 0.1))
    c = sim.caught
    if c.any():
        push[c] = wr[c] + 0.5 * wv[c] - me[c]
    out = np.zeros((n, 3), np.float32)
    out[:, :2] = push
    # Face the receiver (backpedal) while he is far; turn and run once he is within `turn_at` yd.
    far = np.linalg.norm(wr - me, axis=-1) > turn_at
    out[:, 2] = np.where(fly | sim.caught | ~far, 0.0, 1.0)
    return out


def go_and_throw(sim, at: float = 2.5) -> np.ndarray:
    """Baseline: run straight upfield, throw at `at` s. A sim where this wins is too easy (EXP-000)."""
    a = np.zeros((sim.n, 2, 3), np.float32)
    a[:, WR, 0] = 1.0
    a[:, QB, 0] = np.where(sim.t * sim.cfg.field.dt >= at, 2.0, -2.0)
    return a.reshape(sim.n, 6)


def standing_cb(sim) -> np.ndarray:
    return np.zeros((sim.n, 3), np.float32)

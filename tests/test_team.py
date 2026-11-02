import dataclasses

import numpy as np

from sim.core import FieldConfig, OUTCOME_TOUCHDOWN, TackleConfig
from sim.team import CARRIER, TeamConfig, TeamSim
from sim.team_policies import OFFENSE, DEFENSE
from train.evaluate import evaluate


def _sim(n=8, **kw):
    sim = TeamSim(n, TeamConfig(**kw), seed=0)
    sim.reset()
    return sim


def _still(sim, side):
    return np.zeros((sim.n, sim.act_dim_for(side)), np.float32)


def test_shapes():
    sim = _sim()
    assert sim.observe().shape == (8, 4 + 8 * 6) == sim.observe_tackler().shape
    assert sim.act_dim_for("runner") == 9 and sim.act_dim_for("tackler") == 9
    assert sim.roles == ("RB", "OL", "OL", "LB", "DB", "DB")


def _engage(sim, blk=1, dfn=4):
    sim.partner[:, blk] = dfn
    sim.partner[:, dfn] = blk


def test_blocked_defender_cannot_tackle():
    sim = _sim(block=dataclasses.replace(TeamConfig().block, shed_rate=0.0))
    sim.pos[:] = (40.0, 5.0)                       # park everyone far away ...
    sim.pos[:, CARRIER] = (20.0, 26.0)
    sim.pos[:, 1] = (21.5, 26.6)                   # ... except an OL on a DB next to the carrier
    sim.pos[:, 4] = (21.0, 26.0)
    for i in range(sim.p):
        sim.pos[:, i, 1] += 0 if i in (0, 1, 4) else 3 * i
    _engage(sim)
    _, _, _, info = sim.step(_still(sim, "runner"), _still(sim, "tackler"))
    assert not info["tackle_attempt"].any()


def test_lineman_anchors_a_db_more_than_a_lb():
    moved = {}
    for d_idx, role in ((4, "DB"), (3, "LB")):
        sim = _sim(block=dataclasses.replace(TeamConfig().block, shed_rate=0.0), facing=False)
        sim.pos[:] = np.array([[5, 5], [30, 26], [5, 45], [50, 5], [50, 45], [50, 30]], np.float32)
        sim.pos[:, d_idx] = (31.0, 26.0)
        _engage(sim, 1, d_idx)
        d_act = np.zeros((sim.n, 3, 3), np.float32)
        d_act[:, d_idx - 3, 0] = -1.0                 # defender drives into the blocker, toward the backfield
        o_act = np.zeros((sim.n, 3, 3), np.float32)
        o_act[:, 1, 0] = 1.0                          # blocker drives back
        start = sim.pos[:, d_idx, 0].copy()
        for _ in range(10):
            sim.step(o_act.reshape(sim.n, -1), d_act.reshape(sim.n, -1))
        moved[role] = float((start - sim.pos[:, d_idx, 0]).mean())
    assert moved["DB"] < moved["LB"]
    assert moved["DB"] < 0.5                          # the 194 lb DB is driven back or held


def test_heavier_defenders_shed_faster():
    sheds = {}
    for d_idx in (3, 4):                              # LB (254 lb) vs DB (194 lb) against a 315 lb OL
        sim = _sim(n=4000)
        sim.pos[:] = np.array([[5, 5], [30, 26], [5, 45], [50, 5], [50, 45], [50, 30]], np.float32)
        sim.pos[:, d_idx] = (31.0, 26.0)
        _engage(sim, 1, d_idx)
        sim._blocking()
        sheds[d_idx] = (sim.partner[:, 1] < 0).mean()
    assert sheds[3] > sheds[4] > 0


def test_breakaway_needs_lead_over_every_defender():
    sim = _sim(field=FieldConfig(breakaway_lead=2.0, full_play_share=0.0), facing=False)
    sim.pos[:] = np.array([[30, 26], [5, 5], [5, 45], [10, 10], [10, 40], [29, 50]], np.float32)
    sim.vel[:, CARRIER] = (8.0, 0.0)
    _, _, done, info = sim.step(np.tile([1, 0, 0] * 3, (sim.n, 1)), _still(sim, "tackler"))
    assert not info["breakaway"].any()               # one defender is level with him
    sim.pos[:, 5] = (10.0, 50.0)
    _, _, done, info = sim.step(np.tile([1, 0, 0] * 3, (sim.n, 1)), _still(sim, "tackler"))
    assert info["breakaway"].all() and np.all(info["outcome"] == OUTCOME_TOUCHDOWN)


def test_blocking_is_worth_something():
    cfg = TeamConfig(field=FieldConfig(reaction_delay=2, breakaway_lead=2.0),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    with_b = evaluate(lambda s, _: OFFENSE["screen"](s), DEFENSE["pursuit"], cfg, n_plays=2048)
    without = evaluate(lambda s, _: OFFENSE["carrier_only"](s), DEFENSE["pursuit"], cfg, n_plays=2048)
    assert with_b["rate/touchdown"] > without["rate/touchdown"] + 0.10

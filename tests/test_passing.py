import numpy as np

from sim.core import (OUTCOME_INCOMPLETE, OUTCOME_INTERCEPTION, OUTCOME_SACK, OUTCOME_TACKLED, OUTCOME_TOUCHDOWN,
                      FieldConfig, TackleConfig)
from sim.pass_policies import go_and_throw, man_coverage, route_offense, standing_cb
from sim.passing import CB, QB, WR, PassSim, pass_config
from sim.team import make_sim
from train.run_config import build_config, run_args


def cfg(**kw):
    return pass_config(field=FieldConfig(reaction_delay=2), tackle=TackleConfig(base_logit=3.2), facing=True,
                       timeout_ends=True, **kw)


def play_out(sim, off, dfn, steps=90):
    """Run every slot to its first end; outcomes and yards."""
    out = np.zeros(sim.n, np.int8)
    yards = np.zeros(sim.n, np.float32)
    fin = np.zeros(sim.n, bool)
    for _ in range(steps):
        _, _, done, info = sim.step(off(sim), dfn(sim))
        new = done & ~fin
        out[new], yards[new] = info["outcome"][new], info["yards"][new]
        fin |= done
        if fin.all():
            break
    return out, yards


def test_shapes_and_start():
    sim = PassSim(64, cfg(), seed=0)
    ob = sim.reset()
    assert ob.shape == (64, sim.obs_dim) and sim.observe_tackler().shape == (64, sim.obs_dim)
    assert np.isfinite(ob).all()
    assert sim.act_dim_for("runner") == 6 and sim.act_dim_for("tackler") == 3
    assert (sim.pos[:, QB, 0] == 4.0).all()                          # 6 yd behind the line at 10
    assert (sim.pos[:, CB, 0] > sim.pos[:, WR, 0]).all()             # the CB starts downfield of the receiver
    assert ((sim.pos[:, WR, 1] < 15) | (sim.pos[:, WR, 1] > 53.3 - 15)).all()


def test_no_throw_is_a_sack_and_costs_yards():
    sim = PassSim(16, cfg(), seed=0)
    sim.reset()
    hold = lambda s: np.tile([1.0, 0, 0, -1.0, 0, 0], (s.n, 1)).astype(np.float32)
    out, yards = play_out(sim, hold, standing_cb)
    assert (out == OUTCOME_SACK).all() and np.allclose(yards, -7.0)


def test_qb_stays_put_and_ball_leads_a_straight_receiver():
    sim = PassSim(256, cfg(), seed=1)
    sim.reset()
    spot = sim.pos[:, QB].copy()
    out, yards = play_out(sim, go_and_throw, standing_cb)
    assert np.allclose(sim.pos[:, QB], spot)
    # Thrown on the receiver's line; misses only from his late acceleration, catches when he is there.
    assert np.isin(out, [OUTCOME_TOUCHDOWN, OUTCOME_INCOMPLETE, OUTCOME_TACKLED]).all()


def test_receiver_who_turns_away_after_the_throw_drops_it():
    sim = PassSim(256, cfg(), seed=2)
    sim.reset()

    def throw_then_turn(s):
        a = go_and_throw(s, at=1.5).reshape(s.n, 2, 3)
        a[s.thrown & ~s.resolved, WR, :2] = (-1.0, 0.0)       # stops and comes back while the ball flies
        return a.reshape(s.n, 6)
    out, _ = play_out(sim, throw_then_turn, standing_cb)
    assert (out == OUTCOME_INCOMPLETE).mean() > 0.9


def test_no_tackle_before_the_catch():
    sim = PassSim(128, cfg(), seed=3)
    sim.reset()
    sim.pos[:, CB] = sim.pos[:, WR] + [0.8, 0.0]                  # in reach from the start
    attempts = 0
    for _ in range(4):                                            # before any throw is allowed
        _, _, _, info = sim.step(np.zeros((sim.n, 6), np.float32), np.zeros((sim.n, 3), np.float32))
        attempts += info["tackle_attempt"].sum()
    assert attempts == 0


def test_catches_happen_more_with_separation_and_tackles_after():
    sim = PassSim(4096, cfg(), seed=4)
    sim.reset()
    out, yards = play_out(sim, route_offense, man_coverage)
    comp = np.isin(out, [OUTCOME_TOUCHDOWN, OUTCOME_TACKLED]).mean()
    assert 0.3 < comp < 0.8                                        # NFL outside WR vs man: 52%
    assert (out == OUTCOME_TACKLED).mean() > 0.05                  # the CB makes tackles after catches
    assert 0.0 < (out == OUTCOME_INTERCEPTION).mean() < 0.12
    assert np.allclose(yards[out == OUTCOME_INTERCEPTION], -15.0)
    assert yards.max() <= 30.0 + 1e-4                              # capped at the goal line


def test_built_from_args_like_training():
    a = run_args(__import__("pathlib").Path(__file__).resolve().parents[1] / "experiments" / "EXP-019-mix" / "s0")
    a.sim, a.brake = "pass", 2
    sim = make_sim(8, build_config(a), seed=0)
    assert isinstance(sim, PassSim) and sim.cfg.brake == 2

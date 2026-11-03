import dataclasses

import numpy as np

from sim.core import FieldConfig, OUTCOME_TOUCHDOWN, TackleConfig
from sim.team import CARRIER, TeamConfig, TeamSim, lane_drill_config
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


def test_blocker_potential_is_zero_on_the_line_and_negative_off_it():
    sim = _sim(n=2)
    sim.pos[:] = np.array([[10, 26], [15, 26], [15, 40], [20, 26], [50, 5], [50, 50]], np.float32)
    # Blocker 1 sits right between the LB (20, 26) and the carrier; blocker 2 is 14 yd off any line.
    phi = sim.blocker_potential()
    assert np.all(phi < 0) and np.all(phi > -20)
    sim.partner[:, 2] = 4                    # once engaged, a blocker adds nothing
    sim.partner[:, 4] = 2
    assert np.allclose(sim.blocker_potential(), 0.0, atol=1e-5)


def test_shaping_goes_to_the_offense_only():
    from sim.core import RewardConfig
    sim = _sim(n=64, reward=RewardConfig(yards_scale=0.1, blocker_shaping=0.05))
    a = OFFENSE["screen"](sim)
    _, rew, _, info = sim.step(a, DEFENSE["pursuit"](sim))
    plain = 0.1 * (info["r_pos"][:, 0] - 5.0)    # carriers start at x=5
    assert np.allclose(-info["tackler_reward"], plain, atol=1e-4)
    assert not np.allclose(rew, plain, atol=1e-4)


def test_shaping_pays_no_end_bonus_on_timeout():
    from sim.core import OUTCOME_TIMEOUT, RewardConfig
    sim = _sim(n=4, field=FieldConfig(max_steps=1), reward=RewardConfig(yards_scale=0.0, blocker_shaping=1.0),
               facing=False)
    sim.pos[:] = np.array([[10, 26], [5, 5], [5, 45], [30, 26], [40, 10], [40, 40]], np.float32)  # blockers far off
    phi0 = sim.blocker_potential()
    _, rew, done, info = sim.step(_still(sim, "runner"), _still(sim, "tackler"))
    assert np.all(info["outcome"] == OUTCOME_TIMEOUT)
    # gamma * Phi(s') - Phi(s) with s' kept, ~0 for players who barely moved; not -Phi(s) (a big bonus).
    assert np.all(np.abs(rew) < 0.2 * np.abs(phi0))


# ------------------------------------------------------------ lane drill
def _lane(n=8, **kw):
    sim = TeamSim(n, lane_drill_config(**kw), seed=0)
    sim.reset()
    return sim


def test_lane_starts_engaged_on_the_line():
    sim = _lane()
    assert sim.roles == ("RB", "OL", "DL")
    assert (sim.partner[:, 1] == 2).all() and (sim.partner[:, 2] == 1).all()
    np.testing.assert_allclose(sim.los, 10.0)
    np.testing.assert_allclose(sim.yards, -5.0)                 # RB 5 yd behind the line
    assert (sim.pos[:, 1, 0] < sim.los).all() and (sim.pos[:, 2, 0] > sim.los).all()
    assert (sim.observe().reshape(sim.n, -1)[:, 4 + 8 + 6] == 1).all()   # OL shows as engaged


def test_lane_linebacker_is_unblocked_off_the_line():
    sim = _lane(linebacker=True)
    assert sim.roles == ("RB", "OL", "DL", "LB")
    assert (sim.partner[:, 3] == -1).all() and (sim.pos[:, 3, 0] >= sim.los + 4.0).all()


def _carrier_next_to_engaged_dl(sim, carrier_xy):
    sim.pos[:, 1] = (10.0, 2.5)
    sim.pos[:, 2] = (10.9, 2.5)
    sim.pos[:, CARRIER] = carrier_xy
    sim.vel[:] = 0


def test_engaged_dl_reaches_a_carrier_beside_him_not_behind_his_blocker():
    block = dataclasses.replace(lane_drill_config().block, shed_rate=0.0, engaged_tackle_logit=0.0)
    beside = _lane(64, block=block)
    _carrier_next_to_engaged_dl(beside, (10.9, 3.7))           # 1.2 yd to his side
    _, _, _, info = beside.step(_still(beside, "runner"), _still(beside, "tackler"))
    assert info["tackle_attempt"].all()
    behind = _lane(64, block=block)
    _carrier_next_to_engaged_dl(behind, (10.06, 3.5))          # in reach, 50 deg off the line to the OL
    _, _, _, info = behind.step(_still(behind, "runner"), _still(behind, "tackler"))
    assert not info["tackle_attempt"].any()
    screen = _lane(64, block=dataclasses.replace(block, engaged_tackle_logit=None))
    _carrier_next_to_engaged_dl(screen, (10.9, 3.7))
    _, _, _, info = screen.step(_still(screen, "runner"), _still(screen, "tackler"))
    assert not info["tackle_attempt"].any()


def test_tackle_out_of_a_block_ends_the_block():
    block = dataclasses.replace(lane_drill_config().block, shed_rate=0.0, engaged_tackle_logit=20.0)
    sim = _lane(16, block=block)
    _carrier_next_to_engaged_dl(sim, (10.9, 3.7))
    _, _, _, info = sim.step(_still(sim, "runner"), _still(sim, "tackler"))
    assert info["tackle_made"].all()
    assert (sim.partner == -1).all()


def test_lane_scripted_runner_beats_scripted_dl_sometimes():
    from sim.team_policies import gap_defense, lane_offense
    for lb in (False, True):
        m = evaluate(lambda s, o=None: lane_offense(s), lambda s, o=None: gap_defense(s),
                     lane_drill_config(linebacker=lb), n_plays=512)
        assert 0.05 < m["rate/touchdown"] < 0.9


# ------------------------------------------------------------ balance / hips
def _bump(speed, **kw):
    """RB running at `speed` into the back of a standing OL, far from the DL."""
    sim = _lane(4, **kw)
    sim.partner[:] = -1
    sim.pos[:, 2] = (17.0, 2.5)
    sim.pos[:, 1] = (8.0, 2.5)
    sim.pos[:, CARRIER] = (8.0 - 0.9 - speed * 0.1 + 0.05, 2.5)
    sim.vel[:] = 0
    sim.vel[:, CARRIER] = (speed, 0.0)
    a = np.zeros((sim.n, 2, 3), np.float32)
    a[:, CARRIER, 0] = 1.0
    sim.step(a.reshape(sim.n, -1), _still(sim, "tackler"))
    return sim


def test_hard_collision_knocks_the_runner_off_balance():
    assert (_bump(6.0, stumble=0.1).stun[:, CARRIER] > 0.1).all()
    assert (_bump(1.0, stumble=0.1).stun[:, CARRIER] == 0).all()     # a soft bump is fine
    assert (_bump(6.0).stun[:, CARRIER] == 0).all()                  # off by default


def test_broken_free_carrier_squares_up_downfield():
    def hips(**kw):
        sim = _lane(4, block=dataclasses.replace(lane_drill_config().block, shed_rate=0.0), **kw)
        sim.pos[:, CARRIER] = (11.5, 2.5)                            # just past the DL at ~10.5
        sim.vel[:, CARRIER] = (3.0, 0.0)
        sim.facing[:, CARRIER] = (1.0, 0.0)
        a = np.zeros((sim.n, 2, 3), np.float32)
        a[:, CARRIER] = (1.0, 0.0, 1.0)                              # full square-up
        for _ in range(8):
            _, _, done, _ = sim.step(a.reshape(sim.n, -1), _still(sim, "tackler"))
            assert not done.any()
        return sim.facing[:, CARRIER, 0]
    assert (hips() < 0).all()                                        # turns toward the DL behind him
    assert (hips(carrier_square_ahead=True) > 0.9).all()

import numpy as np

from sim.core import (OBS_DIM, OUTCOME_TACKLED, OUTCOME_TOUCHDOWN, FieldConfig, OneVOneSim,
                      SimConfig)
from sim.physics import PROFILES
from sim.policies import pursuit_tackler, straight_runner


def test_obs_shape_and_range():
    sim = OneVOneSim(8, seed=1)
    obs = sim.reset()
    assert obs.shape == (8, OBS_DIM)
    assert np.all(np.abs(obs) <= 1.5)


def test_straight_runner_scores_when_tackler_is_far():
    sim = OneVOneSim(4, SimConfig(), seed=2)
    sim.reset()
    sim.t_pos[:, 1] = 1.0      # tackler parked at the far sideline, never moves
    sim.r_pos[:, 1] = 40.0
    total = np.zeros(4, np.float32)
    for _ in range(sim.cfg.field.max_steps):
        _, r, done, info = sim.step(straight_runner(sim), np.zeros((4, 2), np.float32))
        total += r
        if done.all():
            break
    assert np.all(info["outcome"] == OUTCOME_TOUCHDOWN)
    assert np.allclose(total, 55.0, atol=1.0)   # start x=5, goal x=60


def test_runner_accelerates_like_a_receiver():
    """From a standstill a WR-profile runner should cover 10 yd in ~1.6-1.9 s
    (NFL combine 10-yd splits for WRs are ~1.5-1.6 s from a 3-point stance)."""
    sim = OneVOneSim(1, seed=3)
    sim.reset()
    sim.t_pos[:, 1] = 1.0
    sim.r_pos[:, 1] = 40.0
    start = sim.r_pos[0, 0]
    for k in range(1, 40):
        sim.step(straight_runner(sim), np.zeros((1, 2), np.float32))
        if sim.r_pos[0, 0] - start >= 10.0:
            break
    assert 1.4 <= k * 0.1 <= 2.0, k * 0.1


def test_cannot_cut_sharply_at_top_speed():
    """At full speed a 90-degree cut must take much longer than from a jog."""
    def time_to_turn(v0: float) -> float:
        sim = OneVOneSim(1, seed=4)
        sim.reset()
        sim.t_pos[:, 1] = 1.0
        sim.r_pos[0] = [10.0, 26.0]
        sim.r_vel[0] = [v0, 0.0]
        for k in range(1, 60):
            sim.step(np.array([[0.0, 1.0]], np.float32), np.zeros((1, 2), np.float32))
            vx, vy = sim.r_vel[0]
            if np.degrees(np.arctan2(vy, vx)) >= 80:
                return k * 0.1
        return 6.0
    assert time_to_turn(9.5) > 1.5 * time_to_turn(3.0)


def test_big_players_can_still_brake_near_top_speed():
    """NFL DL / LB shed ~2.1 / 2.7 yd/s in 0.5 s from 7-9 yd/s (p99, EXP-038);
    the old brake form left them ~0.4 / 1.1."""
    from sim.physics import Bodies, apply_controls
    for name, want in (("DL", 1.8), ("LB", 2.3)):
        b = Bodies(1, [PROFILES[name]])
        pos = np.zeros((1, 1, 2), np.float32)
        vel = np.array([[[8.0, 0.0]]], np.float32)
        for _ in range(5):
            apply_controls(pos, vel, np.array([[[-1.0, 0.0]]], np.float32), b, 0.1)
        assert 8.0 - vel[0, 0, 0] > want, name


def test_players_do_not_pass_through_each_other():
    sim = OneVOneSim(1, SimConfig(tackle=SimConfig().tackle.__class__(reach=-10.0)), seed=5)
    sim.reset()
    sim.r_pos[0] = [10.0, 26.0]
    sim.t_pos[0] = [14.0, 26.0]
    for _ in range(30):
        sim.step(np.array([[1.0, 0.0]], np.float32), np.array([[-1.0, 0.0]], np.float32))
        gap = np.linalg.norm(sim.r_pos[0] - sim.t_pos[0])
        assert gap >= 0.8, gap      # radii 0.45 + 0.45 minus small solver slack


def test_head_on_slow_runner_gets_tackled():
    cfg = SimConfig(runner=PROFILES["WR"].scaled(max_speed=0.6))
    sim = OneVOneSim(64, cfg, seed=6)
    sim.reset()
    sim.t_pos[:, 1] = sim.r_pos[:, 1]  # lined up directly in front
    outcomes = np.zeros(64, np.int8)
    finished = np.zeros(64, bool)
    for _ in range(cfg.field.max_steps):
        _, _, done, info = sim.step(straight_runner(sim), pursuit_tackler(sim))
        outcomes[done & ~finished] = info["outcome"][done & ~finished]
        finished |= done
        if finished.all():
            break
    assert (outcomes == OUTCOME_TACKLED).mean() > 0.9


def test_made_tackle_still_gains_yards_after_contact():
    """A runner at speed hitting a standing tackler drives the pile forward.
    (Head-on at equal speed and mass the momenta cancel: no yards after contact.)"""
    sim = OneVOneSim(256, SimConfig(field=FieldConfig(tackler_lateral_spread=0.0)), seed=7)
    sim.reset()
    contact_x = np.full(256, np.nan)
    end_x = np.full(256, np.nan)
    finished = np.zeros(256, bool)
    still = np.zeros((256, 2), np.float32)
    for _ in range(100):
        made_before = ~np.isnan(contact_x)
        _, _, done, info = sim.step(straight_runner(sim), still)
        new_made = info["tackle_made"] & ~made_before & ~finished
        contact_x[new_made] = info["r_pos"][new_made, 0]
        fin = done & ~finished & (info["outcome"] == OUTCOME_TACKLED)
        end_x[fin] = info["r_pos"][fin, 0]
        finished |= done
        if finished.all():
            break
    ok = ~np.isnan(contact_x) & ~np.isnan(end_x)
    assert ok.sum() > 50
    assert np.median(end_x[ok] - contact_x[ok]) > 0.3   # pile keeps moving forward


def test_auto_reset_keeps_running():
    sim = OneVOneSim(16, seed=8)
    sim.reset()
    for _ in range(300):
        obs, _, _, _ = sim.step(straight_runner(sim), pursuit_tackler(sim))
    assert obs.shape == (16, OBS_DIM)
    assert np.all(sim.t <= sim.cfg.field.max_steps)


def test_breakaway_ends_play_as_touchdown_with_full_yards():
    cfg = SimConfig(field=FieldConfig(breakaway_lead=2.0, full_play_share=0.0))
    sim = OneVOneSim(4, cfg, seed=9)
    sim.reset()
    sim.r_pos[:] = [[20.0, 26.0]] * 4
    sim.t_pos[:] = [[17.0, 26.0]] * 4     # runner already 3 yd past the tackler
    sim.r_vel[:] = [[8.0, 0.0]] * 4       # ...and running away
    _, r, done, info = sim.step(straight_runner(sim), np.zeros((4, 2), np.float32))
    assert done.all() and info["breakaway"].all()
    assert np.all(info["outcome"] == OUTCOME_TOUCHDOWN)
    # Teleported to x=20 with 0 yards logged: this step's own movement plus the rest
    # of the field is paid at once, so yards = 40 and reward == yards for the play.
    assert np.allclose(info["yards"], 40.0, atol=0.01)
    assert np.allclose(r, 40.0, atol=0.01)


def test_breakaway_off_keeps_spawns_identical():
    a = OneVOneSim(64, SimConfig(field=FieldConfig(tackler_lateral_spread=8.0)), seed=10)
    b = OneVOneSim(64, SimConfig(field=FieldConfig(tackler_lateral_spread=8.0, breakaway_lead=None)), seed=10)
    a.reset(); b.reset()
    assert np.array_equal(a.pos, b.pos)


def test_full_play_share_plays_run_to_the_goal():
    cfg = SimConfig(field=FieldConfig(breakaway_lead=2.0, full_play_share=1.0))
    sim = OneVOneSim(4, cfg, seed=11)
    sim.reset()
    sim.r_pos[:] = [[20.0, 26.0]] * 4
    sim.t_pos[:] = [[17.0, 26.0]] * 4
    _, _, done, info = sim.step(straight_runner(sim), np.zeros((4, 2), np.float32))
    assert not done.any() and not info["breakaway"].any()


def test_reaction_delay_shows_the_opponent_late():
    cfg = SimConfig(field=FieldConfig(reaction_delay=3))
    sim = OneVOneSim(2, cfg, seed=12)
    sim.reset()
    start_t = sim.t_pos.copy()
    for _ in range(5):
        sim.step(straight_runner(sim), np.array([[0.0, 1.0]] * 2, np.float32))   # tackler slides sideways
    seen_pos, _ = sim.seen(1)
    # The runner perceives where the tackler was 3 steps ago, not where it is.
    assert not np.allclose(seen_pos, sim.t_pos)
    assert np.all(seen_pos[:, 1] < sim.t_pos[:, 1])
    assert np.all(seen_pos[:, 1] > start_t[:, 1] - 1e-6)


def test_no_delay_is_identical_to_before():
    a = OneVOneSim(32, SimConfig(field=FieldConfig(tackler_lateral_spread=8.0)), seed=13)
    b = OneVOneSim(32, SimConfig(field=FieldConfig(tackler_lateral_spread=8.0, reaction_delay=0)), seed=13)
    oa, ob = a.reset(), b.reset()
    for _ in range(40):
        oa, *_ = a.step(straight_runner(a), pursuit_tackler(a))
        ob, *_ = b.step(straight_runner(b), pursuit_tackler(b))
    assert np.array_equal(oa, ob)


def test_overspeed_tackler_misses_more():
    from sim.core import TackleConfig
    def made_rate(t_speed):
        tc = TackleConfig(overspeed_weight=0.5, sidestep_weight=0.35)
        sim = OneVOneSim(4000, SimConfig(tackle=tc), seed=14)
        sim.reset()
        sim.r_pos[:] = [10.0, 26.0]; sim.t_pos[:] = [11.0, 26.0]
        sim.r_vel[:] = [6.0, 0.0]; sim.t_vel[:] = [-t_speed, 0.0]
        attempt, made = sim._tackle_attempts(sim.vel.copy())
        return made.sum() / attempt.sum()
    assert made_rate(9.0) < made_rate(4.0) - 0.15


def test_long_push_keeps_its_direction():
    """A push longer than 1 must be shortened along its own direction, not clipped per axis."""
    from sim.physics import Bodies, PROFILES, apply_controls
    b = Bodies(1, [PROFILES["WR"], PROFILES["DB"]])
    pos = np.zeros((1, 2, 2), np.float32)
    vel = np.zeros((1, 2, 2), np.float32)
    apply_controls(pos, vel, np.array([[[3.0, 1.0], [0.0, 0.0]]], np.float32), b, 0.1)
    v = vel[0, 0]
    assert np.isclose(np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arctan2(1.0, 3.0)), atol=0.5)


def test_vec_env_action_box_is_wide():
    from train.vec_env import ACTION_BOUND, SideVecEnv
    from sim.policies import angle_tackler
    env = SideVecEnv(4, SimConfig(), angle_tackler)
    assert env.action_space.high[0] == ACTION_BOUND >= 3


def test_breakaway_needs_the_runner_to_be_running_away():
    """A tackler slipping behind a runner who is standing still must not hand him a touchdown."""
    cfg = SimConfig(field=FieldConfig(breakaway_lead=2.0, full_play_share=0.0))
    sim = OneVOneSim(2, cfg, seed=15)
    sim.reset()
    sim.r_pos[:] = [[20.0, 26.0]] * 2
    sim.t_pos[:] = [[17.0, 30.0]] * 2      # 3 yd behind, off to the side
    _, _, done, info = sim.step(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))
    assert not info["breakaway"].any() and not done.any()


# ---------------------------------------------------------------- facing (EXP-008)
def _facing_sim(n=4, **field):
    sim = OneVOneSim(n, SimConfig(field=FieldConfig(**field), facing=True), seed=3)
    sim.reset()
    sim.t_pos[:] = (40.0, 5.0)   # tackler far away: no contact
    return sim


def test_facing_adds_an_action_and_observations():
    sim = _facing_sim()
    assert sim.act_dim == 3 and sim.observe().shape == (4, OBS_DIM + 4)
    assert np.allclose(sim.facing[:, 0], (1, 0)) and np.allclose(sim.facing[:, 1], (-1, 0))


def test_hips_turn_at_a_limited_rate():
    sim = _facing_sim()
    back = np.tile([-1.0, 0.0, 0.0], (4, 1)).astype(np.float32)     # push straight back, face it
    sim.step(back, np.zeros((4, 2)))
    turned = np.degrees(np.arccos(np.clip(sim.facing[:, 0, 0], -1, 1)))
    assert np.allclose(turned, sim.cfg.runner.turn_rate * 0.1, atol=1.0)  # 33.5 deg, not 180


def test_backpedal_is_slower_than_running():
    sim = _facing_sim(max_steps=1000)
    sim.r_pos[:] = (30.0, 26.0)
    sim.t_pos[:] = (55.0, 26.0)
    # Square up to the tackler (downfield at x=40) while pushing away from him: backpedal.
    away = np.tile([-1.0, 0.0, 1.0], (4, 1)).astype(np.float32)
    for _ in range(25):
        sim.step(away, np.zeros((4, 2)))
    back_speed = np.linalg.norm(sim.r_vel, axis=1)
    V = sim.cfg.runner.max_speed
    assert np.all(back_speed < sim.cfg.runner.back_speed * V + 0.05)
    assert np.all(back_speed > 0.9 * sim.cfg.runner.back_speed * V)


def test_square_up_faces_the_opponent_while_shuffling():
    sim = _facing_sim()
    sim.r_pos[:] = (30.0, 26.0)
    sim.t_pos[:] = (40.0, 26.0)
    side = np.tile([0.0, 1.0, 1.0], (4, 1)).astype(np.float32)        # shuffle sideways, hips square
    for _ in range(8):
        sim.step(side, np.zeros((4, 2)))
    to_t = sim.t_pos - sim.r_pos
    to_t /= np.linalg.norm(to_t, axis=1, keepdims=True)
    assert np.all(np.sum(sim.facing[:, 0] * to_t, axis=1) > 0.95)
    assert np.all(sim.r_vel[:, 1] > 3.0)


def test_opponent_facing_is_seen_late():
    sim = _facing_sim(reaction_delay=2)
    back = np.tile([-1.0, 0.0, 0.0], (4, 1)).astype(np.float32)
    sim.step(back, np.zeros((4, 2)))
    assert np.allclose(sim.seen_facing(0), (1, 0))       # tackler still sees the old hips
    assert not np.allclose(sim.facing[:, 0], (1, 0))

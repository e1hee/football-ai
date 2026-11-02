import numpy as np

from sim.core import OUTCOME_TACKLED, OUTCOME_TOUCHDOWN, OBS_DIM, OneVOneSim, SimConfig
from sim.policies import pursuit_tackler, straight_runner


def test_obs_shape_and_range():
    sim = OneVOneSim(8, seed=1)
    obs = sim.reset()
    assert obs.shape == (8, OBS_DIM)
    assert np.all(np.abs(obs) <= 1.5)


def test_straight_runner_scores_when_tackler_is_slow():
    cfg = SimConfig(tackler_speed=1.0, tackler_accel=1.0)
    sim = OneVOneSim(4, cfg, seed=2)
    sim.reset()
    # Put the tackler far to the side so a straight run never touches it.
    sim.t_pos[:, 1] = 1.0
    sim.r_pos[:, 1] = 40.0
    total = np.zeros(4, np.float32)
    for _ in range(cfg.field.max_steps):
        _, r, done, info = sim.step(straight_runner(sim), np.zeros((4, 2), np.float32))
        total += r
        if done.all():
            break
    assert np.all(info["outcome"] == OUTCOME_TOUCHDOWN)
    # Reward is yards gained: start at x=5, goal at 60 -> 55 yards.
    assert np.allclose(total, 55.0, atol=0.5)


def test_pursuit_tackler_catches_straight_runner_head_on():
    cfg = SimConfig(runner_speed=6.0, tackler_speed=9.0)
    sim = OneVOneSim(4, cfg, seed=3)
    sim.reset()
    sim.t_pos[:, 1] = sim.r_pos[:, 1]  # lined up directly in front
    for _ in range(cfg.field.max_steps):
        _, _, done, info = sim.step(straight_runner(sim), pursuit_tackler(sim))
        if done.all():
            break
    assert np.all(info["outcome"] == OUTCOME_TACKLED)


def test_auto_reset_keeps_running():
    sim = OneVOneSim(16, seed=4)
    sim.reset()
    for _ in range(300):
        obs, _, _, _ = sim.step(straight_runner(sim), pursuit_tackler(sim))
    assert obs.shape == (16, OBS_DIM)
    assert np.all(sim.t < sim.cfg.field.max_steps + 1)

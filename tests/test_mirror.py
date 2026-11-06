import numpy as np

from sim.core import FieldConfig
from sim.team import TeamSim, box_config
from train.mirror import DEFENSE, OFFENSE, mirror_action, mirror_obs
from train.selfplay import Frozen, Pool


def _flipped_pair():
    cfg = box_config(mix=True, field=FieldConfig(reaction_delay=2))
    a, b = TeamSim(32, cfg, seed=3), TeamSim(32, cfg, seed=4)
    a.reset(), b.reset()
    for _ in range(4):       # moving players, with histories that differ per slot
        a.step(np.random.default_rng(0).normal(size=(32, 18)).astype(np.float32),
               np.random.default_rng(1).normal(size=(32, 18)).astype(np.float32))
    perm = OFFENSE + [i + 6 for i in DEFENSE]
    inv = np.argsort(perm)
    flip = lambda v, by=0.0: np.stack([v[..., 0], by - v[..., 1]], -1)
    b.pos[...] = flip(a.pos[:, perm], cfg.field.width)
    b.vel[...] = flip(a.vel[:, perm])
    b.facing[...] = flip(a.facing[:, perm])
    b.stun[...] = a.stun[:, perm]
    part = a.partner[:, perm]
    b.partner[...] = np.where(part >= 0, inv[np.maximum(part, 0)], -1)
    b.wrap[...], b.t[...] = a.wrap, a.t
    for h_b, h_a, by in ((b._hist_pos, a._hist_pos, cfg.field.width), (b._hist_vel, a._hist_vel, 0.0),
                         (b._hist_face, a._hist_face, 0.0)):
        h_b[...] = flip(h_a[:, :, perm], by)
    return a, b


def test_mirror_obs_is_what_the_flipped_play_looks_like():
    a, b = _flipped_pair()
    assert np.allclose(mirror_obs(a.observe(), "runner"), b.observe(), atol=1e-5)
    assert np.allclose(mirror_obs(a.observe_tackler(), "tackler"), b.observe_tackler(), atol=1e-5)
    assert np.allclose(mirror_obs(mirror_obs(a.observe(), "runner"), "runner"), a.observe(), atol=1e-5)


def test_mirror_action_swaps_slots_and_flips_y():
    a = np.arange(18, dtype=np.float32)[None] + 1
    out = mirror_action(a, "runner").reshape(6, 3)
    assert out[0].tolist() == [1, -2, 3]        # RB stays in slot 0
    assert out[1].tolist() == [16, -17, 18]     # left tackle does what the right tackle chose
    assert np.array_equal(mirror_action(mirror_action(a, "tackler"), "tackler"), a)


class _Stub(Frozen):
    """Returns a fixed action and remembers the observations it was shown."""

    def __init__(self):
        self.side = "runner"

    def observe(self, sim):
        return sim.observe()

    def act(self, obs):
        self.seen = obs.copy()
        return np.tile(np.arange(18, dtype=np.float32) + 1, (len(obs), 1))


def test_pool_flips_only_the_flip_slots():
    sim = TeamSim(64, box_config(mix=True, field=FieldConfig(reaction_delay=2)), seed=0)
    sim.reset()
    stub = _Stub()
    pool = Pool(64, [stub], [1.0], seed=0, mirror_share=0.5)
    assert 10 < pool.flip.sum() < 54
    out = pool(sim)
    raw = np.arange(18, dtype=np.float32) + 1
    assert np.array_equal(out[~pool.flip], np.tile(raw, ((~pool.flip).sum(), 1)))
    assert np.array_equal(out[pool.flip], mirror_action(np.tile(raw, (pool.flip.sum(), 1)), "runner"))
    assert np.array_equal(stub.seen[pool.flip], mirror_obs(sim.observe()[pool.flip], "runner"))
    assert np.array_equal(stub.seen[~pool.flip], sim.observe()[~pool.flip])


def test_pool_survives_an_opponent_with_no_flipped_slots():
    # A late-generation pool gives old opponents a few slots each; all of them can be unflipped (EXP-024 crash).
    sim = TeamSim(8, box_config(mix=True, field=FieldConfig(reaction_delay=2)), seed=0)
    sim.reset()
    pool = Pool(8, [_Stub()], [1.0], seed=0, mirror_share=0.5)
    for flip in (np.zeros(8, bool), np.ones(8, bool)):
        pool.flip = flip
        assert pool(sim).shape == (8, 18)


def test_pool_without_mirror_is_unchanged():
    sim = TeamSim(16, box_config(mix=True, field=FieldConfig(reaction_delay=2)), seed=0)
    sim.reset()
    pool = Pool(16, [_Stub()], [1.0], seed=0)
    assert pool.flip is None
    assert np.array_equal(pool(sim), np.tile(np.arange(18, dtype=np.float32) + 1, (16, 1)))


def test_extra_opponents_take_their_share_from_the_newest():
    from train.selfplay import make_pool
    names = [object() for _ in range(6)]        # three generations, scripted, two extras
    pool = make_pool(200_000, names[:3], names[3], seed=0, scripted_share=0.3, extra=names[4:], extra_share=0.2)
    share = {id(o): (pool.assign == k).mean() for k, o in enumerate(pool.opponents)}
    assert abs(share[id(names[2])] - 0.4) < 0.01        # newest: 0.6 - 0.2
    assert abs(share[id(names[3])] - 0.3) < 0.01
    assert abs(share[id(names[4])] - 0.1) < 0.01 and abs(share[id(names[5])] - 0.1) < 0.01
    assert abs(share[id(names[0])] - 0.05) < 0.01       # the older ones keep their 0.1
    plain = make_pool(1000, names[:3], names[3], seed=0, scripted_share=0.3)
    assert len(plain.opponents) == 4


def test_a_young_offense_is_not_replaced():
    from train.selfplay import may_replace
    assert may_replace(9.0, 3.0, 1.0, age=4, min_age=4, last_phase=False)
    assert not may_replace(9.0, 3.0, 1.0, age=3, min_age=4, last_phase=False)
    assert not may_replace(9.0, 3.0, 1.0, age=9, min_age=4, last_phase=True)
    assert not may_replace(3.5, 3.0, 1.0, age=9, min_age=0, last_phase=False)
    assert may_replace(4.0, 3.0, 1.0, age=1, min_age=0, last_phase=False)     # EXP-027 behaviour


def test_reset_head_keeps_the_hidden_layers():
    from stable_baselines3 import PPO
    from scripts.recovery_test import reset_head
    model = PPO("MlpPolicy", "Pendulum-v1", device="cpu", seed=0, n_steps=64, batch_size=64)
    with __import__("torch").no_grad():
        model.policy.action_net.weight.fill_(1.0)
    hidden = [p.clone() for p in model.policy.mlp_extractor.parameters()]
    reset_head(model)
    w = model.policy.action_net.weight
    assert w.abs().max() < 0.05 and float(model.policy.action_net.bias.abs().max()) == 0.0
    assert all((a == b).all() for a, b in zip(hidden, model.policy.mlp_extractor.parameters()))


def test_bounded_push_policy_keeps_the_push_short_and_survives_save(tmp_path):
    import torch
    from stable_baselines3 import PPO
    from train.policies import BoundedPushPolicy, policy_for, squash
    x = torch.tensor([[30.0, 40.0, 9.0, 0.3, -0.4, -0.1]])        # two players: one long push, one short
    y = squash(x, 1.25)
    assert abs(float(y[0, :2].norm()) - 1.25) < 1e-4 and torch.allclose(y[0, :2] / y[0, :2].norm(), x[0, :2] / 50)
    assert float(y[0, 2]) <= 1.25 and torch.allclose(y[0, 3:], x[0, 3:], atol=0.05)
    policy, kw = policy_for(1.25, 16)
    model = PPO(policy, "Pendulum-v1", device="cpu", seed=0, n_steps=64, batch_size=64, policy_kwargs=kw)
    with torch.no_grad():
        model.policy.action_net.weight.fill_(100.0)
    obs = torch.ones(4, 3)
    mean = model.policy.get_distribution(obs).distribution.mean
    assert 1.2 < float(mean.abs().max()) <= 1.25 + 1e-6
    model.save(tmp_path / "m.zip")
    back = PPO.load(tmp_path / "m.zip", device="cpu")
    assert isinstance(back.policy, BoundedPushPolicy) and back.policy.push_bound == 1.25
    assert torch.allclose(back.policy.get_distribution(obs).distribution.mean, mean)
    assert policy_for(None, 16)[0] == "MlpPolicy"


def test_sde_noise_is_held_calibrated_and_frozen_opponents_can_use_it(tmp_path):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from train.policies import action_std, calibrate_sde, policy_for, resample_noise
    from train.selfplay import Frozen, StdFloor
    policy, kw = policy_for(1.25, 16)
    model = PPO(policy, "Pendulum-v1", device="cpu", seed=0, n_steps=64, batch_size=64, policy_kwargs=kw,
                use_sde=True, sde_sample_freq=-1)
    obs = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    log0 = calibrate_sde(model.policy, obs, 0.7)
    assert abs(float(action_std(model.policy, obs).mean()) - 0.7) < 1e-4 and model.policy.log_std0 == log0
    resample_noise(model.policy, 32)
    a1, a2 = (model.predict(obs, deterministic=False)[0] for _ in range(2))
    assert np.allclose(a1, a2)                                   # same state, same held noise
    resample_noise(model.policy, 32)
    assert not np.allclose(model.predict(obs, deterministic=False)[0], a1)
    model.save(tmp_path / "m.zip")
    frozen = Frozen(tmp_path / "m.zip", "runner")
    b1 = frozen.act(obs)
    assert b1.shape == (32, 1) and np.allclose(frozen.act(obs), b1)
    assert frozen.act(obs[:5]).shape == (5, 1)                    # a new slot count gets its own noise
    with torch.no_grad():
        model.policy.log_std.fill_(log0 - 5.0)
    floor = StdFloor(0.8)
    floor.init_callback(model)
    floor.on_rollout_start()
    assert abs(float(model.policy.log_std.min()) - (log0 + np.log(0.8))) < 1e-5


def test_to_sde_keeps_the_trained_mean(tmp_path):
    import argparse
    import numpy as np
    from stable_baselines3 import PPO
    from scripts.recovery_test import to_sde
    from train.policies import action_std, policy_for
    policy, kw = policy_for(1.25, 16)
    old = PPO(policy, "Pendulum-v1", device="cpu", seed=0, n_steps=64, batch_size=64, policy_kwargs=kw)
    obs = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    new = to_sde(old, old.get_env(), argparse.Namespace(seed=0, sde_freq=-1, sde_std=0.3), obs)
    assert new.use_sde and np.allclose(new.predict(obs, deterministic=True)[0], old.predict(obs, deterministic=True)[0])
    assert abs(float(action_std(new.policy, obs).mean()) - 0.3) < 1e-4

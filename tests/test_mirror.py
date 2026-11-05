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

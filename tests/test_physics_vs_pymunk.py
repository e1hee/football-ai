"""Cross-check our numpy collision response against a real 2D physics engine.

Pymunk (Chipmunk2D) is too slow to train with (one world at a time), but it
is a trustworthy referee: for the same two discs, masses and restitution,
the velocities after impact should match.
"""
import numpy as np
import pymunk
import pytest

from sim.physics import Bodies, Profile, resolve_collisions

E = 0.1  # restitution used by the sim


def _pymunk_collide(m1, m2, r, p1, p2, v1, v2):
    space = pymunk.Space()
    space.gravity = (0, 0)
    out = []
    for m, p, v in ((m1, p1, v1), (m2, p2, v2)):
        body = pymunk.Body(m, pymunk.moment_for_circle(m, 0, r))
        body.position, body.velocity = tuple(p), tuple(v)
        shape = pymunk.Circle(body, r)
        shape.elasticity = np.sqrt(E)   # pymunk multiplies the two shapes' values
        shape.friction = 0.0
        space.add(body, shape)
        out.append(body)
    for _ in range(400):
        space.step(0.001)
    # Rotation can't carry energy without friction, so linear velocity is comparable.
    return np.array(out[0].velocity), np.array(out[1].velocity)


def _ours(m1, m2, r, p1, p2, v1, v2):
    prof = lambda m: Profile(10, 6, 2, 6, 3, 6, 0.2, m, radius=r)
    b = Bodies(1, [prof(m1), prof(m2)])
    pos = np.array([[p1, p2]], np.float32)
    vel = np.array([[v1, v2]], np.float32)
    # Advance until contact, then resolve once (what the sim does each step).
    for _ in range(400):
        pos += vel * 0.001
        if resolve_collisions(pos, vel, b, restitution=E).any():
            break
    return vel[0, 0], vel[0, 1]


@pytest.mark.parametrize("case", [
    # head-on, equal-ish mass (runner vs DB)
    dict(m1=198, m2=194, p1=(0, 0), p2=(1.0, 0), v1=(8, 0), v2=(-6, 0)),
    # big lineman stops a receiver
    dict(m1=198, m2=315, p1=(0, 0), p2=(1.0, 0), v1=(8, 0), v2=(-1, 0)),
    # glancing hit from the side
    dict(m1=198, m2=194, p1=(0, 0), p2=(0.95, 0.5), v1=(9, 0), v2=(0, -5)),
])
def test_collision_matches_pymunk(case):
    r = 0.45
    pm1, pm2 = _pymunk_collide(r=r, **case)
    us1, us2 = _ours(r=r, **case)
    np.testing.assert_allclose(us1, pm1, atol=0.02)
    np.testing.assert_allclose(us2, pm2, atol=0.02)
    # Momentum is conserved either way.
    p_before = case["m1"] * np.array(case["v1"]) + case["m2"] * np.array(case["v2"])
    np.testing.assert_allclose(case["m1"] * us1 + case["m2"] * us2, p_before, rtol=1e-3, atol=1.0)

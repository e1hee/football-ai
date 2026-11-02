"""Vectorized player physics, shared by every sim size (1v1, 3v3, 11v11).

Arrays are shaped (N plays, P players, ...). Three pieces:

1. `apply_controls`: a player chooses where to push (action in the unit
   disk). How hard they can push depends on their current speed and on the
   direction relative to their motion, using curves fitted to real NFL
   tracking (see `analysis/physics_from_bdb.py`):
     - forward:  accel(v) = A0 (1 - (v/V)^k)   -> fast to start, nothing left at top speed
     - braking:  brake(v) = B0 (1 - (v/V)^kb)
     - sideways: lateral(v) = L0 (1 - c (v/V)^2) -> sharp cuts only at low speed
   Turning at speed therefore costs: the lateral cap bends the path slowly,
   and slowing down first to cut harder is a real trade-off.

2. `resolve_collisions`: players are discs that can't overlap. Overlaps are
   pushed apart by inverse mass, and the approaching velocity component is
   exchanged as an inelastic impulse (bodies thud, they don't bounce).

3. Tackles live in the sim that knows who is who (`core.py`); this module
   only provides contact detection and the geometry tackles need.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PARAMS_PATH = Path(__file__).with_name("physics_params.json")
LB_TO_KG = 0.4536


@dataclass(frozen=True)
class Profile:
    """Physical profile of one kind of player. Units: yd, s, lb."""
    max_speed: float
    accel: float
    accel_k: float
    brake: float
    brake_k: float
    agility: float
    agility_drop: float
    mass_lb: float
    radius: float = 0.45  # yd; ~shoulder half-width plus pads

    def scaled(self, **mult: float) -> "Profile":
        """Copy with some fields multiplied, e.g. scaled(max_speed=0.9)."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, m in mult.items():
            d[k] = d[k] * m
        return Profile(**d)


def load_profiles(path: Path = PARAMS_PATH) -> dict[str, Profile]:
    fit = json.loads(path.read_text())["fit"]
    return {g: Profile(**{k: v for k, v in p.items() if k != "rmse"}) for g, p in fit.items()}


# Fallback numbers (from the 2024 fit) so the sim runs without the json.
DEFAULT_PROFILES = {
    "WR": Profile(10.02, 6.75, 2.35, 6.9, 3.4, 6.95, 0.16, 198),
    "DB": Profile(9.65, 5.95, 2.7, 6.25, 3.95, 7.0, 0.28, 194),
}
try:
    PROFILES = {**DEFAULT_PROFILES, **load_profiles()}
except (FileNotFoundError, KeyError):
    PROFILES = DEFAULT_PROFILES


class Bodies:
    """Per-player physical attributes as (N, P) arrays, so every play and
    every player can differ (roster experiments edit these)."""

    FIELDS = ("max_speed", "accel", "accel_k", "brake", "brake_k", "agility", "agility_drop", "mass_lb", "radius")

    def __init__(self, n: int, profiles: list[Profile]):
        self.n, self.p = n, len(profiles)
        for f in self.FIELDS:
            row = np.array([getattr(pr, f) for pr in profiles], np.float32)
            setattr(self, f, np.tile(row, (n, 1)))

    def set_profile(self, idx: int, profile: Profile, mask: np.ndarray | None = None) -> None:
        sel = slice(None) if mask is None else mask
        for f in self.FIELDS:
            getattr(self, f)[sel, idx] = getattr(profile, f)


def _caps(speed: np.ndarray, b: Bodies):
    r = np.clip(speed / b.max_speed, 0.0, 1.0)
    fwd = b.accel * np.maximum(0.0, 1.0 - r ** b.accel_k)
    brk = b.brake * np.maximum(0.15, 1.0 - r ** b.brake_k)  # you can always slow down a little
    lat = b.agility * np.maximum(0.2, 1.0 - b.agility_drop * r ** 2)
    return fwd, brk, lat


def apply_controls(pos, vel, action, b: Bodies, dt: float, stunned=None):
    """Advance positions/velocities in place.

    action: (N, P, 2) desired push direction and strength, clipped to the unit disk.
    stunned: optional (N, P) bool; stunned players get no control (they coast
    and decelerate, e.g. after whiffing a tackle).
    """
    a = np.asarray(action, np.float32)
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a = np.where(norm > 1.0, a / np.maximum(norm, 1e-6), a)

    speed = np.linalg.norm(vel, axis=-1)
    fwd_cap, brk_cap, lat_cap = _caps(speed, b)

    # Basis along / across current motion. Standing still: "along" = wherever you push.
    moving = speed > 0.3
    push_dir = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-6)
    u = np.where(moving[..., None], vel / np.maximum(speed, 1e-6)[..., None], push_dir)
    n_ = np.stack([-u[..., 1], u[..., 0]], axis=-1)

    a_t = np.sum(a * u, axis=-1)      # in [-1, 1]
    a_l = np.sum(a * n_, axis=-1)
    acc_t = np.where(a_t >= 0, a_t * fwd_cap, a_t * brk_cap)
    acc_l = a_l * lat_cap
    acc = acc_t[..., None] * u + acc_l[..., None] * n_

    if stunned is not None and stunned.any():
        # No control: friction slows them at a third of their braking power.
        coast = -u * (brk_cap / 3.0)[..., None] * moving[..., None]
        acc = np.where(stunned[..., None], coast, acc)

    vel += acc * dt
    new_speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    vel *= np.where(new_speed > b.max_speed[..., None], b.max_speed[..., None] / np.maximum(new_speed, 1e-6), 1.0)
    pos += vel * dt


def resolve_collisions(pos, vel, b: Bodies, restitution: float = 0.1):
    """Separate overlapping discs and exchange momentum along the contact normal.

    Returns contact (N, P, P) bool: which pairs touched this step (symmetric).
    O(P^2) per play, vectorized across plays; fine up to 22 players.
    """
    n, p = pos.shape[:2]
    d = pos[:, None, :, :] - pos[:, :, None, :]          # (N, P, P, 2): from i to j
    dist = np.linalg.norm(d, axis=-1)
    rsum = b.radius[:, :, None] + b.radius[:, None, :]
    eye = np.eye(p, dtype=bool)[None]
    contact = (dist < rsum) & ~eye
    if not contact.any():
        return contact

    m = b.mass_lb
    inv_m = 1.0 / m
    normal = d / np.maximum(dist, 1e-6)[..., None]      # unit i -> j

    # Only handle each pair once (i < j), then apply to both.
    upper = np.triu(np.ones((p, p), bool), 1)[None] & contact
    ii, jj, kk = np.nonzero(upper.transpose(1, 2, 0))   # player i, player j, play k
    if ii.size == 0:
        return contact
    nrm = normal[kk, ii, jj]                             # (C, 2)
    overlap = (rsum[kk, ii, jj] - dist[kk, ii, jj])
    wi, wj = inv_m[kk, ii], inv_m[kk, jj]
    w = wi + wj

    # Positional correction, split by inverse mass (heavier moves less).
    np.add.at(pos, (kk, ii), -nrm * (overlap * wi / w)[:, None])
    np.add.at(pos, (kk, jj), nrm * (overlap * wj / w)[:, None])

    # Impulse only if they are approaching along the normal.
    v_rel = vel[kk, jj] - vel[kk, ii]
    vn = np.sum(v_rel * nrm, axis=-1)
    j = np.where(vn < 0, -(1.0 + restitution) * vn / w, 0.0)
    np.add.at(vel, (kk, ii), -nrm * (j * wi)[:, None])
    np.add.at(vel, (kk, jj), nrm * (j * wj)[:, None])
    return contact

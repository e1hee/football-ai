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
   Optional body facing (hips), from NFL tracking `o` (see
   `analysis/orientation_from_bdb.py`): facing turns at a limited rate, and
   pushing or moving away from where you face is weaker / slower (backpedal).

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
    # Body facing (only used when the sim turns facing on). Defaults: DB.
    back_speed: float = 0.64     # top speed moving straight backwards / forwards
    back_accel: float = 0.78     # push strength straight backwards / forwards
    turn_rate: float = 250.0     # deg/s the hips can turn when slow ...
    turn_rate_top: float = 150.0 # ... and at top speed

    def scaled(self, **mult: float) -> "Profile":
        """Copy with some fields multiplied, e.g. scaled(max_speed=0.9)."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, m in mult.items():
            d[k] = d[k] * m
        return Profile(**d)


ORIENT_PATH = Path(__file__).with_name("orientation_params.json")


def facing_params(g: dict) -> dict:
    """Facing limits from one group's `orientation_from_bdb` aggregates (p99s)."""
    acc = list(g["acc_p99_by_angle_slow"].values())
    turn = list(g["o_turn_p99_by_speed"].values())
    return {
        "back_speed": round(g["speed_p99_off_135_181"] / g["speed_p99_off_0_45"], 3),
        "back_accel": round(acc[-1] / max(acc), 3),
        "turn_rate": float(max(turn[:5])),      # slower than 5 yd/s
        "turn_rate_top": float(turn[-1]),       # 8+ yd/s
    }


def load_profiles(path: Path = PARAMS_PATH, orient_path: Path = ORIENT_PATH) -> dict[str, Profile]:
    fit = json.loads(path.read_text())["fit"]
    orient = json.loads(orient_path.read_text())["groups"] if orient_path.exists() else {}
    return {g: Profile(**{k: v for k, v in p.items() if k != "rmse"},
                       **(facing_params(orient[g]) if g in orient else {}))
            for g, p in fit.items()}


# Fallback numbers (from the 2024 fit) so the sim runs without the json.
DEFAULT_PROFILES = {
    "WR": Profile(10.02, 6.75, 2.35, 6.9, 3.4, 6.95, 0.16, 198,
                  back_speed=0.78, back_accel=0.87, turn_rate=335, turn_rate_top=174),
    "DB": Profile(9.65, 5.95, 2.7, 6.25, 3.95, 7.0, 0.28, 194),
}
try:
    PROFILES = {**DEFAULT_PROFILES, **load_profiles()}
except (FileNotFoundError, KeyError):
    PROFILES = DEFAULT_PROFILES
# No running back is tracked in the BDB 2027 data (pass plays; 1 RB in 2024), so
# the RB moves like a WR but carries a RB's weight (combine median, 210 lb).
PROFILES["RB"] = PROFILES["WR"].scaled(mass_lb=210 / PROFILES["WR"].mass_lb)
# Pass-play tracking undercounts how fast linemen can run (they pass-protect,
# rarely sprint): the fitted OL needs 7.05 s for 40 yd, 1.37x its combine 40
# (5.15 s), while WR/DB/TE fits run 1.14x theirs. Scale OL top speed and
# acceleration so its sim 10/40-yd times sit at the same ratio to the combine
# (2.23 s / 5.86 s; grid search in the EXP-009 notes). Needed for screens.
if "OL" in PROFILES:
    PROFILES["OL"] = PROFILES["OL"].scaled(max_speed=1.28, accel=1.10)
# DL rush the passer, so they are tracked sprinting more than OL, but the fit
# still runs 40 yd in 5.87 s, 1.19x a DL combine 40 (~4.95 s, DE ~4.8 / DT ~5.05).
# Same 1.14x target as the others: 1.96 s / 5.62 s for 10 / 40 yd.
if "DL" in PROFILES:
    PROFILES["DL"] = PROFILES["DL"].scaled(max_speed=1.03, accel=1.16)


class Bodies:
    """Per-player physical attributes as (N, P) arrays, so every play and
    every player can differ (roster experiments edit these)."""

    FIELDS = ("max_speed", "accel", "accel_k", "brake", "brake_k", "agility", "agility_drop", "mass_lb", "radius",
              "back_speed", "back_accel", "turn_rate", "turn_rate_top")

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


def _away(dir_a, dir_b):
    """0 when two unit vectors agree, 1 when opposite."""
    return np.clip((1.0 - np.sum(dir_a * dir_b, axis=-1)) / 2.0, 0.0, 1.0)


def turn_facing(facing, target, speed, b: Bodies, dt: float, stunned=None):
    """Rotate unit `facing` (N, P, 2) toward `target` directions in place, at most
    turn_rate(v) * dt. Zero-length targets (or stunned players) keep facing."""
    t_norm = np.linalg.norm(target, axis=-1)
    tgt = target / np.maximum(t_norm, 1e-6)[..., None]
    ang = np.arctan2(facing[..., 0] * tgt[..., 1] - facing[..., 1] * tgt[..., 0],
                     np.sum(facing * tgt, axis=-1))                     # signed, radians
    r = np.clip(speed / b.max_speed, 0.0, 1.0)
    rate = np.deg2rad(b.turn_rate - (b.turn_rate - b.turn_rate_top) * r) * dt
    step = np.clip(ang, -rate, rate) * (t_norm > 1e-3)
    if stunned is not None:
        step = np.where(stunned, 0.0, step)
    c, s = np.cos(step), np.sin(step)
    fx, fy = facing[..., 0].copy(), facing[..., 1].copy()
    facing[..., 0] = c * fx - s * fy
    facing[..., 1] = s * fx + c * fy
    facing /= np.maximum(np.linalg.norm(facing, axis=-1, keepdims=True), 1e-6)


def apply_controls(pos, vel, action, b: Bodies, dt: float, stunned=None, facing=None):
    """Advance positions/velocities in place.

    action: (N, P, 2) desired push direction and strength, clipped to the unit disk.
    stunned: optional (N, P) bool; stunned players get no control (they coast
    and decelerate, e.g. after whiffing a tackle).
    facing: optional (N, P, 2) unit hips direction. Pushing away from it is
    weaker (down to `back_accel`), and moving away from it has a lower top
    speed (down to `back_speed` when backpedalling); above that cap you slow
    down at braking rate rather than stopping dead.
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
    if facing is not None:
        acc *= (1.0 - (1.0 - b.back_accel) * _away(push_dir, facing))[..., None]

    if stunned is not None and stunned.any():
        # No control: friction slows them at a third of their braking power.
        coast = -u * (brk_cap / 3.0)[..., None] * moving[..., None]
        acc = np.where(stunned[..., None], coast, acc)

    vel += acc * dt
    new_speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    vel *= np.where(new_speed > b.max_speed[..., None], b.max_speed[..., None] / np.maximum(new_speed, 1e-6), 1.0)
    if facing is not None:
        sp = np.linalg.norm(vel, axis=-1)
        v_dir = vel / np.maximum(sp, 1e-6)[..., None]
        cap = b.max_speed * (1.0 - (1.0 - b.back_speed) * _away(v_dir, facing) ** 0.8)
        allowed = np.maximum(cap, speed - brk_cap * dt)
        vel *= np.where(sp > allowed, allowed / np.maximum(sp, 1e-6), 1.0)[..., None]
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

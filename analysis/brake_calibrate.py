"""Calibrate how much braking power players keep at speed (`brake_drop`) so the
sim's best 0.5 s speed shed matches real NFL tracking.

Why: `physics_from_bdb` fits brake(v) to per-frame p99 decelerations. Hard stops
at high speed are rare, so per-frame p99 there undercounts what big players can
do over a real stop: DL from 7-9 yd/s shed 2.1 yd/s in 0.5 s (p99) but the fitted
sim only 0.4-1.6. This keeps each group's B0 and kb and grid-searches d in
brake(v) = B0 (1 - d (v/V)^kb) against the 0.5 s p99 in the 3-5 / 5-7 / 7-9 yd/s
bins. Aggregates only (BDB is CC BY-NC).

Run after `analysis.physics_from_bdb` (which resets brake_drop to its frame fit):
    uv run python -m analysis.brake_calibrate
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

from analysis.physics_from_bdb import OUT, load
from sim.physics import PROFILES, Bodies, apply_controls

BINS = ((3, 5), (5, 7), (7, 9))
MIN_N = 1000


def nfl_shed(df) -> dict:
    """p99 speed shed over 0.5 s, per group and starting-speed bin."""
    key = df["game_id"].to_numpy() * 10_000_000 + df["play_id"].to_numpy() * 100_000 + df["nfl_id"].to_numpy() % 100_000
    s, g, t = df["s"].to_numpy(), df["group"].to_numpy(), df["t"].to_numpy()
    k = 5
    ok = (key[k:] == key[:-k]) & (np.abs(t[k:] - t[:-k] - 0.5) < 0.03)
    s0, drop, gg = s[:-k][ok], (s[:-k] - s[k:])[ok], g[:-k][ok]
    out = {}
    for grp in np.unique(gg):
        for lo, hi in BINS:
            m = (gg == grp) & (s0 >= lo) & (s0 < hi)
            if m.sum() >= MIN_N:
                out.setdefault(grp, {})[f"{lo}-{hi}"] = (round(float(np.quantile(drop[m], 0.99)), 2), int(m.sum()))
    return out


def sim_shed(profile, v: float) -> float:
    """Best speed shed in 0.5 s from speed v: full brake straight back."""
    b = Bodies(1, [profile])
    pos = np.zeros((1, 1, 2), np.float32)
    vel = np.array([[[v, 0.0]]], np.float32)
    for _ in range(5):
        a = (-vel / np.maximum(np.linalg.norm(vel, axis=-1, keepdims=True), 1e-6)).astype(np.float32)
        apply_controls(pos, vel, a, b, 0.1)
    return v - float(np.linalg.norm(vel))


def calibrate(profile, target: dict) -> tuple[float, dict]:
    best = (np.inf, 1.0)
    for d in np.arange(0.0, 1.001, 0.02):
        p = dataclasses.replace(profile, brake_drop=float(d))
        err = sum((sim_shed(p, (lo + hi) / 2) - target[f"{lo}-{hi}"][0]) ** 2
                  for lo, hi in BINS if f"{lo}-{hi}" in target)
        best = min(best, (err, round(float(d), 2)))
    d = best[1]
    p = dataclasses.replace(profile, brake_drop=d)
    return d, {b: round(sim_shed(p, sum(map(float, b.split("-"))) / 2), 2) for b in target}


def main() -> None:
    target = nfl_shed(load(2024))
    res = json.loads(OUT.read_text())
    table = {}
    for grp, tgt in sorted(target.items()):
        if grp not in res["fit"]:
            continue
        frame_d = res["fit"][grp]["brake_drop"]
        before = {b: round(sim_shed(dataclasses.replace(PROFILES[grp], brake_drop=1.0), sum(map(float, b.split("-"))) / 2), 2)
                  for b in tgt}
        d, after = calibrate(PROFILES[grp], tgt)
        res["fit"][grp]["brake_drop"] = d
        table[grp] = {"nfl_p99": {b: v[0] for b, v in tgt.items()}, "n": {b: v[1] for b, v in tgt.items()},
                      "sim_old_form": before, "sim_calibrated": after, "frame_fit_drop": frame_d, "brake_drop": d}
        print(f"{grp}: brake_drop {frame_d} (frame fit) -> {d} | NFL {table[grp]['nfl_p99']} | "
              f"old {before} | now {after}")
    res["brake_calibration"] = {"what": "best 0.5 s speed shed (yd/s) by starting-speed bin; NFL = p99 of real windows",
                                "groups": table}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

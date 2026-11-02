"""Derive movement limits for the sim from real NFL game tracking (BDB 2027).

Reads the sibling project's parquet (never copied here: CC BY-NC) and writes
only aggregate numbers to `sim/physics_params.json`.

For every tracked frame we rebuild the velocity vector from speed `s` and
direction `dir`, difference it over 0.1 s, and split the acceleration into
  - tangential (along the run: speeding up / braking) and
  - lateral (sideways: what it takes to curve or cut).
Then per position group and speed bin we take high percentiles: what the
best ~1% of frames reach is the body's limit, which is what the sim caps.

    uv run python -m analysis.physics_from_bdb --season 2024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BDB = Path.home() / "projects/big-data-bowl-2027/data"
OUT = Path(__file__).resolve().parents[1] / "sim/physics_params.json"
GROUPS = {"WR": "WR", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
          "TE": "TE", "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
          "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
          "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL"}
SPEED_BINS = np.arange(0, 11, 1.0)   # yd/s


def load(season: int) -> pd.DataFrame:
    df = pd.read_parquet(BDB / f"parquet/game_tracking_{season}.parquet",
                         columns=["game_id", "play_id", "nfl_id", "time", "s", "dir"])
    players = pd.read_csv(BDB / "nfl-big-data-bowl-2027/players.csv", usecols=["nfl_id", "nfl_position"])
    combine = pd.read_csv(BDB / "nfl-big-data-bowl-2027/combine_results.csv", usecols=["nfl_id", "combine_weight"])
    df = df.merge(players, on="nfl_id").merge(combine, on="nfl_id", how="left")
    df["group"] = df["nfl_position"].map(GROUPS)
    df = df.dropna(subset=["group", "s", "dir"])
    ts = pd.to_datetime(df["time"])
    df["t"] = (ts - ts.min()).dt.total_seconds()  # unit-agnostic (pandas may parse to us or ns)
    return df.sort_values(["game_id", "play_id", "nfl_id", "t"]).reset_index(drop=True)


def kinematics(df: pd.DataFrame) -> pd.DataFrame:
    # NFL `dir`: degrees clockwise from +y.
    rad = np.deg2rad(df["dir"].to_numpy())
    s = df["s"].to_numpy()
    vx, vy = s * np.sin(rad), s * np.cos(rad)
    key = df["game_id"].to_numpy() * 10_000_000 + df["play_id"].to_numpy() * 100_000 + df["nfl_id"].to_numpy() % 100_000
    same = np.r_[False, key[1:] == key[:-1]]
    dt = np.r_[np.nan, np.diff(df["t"].to_numpy())]
    ok = same & (np.abs(dt - 0.1) < 0.02)

    ax = np.r_[np.nan, np.diff(vx)] / 0.1
    ay = np.r_[np.nan, np.diff(vy)] / 0.1
    # Project onto the mean velocity direction over the interval.
    mvx = (vx + np.r_[np.nan, vx[:-1]]) / 2
    mvy = (vy + np.r_[np.nan, vy[:-1]]) / 2
    mv = np.hypot(mvx, mvy)
    ux, uy = mvx / np.maximum(mv, 1e-6), mvy / np.maximum(mv, 1e-6)
    a_tan = ax * ux + ay * uy
    a_lat = np.abs(-ax * uy + ay * ux)
    # Heading change rate (deg/s), only meaningful while actually moving.
    d_dir = np.r_[np.nan, np.diff(df["dir"].to_numpy())]
    d_dir = (d_dir + 180) % 360 - 180
    turn = np.abs(d_dir) / 0.1

    out = pd.DataFrame({
        "group": df["group"].to_numpy(), "weight": df["combine_weight"].to_numpy(),
        "pid": key, "s": s, "s_mid": mv, "a_tan": a_tan, "a_lat": a_lat, "turn": turn,
    })[ok]
    # Physically implausible frames (sensor glitches) out: > 12 yd/s or > 15 yd/s^2.
    return out[(out["s"] <= 12) & (np.hypot(out["a_tan"], out["a_lat"]) <= 15)]


def summarise(k: pd.DataFrame) -> dict:
    res: dict = {}
    for g, d in k.groupby("group"):
        top = d.groupby("pid")["s"].max()
        bins = pd.cut(d["s_mid"], SPEED_BINS, right=False)
        by_bin = d.groupby(bins, observed=True)
        accel = by_bin["a_tan"].quantile(0.99)
        brake = (-by_bin["a_tan"].quantile(0.01))
        lateral = by_bin["a_lat"].quantile(0.99)
        moving = d[d["s_mid"] > 2]
        turn_bins = pd.cut(moving["s_mid"], SPEED_BINS, right=False)
        turn99 = moving.groupby(turn_bins, observed=True)["turn"].quantile(0.99)
        res[g] = {
            "frames": int(len(d)),
            "player_plays": int(top.size),
            "weight_lb_median": float(np.nanmedian(d["weight"])),
            "top_speed_p50": round(float(top.quantile(0.5)), 2),
            "top_speed_p99": round(float(top.quantile(0.99)), 2),
            "top_speed_max": round(float(top.max()), 2),
            "speed_bins_yd_s": [float(b.left) for b in accel.index],
            "accel_p99_by_speed": [round(float(v), 2) for v in accel.to_numpy()],
            "brake_p99_by_speed": [round(float(v), 2) for v in brake.to_numpy()],
            "lateral_p99_by_speed": [round(float(v), 2) for v in lateral.to_numpy()],
            "turn_deg_s_p99_by_speed": [round(float(v), 1) for v in turn99.to_numpy()],
        }
    return res


def _grid_fit(v: np.ndarray, y: np.ndarray, f, grids: list[np.ndarray]):
    """Brute-force least squares over a parameter grid (2 params, tiny data)."""
    best, arg = np.inf, None
    for p0 in grids[0]:
        for p1 in grids[1]:
            err = np.nansum((f(v, p0, p1) - y) ** 2)
            if err < best:
                best, arg = err, (p0, p1)
    return arg, float(np.sqrt(best / np.isfinite(y).sum()))


def fit_profile(r: dict) -> dict:
    """Fit smooth curves the sim can evaluate per step:
        accel(v)   = A0 * (1 - (v/V)^k)        forward push left at speed v
        brake(v)   = B0 * (1 - (v/V)^kb)       stopping power
        lateral(v) = L0 * (1 - c (v/V)^2)      sideways grip = how hard you can cut
    V sits just above the p99 top speed so the accel curve reaches 0 there."""
    V = r["top_speed_p99"] + 0.3
    edges = np.array(r["speed_bins_yd_s"])
    keep = (edges >= 1) & (edges + 0.5 < V)
    v = edges[keep] + 0.5
    acc = np.array(r["accel_p99_by_speed"])[keep]
    brk = np.array(r["brake_p99_by_speed"])[keep]
    lat = np.array(r["lateral_p99_by_speed"])[keep]
    power = lambda v, a0, k: a0 * (1 - (v / V) ** k)
    (A0, k), acc_rmse = _grid_fit(v, acc, power, [np.arange(3, 11, 0.05), np.arange(0.5, 4, 0.05)])
    (B0, kb), brk_rmse = _grid_fit(v, brk, power, [np.arange(3, 11, 0.05), np.arange(0.5, 4, 0.05)])
    (L0, c), lat_rmse = _grid_fit(v, lat, lambda v, l0, c: l0 * (1 - c * (v / V) ** 2),
                                  [np.arange(2, 10, 0.05), np.arange(0, 0.9, 0.01)])
    return {"max_speed": round(V, 2), "accel": round(float(A0), 2), "accel_k": round(float(k), 2),
            "brake": round(float(B0), 2), "brake_k": round(float(kb), 2),
            "agility": round(float(L0), 2), "agility_drop": round(float(c), 2),
            "mass_lb": round(r["weight_lb_median"]),
            "rmse": {"accel": round(acc_rmse, 2), "brake": round(brk_rmse, 2), "lateral": round(lat_rmse, 2)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    args = ap.parse_args()
    df = load(args.season)
    k = kinematics(df)
    res = {"source": f"NFL Big Data Bowl 2027 game tracking, REG {args.season}, aggregates only",
           "units": "yd, yd/s, yd/s^2, deg/s; percentiles over 0.1 s frames",
           "groups": summarise(k)}
    res["fit"] = {g: fit_profile(r) for g, r in res["groups"].items()}
    OUT.write_text(json.dumps(res, indent=2))
    print("fitted profiles:")
    for g, p in res["fit"].items():
        print(f"  {g}: {p}")
    for g, r in res["groups"].items():
        print(f"\n{g}: {r['player_plays']:,} player-plays, {r['frames']:,} frames, ~{r['weight_lb_median']:.0f} lb")
        print(f"  top speed p50 {r['top_speed_p50']}  p99 {r['top_speed_p99']}  max {r['top_speed_max']}")
        print("  speed  accel  brake  lateral  turn")
        for i, v in enumerate(r["speed_bins_yd_s"]):
            tv = r["turn_deg_s_p99_by_speed"][i - 2] if i >= 2 and i - 2 < len(r["turn_deg_s_p99_by_speed"]) else float("nan")
            print(f"  {v:4.0f}  {r['accel_p99_by_speed'][i]:6.2f} {r['brake_p99_by_speed'][i]:6.2f} "
                  f"{r['lateral_p99_by_speed'][i]:7.2f}  {tv:6.0f}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

"""How much does body orientation (hips) limit movement? From NFL tracking `o`.

`o` is where the player's body faces, `dir` where he is moving (both degrees
clockwise from +y). Per position group we measure:
  - off-angle |o - dir| by speed: how often players move sideways/backwards
    relative to where they face (shuffle, backpedal), and how fast they can;
  - how fast `o` itself can turn (hip turn rate, p99) by speed;
  - how hard players can accelerate depending on the angle between the
    acceleration and their facing (p99 by angle bin, low speed only).
Aggregates only (BDB is CC BY-NC); prints a table and writes
`sim/orientation_params.json`.

    uv run python -m analysis.orientation_from_bdb --season 2024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.physics_from_bdb import BDB, GROUPS

OUT = Path(__file__).resolve().parents[1] / "sim/orientation_params.json"


def wrap(d):
    return (d + 180) % 360 - 180


def load(season: int) -> pd.DataFrame:
    df = pd.read_parquet(BDB / f"parquet/game_tracking_{season}.parquet",
                         columns=["game_id", "play_id", "nfl_id", "time", "s", "dir", "o"])
    players = pd.read_csv(BDB / "nfl-big-data-bowl-2027/players.csv", usecols=["nfl_id", "nfl_position"])
    df = df.merge(players, on="nfl_id")
    df["group"] = df["nfl_position"].map(GROUPS)
    df = df.dropna(subset=["group", "s", "dir", "o"])
    ts = pd.to_datetime(df["time"])
    df["t"] = (ts - ts.min()).dt.total_seconds()
    return df.sort_values(["game_id", "play_id", "nfl_id", "t"]).reset_index(drop=True)


def measure(df: pd.DataFrame) -> pd.DataFrame:
    key = df["game_id"].to_numpy() * 10_000_000 + df["play_id"].to_numpy() * 100_000 + df["nfl_id"].to_numpy() % 100_000
    same = np.r_[False, key[1:] == key[:-1]]
    dt = np.r_[np.nan, np.diff(df["t"].to_numpy())]
    ok = same & (np.abs(dt - 0.1) < 0.02)
    s, d, o = df["s"].to_numpy(), df["dir"].to_numpy(), df["o"].to_numpy()
    vx, vy = s * np.sin(np.deg2rad(d)), s * np.cos(np.deg2rad(d))
    ax, ay = np.r_[np.nan, np.diff(vx)] / 0.1, np.r_[np.nan, np.diff(vy)] / 0.1
    acc = np.hypot(ax, ay)
    acc_dir = np.rad2deg(np.arctan2(ax, ay))               # clockwise from +y, like o
    o_prev = np.r_[np.nan, o[:-1]]
    out = pd.DataFrame({
        "group": df["group"].to_numpy(), "s": s,
        "off": np.abs(wrap(o - d)),                         # 0 = running forward, 180 = backpedal
        "o_turn": np.abs(wrap(o - o_prev)) / 0.1,           # deg/s
        "acc": acc, "acc_off": np.abs(wrap(acc_dir - o_prev)),
    })[ok]
    return out[(out["s"] <= 12) & (out["acc"] <= 15)]


def summarise(m: pd.DataFrame) -> dict:
    res = {}
    sbins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    for g, d in m.groupby("group"):
        r: dict = {}
        sb = pd.cut(d["s"], sbins)
        moving = d[d["s"] > 1.5]
        r["off_median_by_speed"] = {str(k): round(float(v), 1) for k, v in moving.groupby(pd.cut(moving["s"], sbins[2:]), observed=True)["off"].median().items()}
        r["share_off_gt90"] = round(float((moving["off"] > 90).mean()), 3)
        r["share_off_gt135"] = round(float((moving["off"] > 135).mean()), 3)
        for lo, hi in ((45, 90), (90, 135), (135, 181)):
            sel = moving[(moving["off"] >= lo) & (moving["off"] < hi)]["s"]
            r[f"speed_p99_off_{lo}_{hi}"] = round(float(sel.quantile(0.99)), 2) if len(sel) > 100 else None
        r["speed_p99_off_0_45"] = round(float(moving[moving["off"] < 45]["s"].quantile(0.99)), 2)
        r["o_turn_p99_by_speed"] = {str(k): round(float(v)) for k, v in d.groupby(sb, observed=True)["o_turn"].quantile(0.99).items()}
        slow = d[d["s"] < 3]
        ab = pd.cut(slow["acc_off"], [0, 30, 60, 90, 120, 150, 181])
        r["acc_p99_by_angle_slow"] = {str(k): round(float(v), 2) for k, v in slow.groupby(ab, observed=True)["acc"].quantile(0.99).items()}
        r["frames"] = int(len(d))
        res[g] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    args = ap.parse_args()
    res = summarise(measure(load(args.season)))
    print(json.dumps(res, indent=1))
    OUT.write_text(json.dumps({"season": args.season, "groups": res}, indent=1))


if __name__ == "__main__":
    main()

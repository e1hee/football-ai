"""Pass-play numbers for the passing sim, from real NFL tracking (BDB 2027).

Aggregates only (CC BY-NC data is never copied): writes `sim/pass_params.json`.

The BDB 2027 tracking holds only some players per play (1-8, never the QB), so:
  - timing comes from the event frames (any tracked player carries them):
    time to throw = pass_forward - ball_snap, flight = pass_arrived - pass_forward
  - ball speed: QB estimated 6 yd behind the tracked offensive linemen at the
    snap (their mean y), to the targeted receiver at arrival, over the flight
  - catch / interception rates come from player_play (no tracking needed)
  - separation and cushion only where the targeted outside WR and a CB are both
    tracked, measured to the nearest tracked CB (the man across in man
    coverage, the closest real match to a 1-on-1); other defenders may be closer

    uv run python -m analysis.pass_from_bdb
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BDB = Path.home() / "projects/big-data-bowl-2027/data"
OUT = Path(__file__).resolve().parents[1] / "sim/pass_params.json"
EVENTS = ["ball_snap", "pass_forward", "pass_arrived"]
SEP_BINS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 100]
QB_DEPTH = 6.0
KEY = ["game_id", "play_id"]


def q(a, ps=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {f"p{int(p * 100)}": round(float(np.quantile(a, p)), 2) for p in ps} | {"n": int(a.size)}


def frames(season: int) -> pd.DataFrame:
    df = pd.read_parquet(BDB / f"parquet/game_tracking_{season}.parquet",
                         columns=["game_id", "play_id", "nfl_id", "time", "x", "y", "event"],
                         filters=[("event", "in", EVENTS)])
    df["t"] = (pd.to_datetime(df["time"]) - pd.Timestamp("2000-01-01")).dt.total_seconds()  # us or ns alike
    first = df.groupby(KEY + ["event"])["t"].transform("min")   # first frame if an event repeats
    return df[df["t"] == first].drop(columns="time")


def rates(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "catch": round(float((d["pass_result"] == "C").mean()), 3),
            "interception": round(float((d["pass_result"] == "IN").mean()), 3)}


def main() -> None:
    pp = pd.read_csv(BDB / "nfl-big-data-bowl-2027/player_play.csv",
                     usecols=KEY + ["nfl_id", "team_abbr", "possession_team", "lined_up_position", "target",
                                    "pass_result", "pass_length", "yards_after_catch", "route_ran",
                                    "team_coverage_man_zone", "play_nullified_by_penalty", "play_direction"])
    pp = pp[pp["play_nullified_by_penalty"] != "Y"]
    tg = pp[(pp["target"] == True) & pp["pass_result"].isin(["C", "I", "IN"])].drop_duplicates(KEY)  # noqa: E712
    outside_man = tg[(tg["lined_up_position"] == "WR") & (tg["team_coverage_man_zone"] == "MAN_COVERAGE")]

    df = pd.concat([frames(s) for s in (2023, 2024, 2025)], ignore_index=True)
    df = df.merge(pp[KEY + ["nfl_id", "lined_up_position", "team_abbr", "possession_team"]], on=KEY + ["nfl_id"], how="left")
    ev = {e: g.set_index(KEY) for e, g in df.groupby("event")}
    t = {e: g.groupby(level=KEY)["t"].first() for e, g in ev.items()}
    timing = pd.DataFrame({"time_to_throw": t["pass_forward"] - t["ball_snap"],
                           "flight": t["pass_arrived"] - t["pass_forward"]})
    timing = timing[timing["time_to_throw"].between(0.3, 10) | timing["time_to_throw"].isna()]
    timing = timing[timing["flight"].between(0.2, 5) | timing["flight"].isna()]
    tgt_play = tg.set_index(KEY)
    tt = timing.join(tgt_play[["lined_up_position", "team_coverage_man_zone", "nfl_id", "play_direction",
                               "pass_result"]], how="inner")

    # Ball speed: QB ~6 yd behind the tracked linemen at the snap, to the target at arrival.
    snap = ev["ball_snap"].reset_index()
    ol = snap[snap["lined_up_position"].isin(["T", "G", "C"])].groupby(KEY)[["x", "y"]].mean()
    arr = ev["pass_arrived"].reset_index()
    tgt_arr = arr.merge(tg[KEY + ["nfl_id"]], on=KEY + ["nfl_id"]).set_index(KEY)[["x", "y"]]
    b = tt.join(ol, how="inner").join(tgt_arr, rsuffix="_t", how="inner")
    sign = np.where(b["play_direction"] == "right", 1.0, -1.0)     # offense moves +x when "right"
    b["air_dist"] = np.hypot(b["x_t"] - (b["x"] - sign * QB_DEPTH), b["y_t"] - b["y"])
    b["ball_speed"] = b["air_dist"] / b["flight"]

    # Separation / cushion: targeted outside WR to the CB lined up across from him
    # at the snap (tracked CB within ACROSS yd sideways), followed through the play.
    ACROSS = 2.5

    def pair(event: str, cb_ids: pd.Series | None = None) -> pd.DataFrame:
        g = ev[event].reset_index()
        wr = g.merge(outside_man[KEY + ["nfl_id"]], on=KEY + ["nfl_id"])
        cb = g[g["lined_up_position"].isin(["CB"]) & (g["team_abbr"] != g["possession_team"])]
        m = wr.merge(cb, on=KEY, suffixes=("", "_cb"))
        m["d"] = np.hypot(m["x"] - m["x_cb"], m["y"] - m["y_cb"])
        m["dx"] = (m["x"] - m["x_cb"]).abs()
        m["dy"] = (m["y"] - m["y_cb"]).abs()
        if cb_ids is None:
            m = m[m["dy"] < ACROSS]
            m = m.loc[m.groupby(KEY)["dy"].idxmin()]
        else:
            m = m.merge(cb_ids.rename("cb_id").reset_index(), on=KEY)
            m = m[m["nfl_id_cb"] == m["cb_id"]]
        return m.set_index(KEY)[["d", "dx", "dy", "nfl_id_cb"]]

    cushion = pair("ball_snap")
    sep_fwd = pair("pass_forward", cushion["nfl_id_cb"])
    sep_arr = pair("pass_arrived", cushion["nfl_id_cb"]).join(outside_man.set_index(KEY)["pass_result"], how="inner")
    by_sep = []
    for lo, hi in zip(SEP_BINS[:-1], SEP_BINS[1:]):
        m = sep_arr[(sep_arr["d"] >= lo) & (sep_arr["d"] < hi)]
        if len(m) >= 30:
            by_sep.append({"sep_yd": [lo, hi], **rates(m)})

    om_tt = tt[(tt["lined_up_position"] == "WR") & (tt["team_coverage_man_zone"] == "MAN_COVERAGE")]
    out = {
        "source": "NFL Big Data Bowl 2027, REG 2023-2025, aggregates only. Tracking holds only some players per play (never the QB).",
        "units": "s, yd, yd/s",
        "targeted_passes": rates(tg),
        "time_to_throw_s": q(timing["time_to_throw"]),
        "flight_s": q(timing["flight"]),
        "ball_speed_yd_s": q(b["ball_speed"]) | {"note": f"QB assumed {QB_DEPTH} yd behind the tracked linemen"},
        "air_dist_yd": q(b["air_dist"]),
        "ball_speed_by_air_dist": [{"air_dist_yd": [lo, hi], **q(b["ball_speed"][(b["air_dist"] >= lo) & (b["air_dist"] < hi)], (0.25, 0.5, 0.75))}
                                   for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 60))],
        "outside_wr_man": {
            **rates(outside_man),
            "time_to_throw_s": q(om_tt["time_to_throw"]),
            "flight_s": q(om_tt["flight"]),
            "air_yards": q(outside_man["pass_length"]),
            "yac_yd": q(outside_man["yards_after_catch"][outside_man["pass_result"] == "C"]),
            "routes": outside_man["route_ran"].value_counts(normalize=True).round(3).head(12).to_dict(),
            "sep_to_cb_across_at_throw_yd": q(sep_fwd["d"]),
            "sep_to_cb_across_at_arrival_yd": q(sep_arr["d"]),
            "by_sep_to_cb_across_at_arrival": by_sep,
            "cushion_cb_across_at_snap_yd": {"dist": q(cushion["d"]), "downfield": q(cushion["dx"]), "sideways": q(cushion["dy"])},
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

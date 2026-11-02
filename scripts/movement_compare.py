"""Do sim players move like NFL players? Cuts, braking and speed changes.

Real samples (BDB 2027, REG 2024 tracking, 10 Hz):
  - tacklers: DB player-plays with a solo tackle, last 3 s of the play
  - ball carriers: targeted WRs with 5+ yards after catch, from the catch to play end
Sim samples (same 10 Hz): tackler = last 3 s of each play, runner = whole play.

Per segment:
  cut_deg      largest heading change within any 0.5 s window, while moving >= 3 yd/s
  brake        hardest deceleration (yd/s^2, positive number)
  speed_dip    largest drop from a running peak to a later low (yd/s)
  n_changes    times the speed trend flips between speeding up and slowing down (> 0.5 yd/s swings)

    uv run python -m scripts.movement_compare
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim.core import FieldConfig, OneVOneSim, SimConfig
from sim.policies import angle_tackler, evasive_runner
from train.evaluate import ppo_runner, ppo_tackler

BDB = Path.home() / "projects/big-data-bowl-2027/data"
DT = 0.1


def seg_metrics(speed: np.ndarray, heading_deg: np.ndarray) -> dict:
    """speed (T,), heading (T,) in degrees for one player segment."""
    T = len(speed)
    w = 5  # 0.5 s
    cut = 0.0
    for i in range(T - w):
        if speed[i] >= 3.0:
            d = (heading_deg[i + w] - heading_deg[i] + 180) % 360 - 180
            cut = max(cut, abs(d))
    acc = np.diff(speed) / DT
    brake = float(max(0.0, -acc.min())) if acc.size else 0.0
    run_max = np.maximum.accumulate(speed)
    dip = float((run_max - speed).max())
    # trend flips with hysteresis 0.5 yd/s
    n, trend, ref = 0, 0, speed[0]
    for v in speed[1:]:
        if trend >= 0 and v < ref - 0.5:
            n += trend == 1; trend, ref = -1, v
        elif trend <= 0 and v > ref + 0.5:
            n += trend == -1; trend, ref = 1, v
        else:
            ref = max(ref, v) if trend >= 0 else min(ref, v)
    return {"cut_deg": cut, "brake": brake, "speed_dip": dip, "n_changes": n}


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    return {
        "segments": len(df),
        "cut_deg_median": df.cut_deg.median(),
        "share_cut_over_60deg": (df.cut_deg > 60).mean(),
        "brake_median": df.brake.median(),
        "share_brake_over_4": (df.brake > 4).mean(),
        "speed_dip_median": df.speed_dip.median(),
        "n_changes_median": df.n_changes.median(),
    }


def real_segments(season: int = 2024, max_segments: int = 3000) -> tuple[list[dict], list[dict]]:
    pp = pd.read_csv(BDB / "nfl-big-data-bowl-2027/player_play.csv",
                     usecols=["game_id", "play_id", "nfl_id", "tackle", "target", "yards_after_catch"])
    players = pd.read_csv(BDB / "nfl-big-data-bowl-2027/players.csv", usecols=["nfl_id", "nfl_position"])
    pp = pp.merge(players, on="nfl_id")
    tk = pp[(pp.tackle == 1) & pp.nfl_position.isin(["CB", "S", "FS", "SS", "DB"])]
    bc = pp[(pp.target == 1) & (pp.yards_after_catch >= 5) & (pp.nfl_position == "WR")]
    tr = pd.read_parquet(BDB / f"parquet/game_tracking_{season}.parquet",
                         columns=["game_id", "play_id", "nfl_id", "time", "s", "dir", "event"])
    tr = tr.sort_values(["game_id", "play_id", "nfl_id", "time"])
    key = ["game_id", "play_id", "nfl_id"]
    tk_rows, bc_rows = [], []
    for keys, target in ((tk[key], tk_rows), (bc[key], bc_rows)):
        sub = tr.merge(keys.drop_duplicates(), on=key)
        for _, g in sub.groupby(key, sort=False):
            if target is bc_rows:
                ev = g.event.fillna("").to_numpy()
                idx = np.nonzero(ev == "pass_outcome_caught")[0]
                if not idx.size:
                    continue
                g = g.iloc[idx[0]:]
            else:
                g = g.iloc[-30:]   # last 3 s
            if len(g) < 10:
                continue
            target.append(seg_metrics(g.s.to_numpy(), g.dir.to_numpy()))
            if len(target) >= max_segments:
                break
    return tk_rows, bc_rows


def sim_segments(runner, tackler, n: int = 2000, seed: int = 31) -> tuple[list[dict], list[dict]]:
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=8.0))
    sim = OneVOneSim(n, cfg, seed=seed)
    obs = sim.reset()
    rv, tv = [sim.r_vel.copy()], [sim.t_vel.copy()]
    live = np.ones(n, bool)
    end = np.full(n, -1)
    t = 0
    while live.any():
        obs, _, done, _ = sim.step(runner(sim, obs), tackler(sim))
        t += 1
        rv.append(sim.r_vel.copy()); tv.append(sim.t_vel.copy())  # post-reset values ignored via `end`
        end[done & live] = t
        live &= ~done
    rv, tv = np.stack(rv), np.stack(tv)
    def hd(v):  # heading in NFL-like degrees (only differences matter)
        return np.degrees(np.arctan2(v[..., 0], v[..., 1]))
    t_rows, r_rows = [], []
    for i in range(n):
        e = end[i]   # velocities at index e are already from the next play: stop at e-1
        if e < 10:
            continue
        r = rv[:e, i]; tt = tv[max(0, e - 30):e, i]
        r_rows.append(seg_metrics(np.linalg.norm(r, axis=1), hd(r)))
        t_rows.append(seg_metrics(np.linalg.norm(tt, axis=1), hd(tt)))
    return t_rows, r_rows


def main() -> None:
    real_tk, real_bc = real_segments()
    E3 = "experiments/EXP-003-selfplay-v1/checkpoints"
    champ = ppo_runner("experiments/EXP-002-ppo-runner-v1/checkpoints/gen_025.zip")
    S = lambda f: (lambda sim, obs: f(sim))
    T12, R12 = ppo_tackler(f"{E3}/tackler_g12.zip"), ppo_runner(f"{E3}/runner_g12.zip")
    sims = {
        "AI 셀프플레이 (R12 vs T12)": sim_segments(R12, lambda s: T12(s)),
        "AI 챔피언 러너 vs 각도 태클러": sim_segments(champ, angle_tackler),
        "회피(스크립트) vs 각도 태클러": sim_segments(S(evasive_runner), angle_tackler),
    }
    cols = ["segments", "cut_deg_median", "share_cut_over_60deg", "brake_median", "share_brake_over_4",
            "speed_dip_median", "n_changes_median"]
    head = "구간 수 | 0.5초 최대 방향전환(중앙값) | 60도+ 비율 | 최대 감속(중앙값) | 감속 4+ 비율 | 속도 낙폭 | 속도 변화 횟수"
    for title, real, idx in (("수비(태클러): 마지막 3초", real_tk, 0), ("공 가진 러너", real_bc, 1)):
        print(f"\n== {title} ==\n{head}")
        rows = [("NFL 실제", summarize(real))] + [(k, summarize(v[idx])) for k, v in sims.items()]
        for name, s in rows:
            print(f"{name:<28} {s['segments']:5d} | {s['cut_deg_median']:6.0f}° | {s['share_cut_over_60deg']:6.0%} | "
                  f"{s['brake_median']:5.1f} | {s['share_brake_over_4']:6.0%} | {s['speed_dip_median']:5.1f} | {s['n_changes_median']:4.0f}")


if __name__ == "__main__":
    main()

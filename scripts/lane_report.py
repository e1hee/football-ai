"""Lane drill (Oklahoma) report: does the runner wait for the block and cut?

Per offense-vs-defense pair, over the same starts:
- result: win (10 yd past the line), tackled, out of the lane, yards past the line
- patience: time to the line, slowest speed while approaching it (after 0.5 s),
  share of plays with a deliberate brake before the line (slowing 2.5+ yd/s^2
  from 2+ yd/s with nobody touching him)
- cut: crossing the line moving sideways (|vy| >= 1 yd/s), cutback (sideways
  direction flips within 4 yd of the line), and hitting the open side (crossing
  on the side away from where the DL leans off his blocker)
- who tackles: DL out of the block, DL after shedding it, LB
- with an LB: speed when crossing the line, share that slowed 1+ yd/s from
  their top speed before it, and crossing on the side away from the LB

    uv run python -m scripts.lane_report experiments/EXP-013-lane-drill --gen 12
    uv run python -m scripts.lane_report experiments/EXP-013-lane-drill --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_OUT_OF_BOUNDS, OUTCOME_TACKLED, OUTCOME_TOUCHDOWN, FieldConfig, TackleConfig
from sim.team import CARRIER, TeamSim, lane_drill_config
from sim.team_policies import gap_defense, lane_offense, pursuit_defense
from train.evaluate import ppo_runner, ppo_tackler

DT = 0.1


def lane_stats(offense, defense, cfg, n: int = 2048, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    outcome = np.zeros(n, np.int8)
    yards = np.zeros(n, np.float32)
    t_cross = np.full(n, np.nan)
    min_speed = np.full(n, np.inf)
    braked = np.zeros(n, bool)
    cross_vy = np.zeros(n)
    open_side = np.zeros(n, bool)
    flips = np.zeros(n, bool)
    last_vy_sign = np.zeros(n)
    tackler = np.full(n, -1)                  # 0 DL engaged, 1 DL free, 2 LB
    peak = np.zeros(n)
    line_speed = np.full(n, np.nan)
    away_lb = np.zeros(n, bool)
    lb = sim.dfn[1] if sim.n_def > 1 else None
    rad = sim.bodies.radius
    dl, ol = sim.dfn[0], sim.blk[0]
    while live.any():
        o_act, d_act = offense(sim, obs), defense(sim, obs)
        v0 = sim.vel[:, CARRIER].copy()
        dl_engaged = sim.partner[:, dl] >= 0
        lean = sim.pos[:, dl, 1] - sim.pos[:, ol, 1]
        before = sim.pos[:, CARRIER, 0] < sim.los
        obs, _, done, info = sim.step(o_act, d_act)
        pos = info["pos"]
        v1 = sim.vel[:, CARRIER].copy()            # finished plays are already reset; masked by ~done below
        sp0, sp1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
        gap = np.linalg.norm(pos[:, None, CARRIER] - pos[:, 1:], axis=-1) - (rad[:, CARRIER, None] + rad[:, 1:])
        touching = (gap < 0.05).any(1) | info["wrapped"] | info["tackle_attempt"]
        approach = live & before & ~done
        t = sim.t * DT
        min_speed = np.where(approach & (t > 0.5), np.minimum(min_speed, sp1), min_speed)
        peak = np.where(approach, np.maximum(peak, sp1), peak)
        braked |= approach & ~touching & (sp0 > 2.0) & ((sp0 - sp1) / DT >= 2.5)
        near = approach & (sim.los - pos[:, CARRIER, 0] < 4.0)
        s = np.sign(v1[:, 1]) * (np.abs(v1[:, 1]) >= 0.8)
        flips |= near & (s != 0) & (last_vy_sign != 0) & (s != last_vy_sign)
        last_vy_sign = np.where(near & (s != 0), s, last_vy_sign)
        crossed = live & before & (pos[:, CARRIER, 0] >= sim.los) & np.isnan(t_cross)
        t_cross[crossed] = (info["steps"][crossed]) * DT
        cross_vy[crossed] = v0[crossed, 1]
        side = np.sign(pos[:, CARRIER, 1] - pos[:, ol, 1])
        open_side[crossed] = (side * np.sign(lean) < 0)[crossed]
        line_speed[crossed] = sp0[crossed]
        if lb is not None:
            away_lb[crossed] = (side * np.sign(pos[:, lb, 1] - pos[:, ol, 1]) < 0)[crossed]
        made = live & info["tackle_made"]
        if made.any():
            idx = np.nonzero(made)[0]
            who = np.argmin(np.linalg.norm(pos[idx][:, None, CARRIER] - pos[idx][:, sim.dfn], axis=-1), axis=1)
            tackler[idx] = np.where(who == 0, np.where(dl_engaged[idx], 0, 1), 2)
        newly = done & live
        outcome[newly] = info["outcome"][newly]
        yards[newly] = info["yards"][newly]
        live &= ~done
    got = ~np.isnan(t_cross)
    tk = outcome == OUTCOME_TACKLED
    return {
        "win": float((outcome == OUTCOME_TOUCHDOWN).mean()),
        "tackled": float(tk.mean()),
        "out": float((outcome == OUTCOME_OUT_OF_BOUNDS).mean()),
        "yards_mean": float(yards.mean()),
        "yards_median": float(np.median(yards)),
        "reached_line": float(got.mean()),
        "time_to_line": float(np.nanmedian(t_cross)) if got.any() else float("nan"),
        "min_speed_approach": float(np.median(min_speed[np.isfinite(min_speed)])) if np.isfinite(min_speed).any()
        else float("nan"),
        "brake_share": float(braked.mean()),
        "sideways_at_line": float((np.abs(cross_vy[got]) >= 1.0).mean()) if got.any() else float("nan"),
        "cutback_share": float(flips.mean()),
        "open_side_share": float(open_side[got].mean()) if got.any() else float("nan"),
        "tackle_by_dl_engaged": float((tackler[tk] == 0).mean()) if tk.any() else float("nan"),
        "tackle_by_dl_free": float((tackler[tk] == 1).mean()) if tk.any() else float("nan"),
        "tackle_by_lb": float((tackler[tk] == 2).mean()) if tk.any() else float("nan"),
        "speed_at_line": float(np.nanmedian(line_speed)) if got.any() else float("nan"),
        "slowed_before_line": float((peak[got] - line_speed[got] >= 1.0).mean()) if got.any() else float("nan"),
        "away_from_lb": float(away_lb[got].mean()) if (got.any() and lb is not None) else float("nan"),
    }


COLS = [("win", "승", "%"), ("tackled", "태클", "%"), ("out", "이탈", "%"), ("yards_mean", "야드", "f"),
        ("time_to_line", "라인까지s", "f"), ("min_speed_approach", "최저속도", "f"), ("brake_share", "감속", "%"),
        ("sideways_at_line", "옆으로통과", "%"), ("cutback_share", "컷백", "%"), ("open_side_share", "빈쪽", "%"),
        ("tackle_by_dl_engaged", "블록중태클", "%"), ("tackle_by_lb", "LB태클", "%"),
        ("speed_at_line", "라인속도", "f"), ("slowed_before_line", "늦춤", "%"), ("away_from_lb", "LB반대", "%")]


def fmt_row(name: str, s: dict) -> str:
    cells = []
    for k, _, f in COLS:
        v = s[k]
        cells.append("  -  " if v != v else (f"{v:5.0%}" if f == "%" else f"{v:5.2f}"))
    return f"{name:34s} " + " ".join(f"{c:>8s}" for c in cells)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path, help="folder with s0/s1/... runs (or one run)")
    p.add_argument("--gen", type=int, default=None, help="generation to compare (default: last)")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--lb", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = lane_drill_config(width=args.width, linebacker=args.lb, field=FieldConfig(reaction_delay=2),
                            tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()) or [args.exp]

    def ckpt(run, kind):
        files = sorted((run / "checkpoints").glob(f"{kind}_g*.zip"))
        if args.gen is not None:
            files = [f for f in files if f.stem.endswith(f"g{args.gen:02d}")] or files
        return files[-1]

    off = {"스크립트: 스냅 때 결정": lambda s, o=None: lane_offense(s, read_until=99.0),
           "스크립트: 3.5야드까지 읽기": lambda s, o=None: lane_offense(s)}
    dfn = {"스크립트 갭 DL": lambda s, o=None: gap_defense(s),
           "스크립트 추격 DL": lambda s, o=None: pursuit_defense(s)}
    for r in runs:
        off[f"AI 공격 {r.name} {ckpt(r, 'runner').stem[-3:]}"] = ppo_runner(ckpt(r, "runner"))
        dfn[f"AI 수비 {r.name} {ckpt(r, 'tackler').stem[-3:]}"] = ppo_tackler(ckpt(r, "tackler"))

    results = {}
    print(f"{'공격 vs 수비':34s} " + " ".join(f"{h:>8s}" for _, h, _ in COLS))
    for dname, d in dfn.items():
        for oname, o in off.items():
            s = lane_stats(o, d, cfg, n=args.n)
            results[f"{oname} vs {dname}"] = s
            print(fmt_row(f"{oname[:16]} / {dname[:14]}", s))
        print()
    ai_off = [k for k in off if k.startswith("AI")]
    ai_def = [k for k in dfn if k.startswith("AI")]
    if ai_off and ai_def:
        league = {k: float(np.mean([results[f"{o} vs {d}"][k] for o in ai_off for d in ai_def]))
                  for k, _, _ in COLS}
        print(fmt_row("AI 공격 전체 vs AI 수비 전체 (평균)", league))
        results["league"] = league
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

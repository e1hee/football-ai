"""What the pass game looks like at one generation (EXP-039+): routes, throws, coverage.

Matchups (one play per slot, deterministic actions, fixed starts):
  AI vs AI (same seed), AI offense vs scripted man coverage, scripted routes vs AI
  CB, scripted vs scripted. Per matchup:
  - outcomes: completion, interception, sack, yards per attempt
  - throw: time after the snap; receiver depth past the line at the throw
  - route: a break = the receiver's heading turns 35+ deg within 1 s, from 0.5 s until the
    ball arrives (NFL p99 is ~35 deg per 0.5 s at 5-7 yd/s, so real cuts slow down first);
    his direction when the ball arrives (up / inside / outside / back, back = under 1 yd/s
    or moving toward the line)
  - where he is when the ball arrives: downfield and inside (+) / outside (-) of his start,
    and the share 3+ yd inside or outside (a route that is not a straight line)
  - separation receiver-CB when the ball arrives; CB cushion (downfield) at the throw
  - the same split by the CB's alignment at the snap: press (< 4 yd off) vs off
NFL references (outside WR vs man, `sim/pass_params.json`): throw 2.5 s median,
separation at arrival 1.5 yd, catch 52%, pick 2.8%, routes GO 23% SLANT 15% HITCH 14% OUT 12%.

    uv run python -m scripts.pass_report experiments/EXP-039-pass --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_INTERCEPTION, OUTCOME_OUT_OF_BOUNDS, OUTCOME_SACK, OUTCOME_TACKLED, OUTCOME_TOUCHDOWN
from sim.pass_policies import man_coverage, route_offense
from sim.passing import CB, WR, PassSim
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import config_from_run

DIRS = ["up", "inside", "outside", "back"]


def heading(v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def play(cfg, off, dfn, n: int = 2048, seed: int = 7) -> dict:
    sim = PassSim(n, cfg, seed=seed)
    obs = sim.reset()
    dt = cfg.field.dt
    fin = np.zeros(n, bool)
    out = np.zeros(n, np.int8)
    yards = np.zeros(n, np.float32)
    cushion0 = sim.pos[:, CB, 0] - sim.pos[:, WR, 0]
    start = sim.pos[:, WR].copy()
    disp = np.full((n, 2), np.nan)               # receiver (downfield, inside) from his start when the ball arrives
    inside = sim.wr_inside.copy()
    hist = []                                    # receiver velocity per step
    t_throw = np.full(n, np.nan)
    depth = np.full(n, np.nan)
    v_throw = np.zeros((n, 2))
    v_arr = np.zeros((n, 2))
    cush_throw = np.full(n, np.nan)
    sep = np.full(n, np.nan)
    broke = np.zeros(n, bool)
    thrown = np.zeros(n, bool)
    for _ in range(cfg.field.max_steps + 1):
        a_o, a_d = off(sim, obs), dfn(sim, obs)
        live = ~fin
        hist.append(sim.vel[:, WR].copy())
        obs, _, done, info = sim.step(a_o, a_d)
        tn = info["thrown_now"] & live
        if tn.any():
            t_throw[tn] = sim.t[tn] * dt
            depth[tn] = sim.pos[tn, WR, 0] - sim.los[tn]
            v_throw[tn] = sim.vel[tn, WR]
            cush_throw[tn] = sim.pos[tn, CB, 0] - sim.pos[tn, WR, 0]
            thrown |= tn
        ar = info["arrived"] & live
        if ar.any():
            sep[ar] = np.linalg.norm(info["pos"][ar, WR] - info["pos"][ar, CB], axis=-1)
            v_arr[ar] = sim.vel[ar, WR]
            d = info["pos"][ar, WR] - start[ar]
            disp[ar] = np.stack([d[:, 0], d[:, 1] * inside[ar]], 1)
            h = heading(np.stack(hist, 1)[ar])      # (k, steps) receiver heading up to the arrival
            if h.shape[1] > 15:
                turn = np.abs((h[:, 15:] - h[:, 5:-10] + 180) % 360 - 180)
                speed_ok = np.linalg.norm(np.stack(hist, 1)[ar][:, 15:], axis=-1) > 2.0
                broke[ar] = ((turn >= 35) & speed_ok).any(1)
        new = done & live
        out[new], yards[new] = info["outcome"][new], info["yards"][new]
        fin |= done
        if fin.all():
            break
    speed = np.linalg.norm(v_arr, axis=-1)
    ang = np.degrees(np.arctan2(v_arr[:, 1] * inside, v_arr[:, 0]))          # + = toward the middle
    cls = np.where((speed < 1.0) | (np.abs(ang) > 120), 3, np.where(np.abs(ang) <= 30, 0, np.where(ang > 0, 1, 2)))
    caught = np.isin(out, [OUTCOME_TOUCHDOWN, OUTCOME_TACKLED, OUTCOME_OUT_OF_BOUNDS])

    def summary(m: np.ndarray) -> dict:
        t = m & thrown
        q = lambda a: [round(float(x), 2) for x in np.nanpercentile(a, [25, 50, 75])] if np.isfinite(a).any() else None
        return {"plays": int(m.sum()), "completion": round(float(caught[m].mean()), 3),
                "interception": round(float((out[m] == OUTCOME_INTERCEPTION).mean()), 3),
                "sack": round(float((out[m] == OUTCOME_SACK).mean()), 3), "yards": round(float(yards[m].mean()), 2),
                "throw_s_q": q(t_throw[t]), "depth_at_throw_q": q(depth[t]), "sep_at_arrival_q": q(sep[m]),
                "cb_cushion_at_throw_q": q(cush_throw[t]),
                "downfield_at_arrival_q": q(disp[t, 0]), "inside_at_arrival_q": q(disp[t, 1]),
                "lateral_3yd": {"inside": round(float((disp[t, 1] >= 3).mean()), 3),
                                "outside": round(float((disp[t, 1] <= -3).mean()), 3)} if t.any() else None,
                "break_before_arrival": round(float(broke[t].mean()), 3) if t.any() else None,
                "dir_at_arrival": {d: round(float((cls[t] == i).mean()), 3) for i, d in enumerate(DIRS)} if t.any() else None}

    press = cushion0 < 4.0
    return {"all": summary(np.ones(n, bool)), "press": summary(press), "off": summary(~press)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = {}
    runs = sorted(d for d in args.exp.iterdir() if (d / "config.yaml").is_file())
    script_o = lambda sim, obs: route_offense(sim)
    script_d = lambda sim, obs: man_coverage(sim)
    cfg0 = config_from_run(runs[0])
    res["script_vs_script"] = play(cfg0, script_o, script_d, args.n)
    for run in runs:
        cfg = config_from_run(run)
        ck = run / "checkpoints"
        o = ppo_runner(ck / f"runner_g{args.gen:02d}.zip")
        d = ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip")
        res[run.name] = {"ai_vs_ai": play(cfg, o, d, args.n), "ai_vs_script_cb": play(cfg, o, script_d, args.n),
                         "script_routes_vs_ai": play(cfg, script_o, d, args.n)}
    for k, v in res.items():
        rows = {"": v} if "all" in v else v
        for m, r in rows.items():
            a = r["all"]
            print(f"{k} {m}: comp {a['completion']:.0%} int {a['interception']:.1%} sack {a['sack']:.0%} yd {a['yards']:.1f} | "
                  f"throw {a['throw_s_q']} s depth {a['depth_at_throw_q']} | sep {a['sep_at_arrival_q']} | "
                  f"lat3 {a['lateral_3yd']} break {a['break_before_arrival']} dir {a['dir_at_arrival']} | press comp {r['press']['completion']:.0%} "
                  f"dir {r['press']['dir_at_arrival']} | off comp {r['off']['completion']:.0%} dir {r['off']['dir_at_arrival']}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

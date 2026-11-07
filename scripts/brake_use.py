"""How fast do players run and how hard do they brake, in a run's own games (EXP-038).

Own offense vs own defense at one generation. Per position group:
  - share of live steps at >= 6 / >= 7 yd/s (where the brake fix matters: big
    players' stopping power changed above ~6 yd/s, `analysis/brake_calibrate.py`)
  - share of fast steps (>= 5 yd/s, not engaged in a block) with a hard brake
    push (push component against the motion <= -0.5)
  - mean braking actually achieved on those hard-brake steps (yd/s^2, from the
    commanded push and the physics caps the run used)

    uv run python -m scripts.brake_use experiments/EXP-038-brake --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.physics import _caps
from sim.team import TeamSim
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import config_from_run

GROUPS = {"RB": [0], "OL": [1, 2, 3, 4, 5], "DL": [6, 7, 8, 9], "LB": [10, 11]}


def measure(run: Path, gen: int, n: int, seed: int = 5) -> dict:
    ck = run / "checkpoints"
    off, dfn = ppo_runner(ck / f"runner_g{gen:02d}.zip"), ppo_tackler(ck / f"tackler_g{gen:02d}.zip")
    sim = TeamSim(n, config_from_run(run), seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    acc = {g: dict(steps=0, s6=0, s7=0, fast=0, hard=0, brk=[]) for g in GROUPS}
    for _ in range(80):
        a_off, a_def = np.asarray(off(sim, obs)), np.asarray(dfn(sim, obs))
        push = np.concatenate([a_off.reshape(n, 6, -1), a_def.reshape(n, 6, -1)], 1)[..., :2]
        norm = np.linalg.norm(push, axis=-1, keepdims=True)
        push = np.where(norm > 1, push / np.maximum(norm, 1e-6), push)
        vel = sim.vel.copy()
        sp = np.linalg.norm(vel, axis=-1)
        a_t = np.sum(push * vel / np.maximum(sp, 1e-6)[..., None], -1)
        _, brk_cap, _ = _caps(sp, sim.bodies)
        engaged = sim.partner >= 0 if hasattr(sim, "partner") else np.zeros_like(sp, bool)
        for g, idx in GROUPS.items():
            m = live[:, None] & np.ones((n, len(idx)), bool)
            s, at, cap, eng = sp[:, idx], a_t[:, idx], brk_cap[:, idx], engaged[:, idx]
            acc[g]["steps"] += int(m.sum())
            acc[g]["s6"] += int((m & (s >= 6)).sum())
            acc[g]["s7"] += int((m & (s >= 7)).sum())
            fast = m & (s >= 5) & ~eng
            hard = fast & (at <= -0.5)
            acc[g]["fast"] += int(fast.sum())
            acc[g]["hard"] += int(hard.sum())
            acc[g]["brk"].extend((-at * cap)[hard].tolist())
        obs, _, done, _ = sim.step(a_off, a_def)
        live &= ~done
        if not live.any():
            break
    out = {}
    for g, a in acc.items():
        out[g] = {"share_ge6": round(a["s6"] / max(a["steps"], 1), 4), "share_ge7": round(a["s7"] / max(a["steps"], 1), 4),
                  "hard_brake_share_of_fast": round(a["hard"] / max(a["fast"], 1), 4),
                  "hard_brake_mean_yds2": round(float(np.mean(a["brk"])), 2) if a["brk"] else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = {}
    for run in sorted(d for d in args.exp.iterdir() if (d / "config.yaml").is_file()):
        res[run.name] = r = measure(run, args.gen, args.n)
        print(f"{run.name} g{args.gen}:")
        for g, v in r.items():
            print(f"  {g}: >=6 yd/s {v['share_ge6']:.1%}  >=7 {v['share_ge7']:.1%}  hard brake (fast, free) "
                  f"{v['hard_brake_share_of_fast']:.1%}  achieved {v['hard_brake_mean_yds2']} yd/s^2")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

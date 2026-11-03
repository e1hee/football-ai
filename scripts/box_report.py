"""6v6 inside-run report (`--sim box`): results next to the NFL, and does the
runner wait.

Per offense-vs-defense pair, same starts:
- result: win (10 yd past the line; no secondary), yards (capped at 10), share
  stopped at or behind the line, share of 10+
- runner: time to the line, plays where he slowed 1+ yd/s on his own before
  the line (nobody touching him) vs by contact, crossing inside the tackles
- blocking: DL double-teamed at some point, an LB blocked before the runner
  reached the line
- who tackles: DL or LB

NFL: league yards per carry 4.45 in 2024 (PFF). Our yards stop at 10, so the
mean is a floor of what the same runs would gain on a full field.

    uv run python -m scripts.box_report experiments/EXP-017-box --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TACKLED, OUTCOME_TOUCHDOWN, FieldConfig, TackleConfig
from sim.team import CARRIER, TeamSim, box_config
from sim.team_policies import box_offense, gap_defense, pursuit_defense
from train.evaluate import ppo_runner, ppo_tackler

DT = 0.1


def box_stats(offense, defense, cfg, n: int = 2048, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    roles = np.array(sim.roles)
    dls, lbs = sim.dfn[roles[sim.dfn] == "DL"], sim.dfn[roles[sim.dfn] == "LB"]
    live = np.ones(n, bool)
    outcome = np.zeros(n, np.int8)
    yards = np.zeros(n)
    t_cross = np.full(n, np.nan)
    inside = np.zeros(n, bool)
    lost_free = np.zeros(n)
    lost_touch = np.zeros(n)
    doubled = np.zeros(n, bool)
    lb_blocked = np.zeros(n, bool)
    by_lb = np.zeros(n, bool)
    rad = sim.bodies.radius
    while live.any():
        o, d = offense(sim, obs), defense(sim, obs)
        v0 = np.linalg.norm(sim.vel[:, CARRIER], axis=1)
        before = sim.pos[:, CARRIER, 0] < sim.los
        ol_y = np.sort(sim.pos[:, sim.blk, 1], axis=1)
        obs, _, done, info = sim.step(o, d)
        pos, part = info["pos"], info["partner"]
        v1 = np.linalg.norm(sim.vel[:, CARRIER], axis=1)
        gap = np.linalg.norm(pos[:, None, CARRIER] - pos[:, 1:], axis=-1) - (rad[:, CARRIER, None] + rad[:, 1:])
        touching = (gap < 0.05).any(1) | info["wrapped"] | info["tackle_attempt"]
        ok = live & before & ~done
        drop = np.maximum(0.0, v0 - v1)
        lost_free += np.where(ok & ~touching, drop, 0.0)
        lost_touch += np.where(ok & touching, drop, 0.0)
        on = (part[:, sim.blk][:, :, None] == dls[None, None, :]).sum(1)
        doubled |= live & (on >= 2).any(1)
        lb_blocked |= live & before & (part[:, lbs] >= 0).any(1)
        crossed = live & before & (pos[:, CARRIER, 0] >= sim.los) & np.isnan(t_cross)
        t_cross[crossed] = info["steps"][crossed] * DT
        inside[crossed] = ((pos[:, CARRIER, 1] > ol_y[:, 0]) & (pos[:, CARRIER, 1] < ol_y[:, -1]))[crossed]
        made = live & info["tackle_made"]
        if made.any():
            idx = np.nonzero(made)[0]
            k = np.argmin(np.linalg.norm(pos[idx][:, None, CARRIER] - pos[idx][:, sim.dfn], axis=-1), axis=1)
            by_lb[idx] = roles[sim.dfn[k]] == "LB"
        newly = done & live
        outcome[newly] = info["outcome"][newly]
        yards[newly] = info["yards"][newly]
        live &= ~done
    got = ~np.isnan(t_cross)
    tk = outcome == OUTCOME_TACKLED
    return {
        "win": float((outcome == OUTCOME_TOUCHDOWN).mean()),
        "yards_mean": float(yards.mean()),
        "yards_median": float(np.median(yards)),
        "stuffed": float((yards <= 0).mean()),
        "ten_plus": float((yards >= 10).mean()),
        "time_to_line": float(np.nanmedian(t_cross)) if got.any() else float("nan"),
        "own_brake_1plus": float((lost_free >= 1.0).mean()),
        "contact_slow_1plus": float((lost_touch >= 1.0).mean()),
        "inside_tackles": float(inside[got].mean()) if got.any() else float("nan"),
        "double_team": float(doubled.mean()),
        "lb_blocked_before_line": float(lb_blocked.mean()),
        "tackle_by_lb": float(by_lb[tk].mean()) if tk.any() else float("nan"),
    }


COLS = [("win", "승", "%"), ("yards_mean", "야드", "f"), ("yards_median", "중앙", "f"), ("stuffed", "0이하", "%"),
        ("ten_plus", "10+", "%"), ("time_to_line", "라인s", "f"), ("own_brake_1plus", "스스로감속", "%"),
        ("contact_slow_1plus", "접촉감속", "%"), ("inside_tackles", "안쪽", "%"), ("double_team", "더블팀", "%"),
        ("lb_blocked_before_line", "LB블록", "%"), ("tackle_by_lb", "LB태클", "%")]


def fmt_row(name: str, s: dict) -> str:
    cells = ["  -  " if s[k] != s[k] else (f"{s[k]:5.0%}" if f == "%" else f"{s[k]:5.2f}") for k, _, f in COLS]
    return f"{name:30s} " + " ".join(f"{c:>8s}" for c in cells)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path, help="folder with s0/s1/... runs")
    p.add_argument("--gen", type=int, default=None)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--stumble", type=float, default=0.2)
    p.add_argument("--snap", action="store_true", help="--sim box-snap")
    p.add_argument("--mix", action="store_true", help="--sim box-mix")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = box_config(snap=args.snap, mix=args.mix, stumble=args.stumble, carrier_square_ahead=True, timeout_ends=True,
                     field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir())
    for r in runs:       # this report runs with hips on: a run trained without them is a different game
        if "facing: false" in (r / "config.yaml").read_text():
            raise SystemExit(f"{r} was trained without --facing")

    def ckpt(run, kind):
        files = sorted((run / "checkpoints").glob(f"{kind}_g*.zip"))
        if args.gen is not None:
            files = [f for f in files if f.stem.endswith(f"g{args.gen:02d}")] or files
        return files[-1]

    off = {"스크립트: 구멍 읽기": lambda s, o=None: box_offense(s),
           "스크립트: 스냅 때 결정": lambda s, o=None: box_offense(s, read_until=99.0)}
    dfn = {"스크립트 갭 수비": lambda s, o=None: gap_defense(s),
           "스크립트 추격 수비": lambda s, o=None: pursuit_defense(s)}
    for r in runs:
        off[f"AI 공격 {r.name} {ckpt(r, 'runner').stem[-3:]}"] = ppo_runner(ckpt(r, "runner"))
        dfn[f"AI 수비 {r.name} {ckpt(r, 'tackler').stem[-3:]}"] = ppo_tackler(ckpt(r, "tackler"))
    results = {}
    print(f"{'공격 / 수비':30s} " + " ".join(f"{h:>8s}" for _, h, _ in COLS))
    for dname, d in dfn.items():
        for oname, o in off.items():
            s = box_stats(o, d, cfg, n=args.n)
            results[f"{oname} vs {dname}"] = s
            print(fmt_row(f"{oname[:14]} / {dname[:12]}", s))
        print()
    ai_off = [k for k in off if k.startswith("AI")]
    ai_def = [k for k in dfn if k.startswith("AI")]
    if ai_off and ai_def:
        league = {k: float(np.nanmean([results[f"{o} vs {d}"][k] for o in ai_off for d in ai_def])) for k, _, _ in COLS}
        print(fmt_row("AI 공격 전체 / AI 수비 전체", league))
        results["league"] = league
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

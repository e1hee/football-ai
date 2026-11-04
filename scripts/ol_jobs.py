"""What do the offensive linemen do? Per offense-vs-defense pair, until the
runner crosses the line (or the play ends):

- per OL each step: blocking someone / free near the line (within 1 yd of it,
  either side) / free and 1+ yd downfield ("ran upfield") / free and 1+ yd in
  the backfield. Reported as shares of OL-steps.
- OL who never blocked anyone before the runner reached the line.
- downfield free OL: did he end up within 1.5 yd of an unblocked defender
  (i.e. at least in the way), or nowhere near one.
- unblocked defenders (not engaged, not stunned) when the runner reaches the line.

    uv run python -m scripts.ol_jobs experiments/EXP-020-tackle --gen 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.team import CARRIER, TeamSim
from sim.team_policies import box_offense, gap_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config


def ol_jobs(offense, defense, cfg, n: int = 1024, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    roles = np.array(sim.roles)
    blk, dfn = sim.blk, sim.dfn
    live = np.ones(n, bool)        # play on and runner not yet across the line
    counts = {"blocking": 0, "near_line": 0, "upfield": 0, "backfield": 0}
    ever = np.zeros((n, len(blk)), bool)
    up_near, up_far = 0, 0
    free_def_at_line, crossed_n = [], 0
    lb_free_at_line = []
    snap = int(round(cfg.snap_time / cfg.field.dt))
    t = 0
    while live.any():
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        t += 1
        pos, part = info["pos"], info["partner"]
        los = sim.los[:, None]
        if t > snap:
            eng = part[:, blk] >= 0
            dx = pos[:, blk, 0] - los
            m = live[:, None]
            counts["blocking"] += int((eng & m).sum())
            free = ~eng & m
            counts["near_line"] += int((free & (np.abs(dx) < 1.0)).sum())
            up = free & (dx >= 1.0)
            counts["upfield"] += int(up.sum())
            counts["backfield"] += int((free & (dx <= -1.0)).sum())
            ever |= eng & m
            d_free = (part[:, dfn] < 0) & (sim.stun[:, dfn] <= 0)
            dist = np.linalg.norm(pos[:, blk, None] - pos[:, None, dfn], axis=-1)       # (N, OL, D)
            close = (np.where(d_free[:, None, :], dist, np.inf).min(-1) < 1.5)
            up_near += int((up & close).sum())
            up_far += int((up & ~close).sum())
        crossed = live & (pos[:, CARRIER, 0] >= sim.los)
        if crossed.any():
            d_free = (part[:, dfn] < 0) & (sim.stun[:, dfn] <= 0)
            free_def_at_line += list(d_free[crossed].sum(1))
            lb_free_at_line += list(d_free[crossed][:, roles[dfn] == "LB"].sum(1))
            crossed_n += int(crossed.sum())
        live &= ~done & ~crossed
        if t > 100:
            break
    total = sum(counts.values())
    return {**{k: v / total for k, v in counts.items()},
            "never_blocked": float((~ever).mean()),
            "upfield_near_def": up_near / max(up_near + up_far, 1),
            "free_def_at_line": float(np.mean(free_def_at_line)) if free_def_at_line else float("nan"),
            "free_lb_at_line": float(np.mean(lb_free_at_line)) if lb_free_at_line else float("nan"),
            "reached_line": crossed_n / n}


COLS = [("blocking", "블록중"), ("near_line", "라인근처"), ("upfield", "앞으로"), ("backfield", "뒤쪽"),
        ("never_blocked", "한번도안막음"), ("upfield_near_def", "앞으로중수비옆"),
        ("free_def_at_line", "라인때빈수비"), ("free_lb_at_line", "그중LB"), ("reached_line", "라인도달")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir())
    pairs = {"스크립트 공격 / 스크립트 갭 수비": (lambda s, o=None: box_offense(s), lambda s, o=None: gap_defense(s))}
    for r in runs:
        ck = r / "checkpoints"
        pairs[f"AI {r.name} / AI {r.name}"] = (ppo_runner(ck / f"runner_g{args.gen:02d}.zip"),
                                               ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip"))
    print(f"{'':28s}" + "".join(f"{h:>10s}" for _, h in COLS))
    out = {}
    for name, (o, d) in pairs.items():
        s = ol_jobs(o, d, cfg, n=args.n)
        out[name] = s
        cells = [f"{s[k]:.2f}" if k.startswith("free_") else f"{s[k]:.0%}" for k, _ in COLS]
        print(f"{name:28s}" + "".join(f"{c:>10s}" for c in cells), flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

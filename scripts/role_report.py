"""Does each position do its job? The jobs are the user's (2026-11-04), measured until
the runner crosses the line of scrimmage (or the play ends):

OL: hold, keep the DL off the runner, open a lane so he can run the shortest way.
- ol_free_run: share of OL-steps running upfield with no opponent to block: free, 1+ yd
  past the line, moving forward, no unblocked defender within 2.5 yd.
- reblock_1s: after a block ends with 1+ s left, does he block someone again.
- backfield_contact: plays where a defender gets within 1.5 yd of the runner while the
  runner is still 1+ yd behind the line.
- lane_ahead: when the runner is 2 yd behind the line, sideways distance to the nearest
  defender in front of him (median yd). best_gap: widest gap between neighboring
  defenders near the line then.

RB: shortest way, chooses side or gap.
- path_ratio: path length to the line / straight-line distance (median).
- time_to_line (s, plays that got there), inside (crossed between the tackles),
  right (crossed on the right half), box (share of plays that crossed that crossed within the
  reward's box), room_at_cross (median yd to the nearest defender at the crossing).
- gap_if_box / gap_if_out: mean best_gap when he then crossed in the box / outside. A runner
  who reads the gap goes inside more when it is wide.

DL: break through. dl_x_15: mean yd past the line at 1.5 s, toward the backfield.
LB: read, no charging. lb_early: share of LB-steps in the backfield while the runner is 2+ yd
  behind the line and the LB is 2.5+ yd from him. lb_early_free: the same without an LB chasing a
  runner who has gone wide (`TeamSim.lb_chasing_wide`). lb_x_15: same as dl_x_15.

    uv run python -m scripts.role_report experiments/EXP-024-explore/B --gen 12 24 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.ol_reblock import reblock
from sim.team import CARRIER, TeamSim
from sim.team_policies import box_offense, gap_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config


def role_stats(offense, defense, cfg, n: int = 1024, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    los = sim.los.copy()
    start = sim.pos[:, CARRIER].copy()
    blk, dfn = np.asarray(sim.blk), np.asarray(sim.dfn)
    dl = dfn[:4]
    lb = dfn[4:]
    snap = int(round(cfg.snap_time / cfg.field.dt))
    behind = np.ones(n, bool)     # the play is running and the runner hasn't crossed the line
    prev = start.copy()
    path = np.zeros(n)
    fin = np.zeros(n, bool)
    yards, outcome = np.zeros(n), np.zeros(n, int)
    t_line = np.full(n, np.nan)
    y_line = np.full(n, np.nan)
    inside = np.full(n, np.nan)
    room_x = np.full(n, np.nan)
    contact = np.zeros(n, bool)
    lane = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    ol_steps = ol_run = lb_steps = lb_early = lb_early_free = 0
    dl_15, lb_15 = [], []
    t = 0
    while not fin.all():
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        t += 1
        pos, part = info["pos"], info["partner"]
        new = done & ~fin
        yards[new], outcome[new] = info["yards"][new], info["outcome"][new]
        fin |= done
        rb = pos[:, CARRIER]
        path[behind] += np.linalg.norm(rb - prev, axis=-1)[behind]
        prev = rb.copy()
        to_line = los - rb[:, 0]
        dist = np.linalg.norm(pos[:, dfn] - rb[:, None], axis=-1)

        contact |= behind & (to_line >= 1.0) & (dist.min(axis=1) < 1.5)

        # Lane when the runner is first 2 yd from the line.
        at = behind & np.isnan(lane) & (to_line <= 2.0)
        for i in np.nonzero(at)[0]:
            near = pos[i, dfn][(pos[i, dfn, 0] >= rb[i, 0]) & (pos[i, dfn, 0] <= los[i] + 3.0)]
            lane[i] = np.abs(near[:, 1] - rb[i, 1]).min() if len(near) else 10.0
            ys = np.sort(near[:, 1]) if len(near) else np.zeros(1)
            gap[i] = np.diff(ys).max() if len(ys) > 1 else 0.0

        crossed = behind & (to_line <= 0.0)
        t_line[crossed] = t * cfg.field.dt
        y_line[crossed] = rb[crossed, 1]
        room_x[crossed] = dist.min(axis=1)[crossed]
        if crossed.any():
            ol_y = pos[:, blk, 1]
            inside[crossed] = ((rb[:, 1] >= ol_y.min(axis=1)) & (rb[:, 1] <= ol_y.max(axis=1)))[crossed]

        if t > snap:
            vx = sim.vel[:, blk, 0]
            free_d = part[:, dfn] < 0
            d_ol = np.linalg.norm(pos[:, blk, None] - pos[:, dfn, None].transpose(0, 2, 1, 3), axis=-1)
            nobody = ~np.any((d_ol < 2.5) & free_d[:, None], axis=2)
            running = (part[:, blk] < 0) & (pos[:, blk, 0] - los[:, None] >= 1.0) & (vx > 0) & nobody
            ol_steps += int(behind.sum()) * len(blk)
            ol_run += int(running[behind].sum())
            room = behind & (to_line >= 2.0)
            d_lb = np.linalg.norm(pos[:, lb] - rb[:, None], axis=-1)
            early = (pos[:, lb, 0] < los[:, None]) & (d_lb >= 2.5)
            lb_steps += int(room.sum()) * len(lb)
            lb_early += int(early[room].sum())
            lb_early_free += int((early & ~sim.lb_chasing_wide())[room].sum())
        if t == 15:
            dl_15 = los[behind, None] - pos[behind][:, dl, 0]
            lb_15 = los[behind, None] - pos[behind][:, lb, 0]
        behind &= ~(crossed | done)

    crossed_any = ~np.isnan(t_line)
    straight = np.maximum(los - start[:, 0], 1e-6)
    in_box = np.abs(y_line - cfg.field.width / 2) <= cfg.reward.lane_box
    out_box = crossed_any & ~in_box
    return {
        "yards": float(yards.mean()), "td": float((outcome == 1).mean()), "timeout": float((outcome == 4).mean()),
        "ol_free_run": ol_run / max(ol_steps, 1),
        "backfield_contact": float(contact.mean()),
        "lane_ahead": float(np.nanmedian(lane)), "best_gap": float(np.nanmedian(gap)),
        "path_ratio": float(np.median(path[crossed_any] / straight[crossed_any])) if crossed_any.any() else float("nan"),
        "time_to_line": float(np.nanmedian(t_line)),
        "inside": float(np.nanmean(inside)),
        "right": float(np.nanmean(y_line > cfg.field.width / 2)) if crossed_any.any() else float("nan"),
        "box": float(in_box[crossed_any].mean()) if crossed_any.any() else float("nan"),
        "room_at_cross": float(np.nanmedian(room_x)),
        "gap_if_box": float(np.nanmean(gap[in_box])) if in_box.any() else float("nan"),
        "gap_if_out": float(np.nanmean(gap[out_box])) if out_box.any() else float("nan"),
        "reach_line": float(crossed_any.mean()),
        "dl_x_15": float(np.mean(dl_15)) if len(dl_15) else float("nan"),
        "lb_x_15": float(np.mean(lb_15)) if len(lb_15) else float("nan"),
        "lb_early": lb_early / max(lb_steps, 1),
        "lb_early_free": lb_early_free / max(lb_steps, 1),
    }


def report(off, dfn, cfg, n: int) -> dict:
    return {**role_stats(off, dfn, cfg, n=n), "reblock_1s": reblock(off, dfn, cfg, n=n)["reblock_1s"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    out = {"스크립트": report(lambda s, o=None: box_offense(s), lambda s, o=None: gap_defense(s), cfg, args.n)}
    for g in args.gen:
        rows = {}
        for r in sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()):
            ck = r / "checkpoints"
            rows[r.name] = report(ppo_runner(ck / f"runner_g{g:02d}.zip"), ppo_tackler(ck / f"tackler_g{g:02d}.zip"), cfg, args.n)
        out[f"g{g}"] = rows
    keys = list(out["스크립트"])
    print(f"{'':10}" + "".join(f"{k[:11]:>12}" for k in keys))
    print(f"{'script':10}" + "".join(f"{out['스크립트'][k]:12.2f}" for k in keys))
    for g in args.gen:
        for name, row in out[f"g{g}"].items():
            print(f"{f'g{g} {name}':10}" + "".join(f"{row[k]:12.2f}" for k in keys), flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

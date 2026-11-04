"""After a lineman's block ends, does he block anyone again?

Same-seed AI pairs plus the scripted baseline, until the runner crosses the line
(or the play ends):

- sheds per play: an OL's block ended with the play still going (mostly the DL
  shedding; line_hold splits the causes). The step that ends the play is not counted.
- reblocked: he engaged someone again before the runner reached the line.
  `reblock_1s` only counts sheds with 1+ s of live play left, so a shed right
  before the runner arrives does not count against him.
- upfield_after_shed: of the free OL-steps 1+ yd downfield, share by an OL who
  had been blocking earlier in the play; upfield_moving_up: share still moving
  upfield (vx > 0), and their mean vx
- ai_line / script_line: mean yards with the AI's own line vs the scripted line
  in front of the same AI runner, same defense

    uv run python -m scripts.ol_reblock experiments/EXP-021-hold --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.team import CARRIER, TeamSim
from sim.team_policies import box_blockers, box_offense, gap_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config


def reblock(offense, defense, cfg, n: int = 1024, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    blk = sim.blk
    live = np.ones(n, bool)
    prev = np.full((n, len(blk)), -1)
    ever = np.zeros((n, len(blk)), bool)
    sheds = []                                      # [play, OL, shed step, reblock step or -1]
    open_row = {}
    end = np.full(n, -1)
    up, up_prior, up_moving, up_vx = 0, 0, 0, 0.0
    snap = int(round(cfg.snap_time / cfg.field.dt))
    t = 0
    while live.any() and t <= 100:
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        t += 1
        pos, part = info["pos"], info["partner"]
        if t > snap:
            eng = part[:, blk] >= 0
            for i, j in zip(*np.nonzero(prev >= 0)):
                if live[i] and not done[i] and not eng[i, j]:
                    open_row[(i, j)] = len(sheds)
                    sheds.append([int(i), int(j), t, -1])
            for i, j in zip(*np.nonzero((prev < 0) & eng)):
                if live[i] and (i, j) in open_row:
                    sheds[open_row.pop((i, j))][3] = t
            free_up = ~eng & live[:, None] & (pos[:, blk, 0] - sim.los[:, None] >= 1.0)
            vx = sim.vel[:, blk, 0]
            up += int(free_up.sum())
            up_prior += int((free_up & ever).sum())
            up_moving += int((free_up & (vx > 0)).sum())
            up_vx += float(vx[free_up].sum())
            ever |= eng & live[:, None]
            prev = np.where(eng, part[:, blk], -1)
        over = live & (done | (pos[:, CARRIER, 0] >= sim.los))
        end[over] = t
        live &= ~over
    end[end < 0] = t
    s = np.array(sheds, dtype=np.int64).reshape(-1, 4)
    back = s[:, 3] >= 0
    room = end[s[:, 0]] - s[:, 2] >= 10                  # 1+ s of live play left (10 Hz)
    gap = (s[back, 3] - s[back, 2]) * cfg.field.dt
    return {"sheds_per_play": len(s) / n,
            "reblock": float(back.mean()) if len(s) else float("nan"),
            "reblock_1s": float(back[room].mean()) if room.any() else float("nan"),
            "reblock_median_s": float(np.median(gap)) if len(gap) else float("nan"),
            "upfield_after_shed": up_prior / max(up, 1),
            "upfield_moving_up": up_moving / max(up, 1),
            "upfield_mean_vx": up_vx / max(up, 1)}


def mean_yards(offense, defense, cfg, n: int = 2048, seed: int = 7) -> float:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    yards = np.zeros(n)
    finished = np.zeros(n, bool)
    while not finished.all():
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        new = done & ~finished
        yards[new] = info["yards"][new]
        finished |= done
    return float(yards.mean())


def with_scripted_line(runner):
    """The AI's runner behind the scripted line."""
    def act(sim, obs):
        a = np.array(runner(sim, obs)).reshape(sim.n, sim.n_off, 3)
        pos, _, _ = sim.perceived("offense")
        a[:, sim.blk] = box_blockers(sim, pos, 2.5)
        return a.reshape(sim.n, -1)
    return act


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    out = {"스크립트": reblock(lambda s, o=None: box_offense(s), lambda s, o=None: gap_defense(s), cfg, n=args.n)}
    for r in sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()):
        ck = r / "checkpoints"
        off = ppo_runner(ck / f"runner_g{args.gen:02d}.zip")
        dfn = ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip")
        out[r.name] = {**reblock(off, dfn, cfg, n=args.n),
                       "ai_line": mean_yards(off, dfn, cfg), "script_line": mean_yards(with_scripted_line(off), dfn, cfg)}
    for k, v in out.items():
        print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

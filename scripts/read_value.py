"""Is there anything to read at the snap? (after EXP-027)

`box-mix` draws one of four DL fronts each play and slides both LBs up to 3 yd. If the best
place to cross the line changes with the front, a runner who reads it gains yards; if not,
one fixed path is as good as reading, and no reward or league change will make him read.

Runner: the dumb one from `scripts.lane_check` that picks a point on the line at the snap
and then runs straight ahead, with the team's own blockers, against its defense. One point
per run, ten points across the line, every front alone.
Groups: front (4) x LB slide (left < -1 yd, middle, right > 1 yd), seen at the snap.
read value: yards when each group gets its own best point, minus yards of the one best point
for every play. Split half: the point is chosen on even plays and scored on odd plays and the
other way round, so luck in picking a maximum doesn't count as value.
This only reads at the snap; a runner who reads after it could gain more.

The AI runner on the same fronts: its yards and where it crosses, per group, show whether it
changes with the front at all.

    PYTHONPATH=. uv run python -m scripts.read_value experiments/EXP-027-league/A --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from scripts.lane_check import dumb_runner
from sim.team import CARRIER, FRONTS, TeamSim
from sim.team_policies import box_offense, gap_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config

TARGETS = (-7.5, -5.0, -3.2, -1.95, -0.65, 0.65, 1.95, 3.2, 5.0, 7.5)   # yd from the middle; OL at 0, +-1.3, +-2.6
SLIDE = ("LB 왼쪽", "LB 가운데", "LB 오른쪽")


def one_front(cfg, i: int):
    return dataclasses.replace(cfg, formation=dataclasses.replace(cfg.formation, fronts=(FRONTS[i],)))


def run(off, dfn, cfg, n: int, seed: int) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    w = cfg.field.width / 2
    slide = sim.pos[:, np.asarray(sim.dfn)[4:], 1].mean(axis=1) - w
    los = sim.los.copy()
    fin = np.zeros(n, bool)
    behind = np.ones(n, bool)
    yards = np.zeros(n)
    y_cross = np.full(n, np.nan)
    while not fin.all():
        obs, _, done, info = sim.step(off(sim, obs), dfn(sim, obs))
        rb = info["pos"][:, CARRIER]
        c = behind & ~fin & (rb[:, 0] >= los)
        y_cross[c] = rb[c, 1] - w
        behind &= ~c & ~done
        new = done & ~fin
        yards[new] = info["yards"][new]
        fin |= done
    return {"yards": yards, "slide": np.digitize(slide, [-1.0, 1.0]), "y": y_cross}


def at(y: float):
    return lambda sim: np.full(sim.n, sim.cfg.field.width / 2 + y)


def read_value(table: np.ndarray, groups: np.ndarray) -> dict:
    """table (front, target, play) yards, groups (front, play) slide bin; same plays per target."""
    n = table.shape[-1]
    halves = [np.arange(n) % 2 == 0, np.arange(n) % 2 == 1]
    fixed, read = [], []
    for pick, score in (halves, halves[::-1]):
        mean_pick = table[..., pick].mean(axis=(0, 2))
        best_t = int(np.argmax(mean_pick))
        fixed.append(table[:, best_t][:, score].mean())
        got, cnt = 0.0, 0
        for f in range(table.shape[0]):
            for g in range(3):
                mp = pick & (groups[f] == g)
                ms = score & (groups[f] == g)
                if mp.sum() == 0 or ms.sum() == 0:
                    continue
                t = int(np.argmax(table[f][:, mp].mean(axis=1)))
                got += table[f, t, ms].sum()
                cnt += ms.sum()
        read.append(got / cnt)
    return {"best_fixed": float(np.mean(fixed)), "read_at_snap": float(np.mean(read)),
            "read_value": float(np.mean(read) - np.mean(fixed))}


def measure(name: str, off, dfn, ai_runner, cfg, n: int) -> dict:
    table = np.zeros((4, len(TARGETS), n))
    groups = np.zeros((4, n), int)
    ai = []
    for f in range(4):
        c = one_front(cfg, f)
        for k, y in enumerate(TARGETS):
            r = run(dumb_runner(off, at(y)), dfn, c, n, seed=100 + f)   # same starts for every target
            table[f, k] = r["yards"]
            groups[f] = r["slide"]
        if ai_runner is not None:
            ai.append(run(ai_runner, dfn, c, n, seed=100 + f))
    out = {"name": name, **read_value(table, groups), "by_target": table.mean(axis=(0, 2)).round(2).tolist(), "groups": {}}
    for f in range(4):
        for g in range(3):
            m = groups[f] == g
            if not m.any():
                continue
            means = table[f][:, m].mean(axis=1)
            row = {"share": float(m.mean() / 4), "best_target": TARGETS[int(np.argmax(means))],
                   "best_yards": float(means.max()), "by_target": means.round(2).tolist()}
            if ai:
                y = ai[f]["y"][ai[f]["slide"] == g]
                got = ~np.isnan(y)
                row["ai_yards"] = float(ai[f]["yards"][ai[f]["slide"] == g].mean())
                row["ai_cross_median"] = float(np.median(y[got])) if got.any() else float("nan")
                row["ai_box"] = float((np.abs(y[got]) <= cfg.reward.lane_box).mean()) if got.any() else float("nan")
            out["groups"][f"front{f} {SLIDE[g]}"] = row
    if ai:
        out["ai_yards"] = float(np.mean([a["yards"].mean() for a in ai]))
    return out


def show(o: dict) -> None:
    print(f"\n== {o['name']}: best fixed {o['best_fixed']:.2f} yd, read at snap {o['read_at_snap']:.2f} yd, "
          f"read value {o['read_value']:+.2f} yd" + (f", AI runner {o['ai_yards']:.2f} yd" if "ai_yards" in o else ""))
    print("   all fronts by target " + " ".join(f"{t:+.1f}:{v:.2f}" for t, v in zip(TARGETS, o["by_target"])))
    for k, r in o["groups"].items():
        ai = f" | AI {r['ai_yards']:.2f} yd, crosses at {r['ai_cross_median']:+.1f}, box {r['ai_box']*100:.0f}%" if "ai_yards" in r else ""
        print(f"   {k:16} {r['share']*100:4.1f}% best {r['best_target']:+.2f} -> {r['best_yards']:.2f} yd"
              f" [{' '.join(f'{v:5.2f}' for v in r['by_target'])}]{ai}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    outs = [measure("스크립트 블로킹 vs 스크립트 갭 수비", lambda s, o=None: box_offense(s), lambda s, o=None: gap_defense(s),
                    lambda s, o=None: box_offense(s), cfg, args.n)]
    show(outs[-1])
    for s in range(3):
        ck = args.exp / f"s{s}" / "checkpoints"
        off = ppo_runner(ck / f"runner_g{args.gen:02d}.zip")
        outs.append(measure(f"s{s} g{args.gen}", off, ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip"), off, cfg, args.n))
        show(outs[-1])
    if args.json:
        args.json.write_text(json.dumps(outs, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

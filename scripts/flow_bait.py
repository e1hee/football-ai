"""Is the defense's flow honest, or a bait? (EXP-036)

A runner who reads the flow (`sim.team_policies.flow_offense`: at `read_at` s, aim 5 yd to the side
away from where the defense has moved since the snap) behind the AI offense's own blockers, against
the AI defense; and the same runner going the side the defense moved to.
flow value: reader yards minus with-flow yards. > 0: the side the defense leaves is the better one
(honest flow, worth reading). < 0: the defense leans one way and is better there than it looks
(a bait). lean: mean |y shift| of the six defenders at `read_at`, yd.

`--min-lean` (EXP-036's opponent): the AI runner as it is, cutting away from the flow only when the
defense has moved at least that far by `read_at` (`with_flow_read`). trigger: share of plays it
cuts. cut value: on those plays, its yards minus the same cut toward the flow. A defense that
learned to bait it leans on purpose (trigger up) toward the side it is not defending (cut value < 0).

    PYTHONPATH=. uv run python -m scripts.flow_bait experiments/EXP-034-overload/A --gen 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.read_value import run
from sim.team import CARRIER, TeamSim
from sim.team_policies import flow_offense, overload_defense, with_flow_read
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config


def reader(off, read_at: float, with_flow: bool = False):
    """The AI offense's blockers, the flow-reading runner."""
    def f(sim, obs=None):
        a = np.array(off(sim, obs)).reshape(sim.n, sim.n_off, 3)
        a[:, CARRIER] = flow_offense(sim, read_at=read_at, with_flow=with_flow).reshape(sim.n, sim.n_off, 3)[:, CARRIER]
        return a.reshape(sim.n, -1)
    return f


def lean(off, dfn, cfg, read_at: float, n: int, seed: int) -> float:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    y0 = sim.pos[:, sim.dfn, 1].mean(axis=1).copy()
    for _ in range(int(round(read_at / cfg.field.dt))):
        obs, _, _, _ = sim.step(off(sim, obs), dfn(sim, obs))
    return float(np.abs(sim.pos[:, sim.dfn, 1].mean(axis=1) - y0).mean())


def check(off, dfn, cfg, read_at: float = 1.0, n: int = 1024, seed: int = 13) -> dict:
    r = run(reader(off, read_at), dfn, cfg, n, seed)["yards"].mean()
    w = run(reader(off, read_at, True), dfn, cfg, n, seed)["yards"].mean()
    ai = run(off, dfn, cfg, n, seed)["yards"].mean()
    return {"reader": float(r), "with_flow": float(w), "flow_value": float(r - w), "ai": float(ai),
            "lean": lean(reader(off, read_at), dfn, cfg, read_at, n, seed)}


def play(off, dfn, cfg, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Yards per play, and whether the flow read cut on that play."""
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    fin = np.zeros(n, bool)
    yards = np.zeros(n)
    cut = np.zeros(n, bool)
    while not fin.all():
        a = off(sim)
        cut |= ~fin & (sim._flow_side != 0)
        obs, _, done, info = sim.step(a, dfn(sim, obs))
        new = done & ~fin
        yards[new] = info["yards"][new]
        fin |= done
    return yards, cut


def hybrid(off, dfn, cfg, read_at: float, min_lean: float, n: int = 1024, seed: int = 13) -> dict:
    base = lambda sim: off(sim, sim.observe())
    y, cut = play(with_flow_read(base, read_at, min_lean), dfn, cfg, n, seed)
    w, cut_w = play(with_flow_read(base, read_at, min_lean, with_flow=True), dfn, cfg, n, seed)
    both = cut & cut_w
    return {"hybrid": float(y.mean()), "trigger": float(cut.mean()),
            "cut_value": float(y[both].mean() - w[both].mean()) if both.any() else float("nan"),
            "cut_yards": float(y[both].mean()) if both.any() else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--read-at", type=float, default=1.0)
    p.add_argument("--min-lean", type=float, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    rows = []
    over = lambda s, o=None: overload_defense(s)
    for g in args.gen:
        for s in range(3):
            ck = args.exp / f"s{s}" / "checkpoints"
            off = ppo_runner(ck / f"runner_g{g:02d}.zip")
            for name, dfn in (("AI", ppo_tackler(ck / f"tackler_g{g:02d}.zip")), ("overload", over)):
                r = {"seed": s, "gen": g, "defense": name, **check(off, dfn, cfg, args.read_at)}
                if args.min_lean is not None:
                    r.update(hybrid(off, dfn, cfg, args.read_at, args.min_lean))
                rows.append(r)
                print(f"s{s} g{g:02d} vs {name:8} | flow reader {r['reader']:5.2f} yd, with the flow {r['with_flow']:5.2f} "
                      f"-> flow value {r['flow_value']:+.2f} | AI runner {r['ai']:5.2f} | lean at {args.read_at}s {r['lean']:.2f} yd"
                      + (f" | AI+read {r['hybrid']:5.2f} yd, cuts {r['trigger']*100:3.0f}%, cut value {r['cut_value']:+.2f}"
                         if "hybrid" in r else ""), flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

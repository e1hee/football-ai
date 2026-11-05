"""EXP-026: was the lane bonus paid, and is the middle worth going to at all?

paid: the reward difference to a twin sim with the bonus off (same seed, same actions, so
the same play). by place: outcome of plays that crossed in the box vs outside.
straight: the AI's team and defense, with the runner replaced by a dumb one that aims for a
point on the line and then runs straight ahead. The same dumb runner aimed at the middle and
at the sideline tells whether the middle is open, without the AI runner's habits.

    PYTHONPATH=. uv run python -m scripts.lane_check experiments/EXP-026-lane/A --gen 1 3 6 12
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from sim.team import CARRIER, TeamSim
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config


def twin_cfg(cfg):
    return dataclasses.replace(cfg, reward=dataclasses.replace(cfg.reward, lane_bonus=0.0))


def paid_and_place(off, dfn, cfg, n: int, seed: int = 7) -> dict:
    sim, twin = TeamSim(n, cfg, seed=seed), TeamSim(n, twin_cfg(cfg), seed=seed)
    obs = sim.reset()
    twin.reset()
    los = sim.los.copy()
    fin = np.zeros(n, bool)
    paid = np.zeros(n)
    y_line = np.full(n, np.nan)
    yards, td = np.zeros(n), np.zeros(n, bool)
    behind = np.ones(n, bool)
    while not fin.all():
        a, d = off(sim, obs), dfn(sim, obs)
        obs, rew, done, info = sim.step(a, d)
        _, rew0, _, info0 = twin.step(a, d)
        assert np.allclose(info["pos"], info0["pos"]), "twin diverged"
        paid[~fin] += (rew - rew0)[~fin]
        rb = info["pos"][:, CARRIER]
        crossed = behind & ~fin & (rb[:, 0] >= los)
        y_line[crossed] = rb[crossed, 1]
        behind &= ~crossed
        new = done & ~fin
        yards[new], td[new] = info["yards"][new], info["outcome"][new] == 1
        behind &= ~done
        fin |= done
    got = ~np.isnan(y_line)
    box = got & (np.abs(y_line - cfg.field.width / 2) <= cfg.reward.lane_box)
    out = got & ~box
    mean = lambda v, m: float(v[m].mean()) if m.any() else float("nan")
    return {"paid_per_play": float(paid.mean()), "paid_share": float((paid > 1e-6).mean()),
            "paid_if_paid": mean(paid, paid > 1e-6), "reach": float(got.mean()), "box": float(box.sum() / max(got.sum(), 1)),
            "yards": float(yards.mean()), "yards_box": mean(yards, box), "yards_out": mean(yards, out),
            "td_box": mean(td, box), "td_out": mean(td, out)}


def dumb_runner(off, target_y):
    """The AI's own blockers, with the runner aiming at target_y(sim) on the line, then straight ahead."""
    def f(sim, obs):
        a = np.array(off(sim, obs)).reshape(sim.n, sim.n_off, 3)
        rb = sim.pos[:, CARRIER]
        goal = np.stack([sim.los + 1.0, target_y(sim)], axis=1)
        v = goal - rb
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        v[rb[:, 0] >= sim.los] = (1.0, 0.0)
        a[:, CARRIER, :2] = v
        a[:, CARRIER, 2] = 0.0
        return a.reshape(sim.n, -1)
    return f


def middle(sim):
    return np.full(sim.n, sim.cfg.field.width / 2)


def sideline(sim):
    # 2.5 yd in from the sideline nearer the runner
    w = sim.cfg.field.width
    return np.where(sim.pos[:, CARRIER, 1] > w / 2, w - 2.5, 2.5)


def run_dumb(off, dfn, cfg, n: int, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    los = sim.los.copy()
    fin = np.zeros(n, bool)
    behind = np.ones(n, bool)
    yards, td = np.zeros(n), np.zeros(n, bool)
    got = np.zeros(n, bool)
    while not fin.all():
        obs, _, done, info = sim.step(off(sim, obs), dfn(sim, obs))
        rb = info["pos"][:, CARRIER]
        crossed = behind & ~fin & (rb[:, 0] >= los)
        got |= crossed
        behind &= ~crossed
        new = done & ~fin
        yards[new], td[new] = info["yards"][new], info["outcome"][new] == 1
        behind &= ~done
        fin |= done
    return {"reach": float(got.mean()), "yards": float(yards.mean()), "td": float(td.mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, nargs="+", default=[12])
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    out = {}
    for g in args.gen:
        for r in sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()):
            ck = r / "checkpoints"
            o, d = ppo_runner(ck / f"runner_g{g:02d}.zip"), ppo_tackler(ck / f"tackler_g{g:02d}.zip")
            row = paid_and_place(o, d, cfg, args.n)
            row["dumb_middle"] = run_dumb(dumb_runner(o, middle), d, cfg, args.n)
            row["dumb_side"] = run_dumb(dumb_runner(o, sideline), d, cfg, args.n)
            out[f"g{g} {r.name}"] = row
            m, s = row["dumb_middle"], row["dumb_side"]
            print(f"g{g:<2} {r.name} paid/play {row['paid_per_play']:.3f} ({row['paid_share']*100:.1f}% of plays, {row['paid_if_paid']:.3f} each) "
                  f"box {row['box']*100:.1f}% | yd {row['yards']:.2f} box {row['yards_box']:.2f} out {row['yards_out']:.2f} "
                  f"td box {row['td_box']*100:.0f}% out {row['td_out']*100:.0f}% | "
                  f"dumb middle yd {m['yards']:.2f} reach {m['reach']*100:.0f}% td {m['td']*100:.0f}% | "
                  f"dumb side yd {s['yards']:.2f} reach {s['reach']*100:.0f}% td {s['td']*100:.0f}%", flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

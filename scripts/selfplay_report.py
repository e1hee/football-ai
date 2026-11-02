"""Self-play report: who beats whom across generations, and how the tackler plays.

1. Cross table: every runner generation vs every tackler generation
   (touchdown rate, deterministic, same 1024 starts). If self-play is making
   real progress, later tacklers beat *all* earlier runners, not just the last.
2. Tackler style per generation, vs the EXP-002 champion runner (R00):
   - inside leverage: share of close-range time the tackler is nearer the
     middle of the field than the runner (forcing him toward the sideline)
   - goal-side: share of time the tackler is between runner and goal line
   - where plays end: median distance from the runner to the nearest sideline
   - out-of-bounds rate

    uv run python -m scripts.selfplay_report experiments/EXP-003-selfplay-v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from sim.core import OUTCOME_OUT_OF_BOUNDS, OUTCOME_TOUCHDOWN, FieldConfig, OneVOneSim, SimConfig
from train.evaluate import evaluate, ppo_runner, ppo_tackler

for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()


def style(runner, tackler, cfg: SimConfig, n: int = 2048, seed: int = 777) -> dict:
    sim = OneVOneSim(n, cfg, seed=seed)
    obs = sim.reset()
    mid = cfg.field.width / 2
    live = np.ones(n, bool)
    close_t = inside_t = goal_t = all_t = 0
    end_side = np.zeros(n)
    outcome = np.zeros(n, np.int8)
    while live.any():
        d = np.linalg.norm(sim.r_pos - sim.t_pos, axis=1)
        close = live & (d < 8.0)
        inside = np.abs(sim.t_pos[:, 1] - mid) < np.abs(sim.r_pos[:, 1] - mid)
        goal = sim.t_pos[:, 0] > sim.r_pos[:, 0]
        close_t += close.sum()
        inside_t += (close & inside).sum()
        goal_t += (live & goal).sum()
        all_t += live.sum()
        obs, _, done, info = sim.step(runner(sim, obs), tackler(sim))
        newly = done & live
        ry = info["r_pos"][:, 1]
        end_side[newly] = np.minimum(ry, cfg.field.width - ry)[newly]
        outcome[newly] = info["outcome"][newly]
        live &= ~done
    return {
        "inside_leverage": inside_t / max(close_t, 1),
        "goal_side": goal_t / max(all_t, 1),
        "end_dist_to_sideline": float(np.median(end_side)),
        "out_of_bounds": float((outcome == OUTCOME_OUT_OF_BOUNDS).mean()),
        "touchdown": float((outcome == OUTCOME_TOUCHDOWN).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--spread", type=float, default=8.0)
    p.add_argument("--n", type=int, default=1024)
    args = p.parse_args()
    ck = args.exp / "checkpoints"
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=args.spread))
    rs = sorted(ck.glob("runner_g*.zip"))
    ts = sorted(ck.glob("tackler_g*.zip"))
    R = [ppo_runner(p) for p in rs]
    T = [ppo_tackler(p) for p in ts]

    table = np.zeros((len(R), len(T)))
    for i, r in enumerate(R):
        for j, t in enumerate(T):
            table[i, j] = evaluate(r, t, cfg, n_plays=args.n)["rate/touchdown"]
    np.savetxt(args.exp / "crosstable.csv", table, delimiter=",", fmt="%.4f",
               header="rows = runner gen, cols = tackler gen, value = touchdown rate")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor("#18181f")
    im = ax.imshow(table * 100, cmap="RdYlGn", vmin=0, vmax=100, origin="upper")
    for i in range(len(R)):
        for j in range(len(T)):
            ax.text(j, i, f"{table[i, j] * 100:.0f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_xlabel("AI 수비 세대", color="white", fontsize=13)
    ax.set_ylabel("AI 러너 세대", color="white", fontsize=13)
    ax.set_xticks(range(len(T)))
    ax.set_yticks(range(len(R)))
    ax.tick_params(colors="white")
    ax.set_title("터치다운률 (%) — 초록 = 러너 승, 빨강 = 수비 승", color="white", fontsize=14, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(args.exp / "crosstable.png", facecolor=fig.get_facecolor())

    champion = R[0]
    styles = {}
    print("tackler | TD vs R00 | inside leverage | goal-side | end dist to sideline | out of bounds")
    for j, t in enumerate(T):
        s = style(champion, t, cfg)
        styles[f"T{j:02d}"] = s
        print(f"   T{j:02d} | {s['touchdown']:9.1%} | {s['inside_leverage']:15.0%} | {s['goal_side']:9.0%} | "
              f"{s['end_dist_to_sideline']:20.1f} | {s['out_of_bounds']:13.1%}")
    (args.exp / "tackler_style.json").write_text(json.dumps(styles, indent=1))
    print("\ncross table (rows runner gen, cols tackler gen):")
    print("      " + " ".join(f"T{j:02d}" for j in range(len(T))))
    for i in range(len(R)):
        print(f"R{i:02d}  " + " ".join(f"{table[i, j] * 100:3.0f}" for j in range(len(T))))


if __name__ == "__main__":
    main()

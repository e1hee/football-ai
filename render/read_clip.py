"""Side-by-side read test clip: the same play twice, the second with the LBs
slid sideways at the snap. Defense frozen in place (as in scripts/read_test),
so any change in the runner's path is his own read. Rules come from the run.

    uv run python -m render.read_clip experiments/EXP-020-tackle/s0 --gen 12 --slide 3 --play 45 \
        --out experiments/EXP-020-tackle/clips/s0_read_lb_slide.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from train.evaluate import ppo_runner
from train.run_config import config_from_run
from .clip import HOLD_END_S, OUT_FPS, OUT_SIZE, SIM_HZ, outcome_tag, write_mp4
from .frames import Layout, fit_canvas, stack
from .team_clip import TeamRenderer, record_team_plays


def slide_lbs(dy: float):
    def setup(sim):
        roles = np.array(sim.roles)
        sim.pos[:, sim.dfn[roles[sim.dfn] == "LB"], 1] += dy
    return setup


def still(sim, obs=None):
    return np.zeros((sim.n, sim.n_def * 3), np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path, help="experiments/EXP-NNN/sK")
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--slide", type=float, default=3.0, help="yd the LBs slide (+ = right)")
    p.add_argument("--play", type=int, default=0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--format", choices=["shorts", "episode"], default="shorts")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = config_from_run(args.run)
    cfg = dataclasses.replace(cfg, formation=dataclasses.replace(cfg.formation, fronts=None, lb_shift=0.0))
    off = ppo_runner(args.run / "checkpoints" / f"runner_g{args.gen:02d}.zip")
    a = record_team_plays(off, still, cfg, n_plays=args.n, seed=args.seed)[args.play]
    side = "오른쪽" if args.slide > 0 else "왼쪽"
    b = record_team_plays(off, still, cfg, n_plays=args.n, seed=args.seed, setup=slide_lbs(args.slide))[args.play]

    base = Layout.shorts() if args.format == "shorts" else Layout.episode()
    layout = Layout(vertical=base.vertical, px_per_yd=10, scale=4)
    view = dataclasses.replace(cfg.field, width=max(15.0, cfg.field.width + 6.0))
    shift = (view.width - cfg.field.width) / 2
    lane = (shift, shift + cfg.field.width, cfg.field.runner_start_x + cfg.formation.los_ahead)
    for pl in (a, b):
        pl.pos[..., 1] += shift
    renderer = TeamRenderer(view, layout, lane)
    size = OUT_SIZE[args.format]
    run = args.run.name
    titles = [f"AI {run} · LB 원래 자리", f"AI {run} · LB {side}으로 {abs(args.slide):g}야드"]

    def frames():
        longest = max(a.steps, b.steps)
        sub = OUT_FPS // SIM_HZ
        for k in range(longest * sub + 1 + int(HOLD_END_S * OUT_FPS)):
            parts = []
            for pl, t in zip((a, b), titles):
                s = min(k / sub, pl.steps)
                parts.append(renderer.frame(pl, s, t, outcome_tag(pl) if s >= pl.steps else None))
            yield fit_canvas(stack(parts, vertical=layout.vertical), size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = write_mp4(frames(), args.out, size)
    print(f"{n} frames ({n / OUT_FPS:.1f}s) -> {args.out}  play #{args.play}: "
          f"{a.outcome} {a.yards:.1f} / {b.outcome} {b.yards:.1f}")


if __name__ == "__main__":
    main()

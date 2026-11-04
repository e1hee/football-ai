"""Side-by-side line clip: the same AI runner against the same AI defense from
the same start, behind his own AI line (top) and behind the scripted line (bottom).
Rules come from the run. Plays diverge after the snap, only the start is shared.

    uv run python -m render.line_clip experiments/EXP-021-hold/s1 --gen 12 --list
    uv run python -m render.line_clip experiments/EXP-021-hold/s1 --gen 12 --play 2 \
        --out experiments/EXP-021-hold/clips/s1_own_line_vs_script.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from scripts.ol_reblock import with_scripted_line
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import config_from_run
from .clip import HOLD_END_S, OUT_FPS, OUT_SIZE, SIM_HZ, outcome_tag, write_mp4
from .frames import Layout, fit_canvas, stack
from .team_clip import TeamRenderer, record_team_plays


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path, help="experiments/EXP-NNN/sK")
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--play", type=int, default=0)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=["shorts", "episode"], default="shorts")
    p.add_argument("--list", action="store_true", help="print both outcomes per play and stop")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = config_from_run(args.run)
    ck = args.run / "checkpoints"
    off = ppo_runner(ck / f"runner_g{args.gen:02d}.zip")
    dfn = ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip")
    plays_a = record_team_plays(off, dfn, cfg, n_plays=args.n, seed=args.seed)
    plays_b = record_team_plays(with_scripted_line(off), dfn, cfg, n_plays=args.n, seed=args.seed)
    if args.list or args.out is None:
        for i, (x, y) in enumerate(zip(plays_a, plays_b)):
            print(f"{i:3d}  AI line {x.outcome:13s} {x.yards:6.1f}   scripted line {y.outcome:13s} {y.yards:6.1f}")
        return
    a, b = plays_a[args.play], plays_b[args.play]

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
    titles = [f"AI {run} 러너 · 자기 AI 라인", f"AI {run} 러너 · 스크립트 라인"]

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

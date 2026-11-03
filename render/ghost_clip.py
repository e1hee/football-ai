"""Ghost overlay: every generation's runner on one field at once, each offense
vs its own defense. Pale = early generation, dark = late. An X marks a tackle;
a runner who reaches the end just leaves his trail.

    uv run python -m render.ghost_clip experiments/EXP-018-snap --sim box-snap --out ghosts.mp4
    uv run python -m render.ghost_clip experiments/EXP-018-snap --run s0 --format shorts --out s0.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.core import FieldConfig, SimConfig, TackleConfig
from sim.team import CARRIER, SIMS, TeamSim, config_for
from train.evaluate import ppo_runner, ppo_tackler
from .clip import HOLD_END_S, OUT_FPS, OUT_SIZE, SIM_HZ, write_mp4
from .frames import Layout, fit_canvas, stack
from .team_clip import BLOCKER, TACKLER, TeamRenderer, record_team_plays

EARLY, LATE = (250, 245, 205), (215, 25, 90)
BG = (24, 24, 32)


def gen_color(i: int, n: int) -> tuple[int, int, int]:
    a = i / max(n - 1, 1)
    return tuple(int(e + (l - e) * a) for e, l in zip(EARLY, LATE))


class GhostRenderer(TeamRenderer):
    def __init__(self, field: FieldConfig, layout: Layout, lane, line: np.ndarray, n_off: int):
        super().__init__(field, layout, lane)
        self.line, self.n_off = line, n_off       # (P, 2) average starting spots

    def ghost_frame(self, gens: list[list], step: float, title: str) -> Image.Image:
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        self._field(d)
        self._lane(d)
        for j, spot in enumerate(self.line):      # the formation at the snap, for reference
            if j != CARRIER:
                x, y = self._xy(*spot)
                d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=BLOCKER if j < self.n_off else TACKLER)
        total = len(gens)
        for g, plays in enumerate(gens):
            color = gen_color(g, total)
            for pl in plays:
                s = min(step, pl.steps)
                i0 = int(s)
                pts = [self._xy(*p) for p in pl.pos[: i0 + 1, CARRIER]]
                a = s - i0
                cur = pl.pos[i0, CARRIER] * (1 - a) + pl.pos[min(i0 + 1, pl.steps), CARRIER] * a
                pts.append(self._xy(*cur))
                d.line(pts, fill=color, width=1)
                x, y = pts[-1]
                if s >= pl.steps and pl.outcome == "tackled":
                    d.line([x - 3, y - 3, x + 3, y + 3], fill=color, width=1)
                    d.line([x - 3, y + 3, x + 3, y - 3], fill=color, width=1)
                else:
                    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
        self._caption(d, f"{title}  {step / SIM_HZ:.1f}초", None)
        # Legend: early to late.
        x0, y0, w = self.L.margin + 4, self.L.caption_h + self.L.margin + 3, 8
        for g in range(total):
            d.rectangle([x0 + g * w, y0, x0 + (g + 1) * w - 1, y0 + 4], fill=gen_color(g, total))
        d.text((x0 + total * w + 4, y0 - 3), f"1→{total}세대", fill=(240, 240, 230), font=self.font_small)
        s = self.L.scale
        return img.resize((self.W * s, self.H * s), Image.NEAREST)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--run", action="append", default=None, help="repeat; default all runs side by side")
    p.add_argument("--sim", choices=[s for s in SIMS if s != "1v1"], default="box-snap")
    p.add_argument("--plays", type=int, default=6, help="plays per generation")
    p.add_argument("--format", choices=["shorts", "episode"], default="episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", action="store_true")
    args = p.parse_args()

    cfg = config_for(args.sim, SimConfig(field=FieldConfig(reaction_delay=2), facing=True,
                                         tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2)),
                     stumble=0.2, carrier_square_ahead=True)
    runs = args.run or sorted(d.name for d in args.exp.iterdir() if (d / "checkpoints").is_dir())
    view = dataclasses.replace(cfg.field, width=cfg.field.width + 6.0)
    shift = 3.0
    lane = (shift, shift + cfg.field.width, cfg.field.runner_start_x + cfg.formation.los_ahead)
    layout = Layout(vertical=True, px_per_yd=10, scale=4 if len(runs) == 1 else 3)
    start = TeamSim(1024, cfg, seed=1)
    start.reset()
    line = start.pos.mean(0) + np.array([0.0, shift])
    renderer = GhostRenderer(view, layout, lane, line, start.n_off)

    panels = []
    for run in runs:
        ck = args.exp / run / "checkpoints"
        gens = []
        for f in sorted(ck.glob("runner_g*.zip")):
            g = f.stem.split("_g")[1]
            if int(g) == 0 or not (ck / f"tackler_g{g}.zip").exists():
                continue
            plays = record_team_plays(ppo_runner(f), ppo_tackler(ck / f"tackler_g{g}.zip"), cfg, n_plays=args.plays,
                                      seed=args.seed)
            for pl in plays:
                pl.pos[..., 1] += shift
            gens.append(plays)
        panels.append((run, gens))

    longest = max(pl.steps for _, gens in panels for plays in gens for pl in plays)
    size = OUT_SIZE[args.format]

    def frames():
        sub = OUT_FPS // SIM_HZ
        for k in range(longest * sub + 1 + int(HOLD_END_S * 2 * OUT_FPS)):
            parts = [renderer.ghost_frame(gens, min(k / sub, longest), run) for run, gens in panels]
            yield fit_canvas(parts[0] if len(parts) == 1 else stack(parts, vertical=False), size)

    if args.png:
        png = args.out.with_suffix(".png")
        parts = [renderer.ghost_frame(gens, longest, run) for run, gens in panels]
        fit_canvas(parts[0] if len(parts) == 1 else stack(parts, vertical=False), size).save(png)
        print(f"preview -> {png}")
    n = write_mp4(frames(), args.out, size)
    print(f"{n} frames ({n / OUT_FPS:.1f}s) -> {args.out}")


if __name__ == "__main__":
    main()

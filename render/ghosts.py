"""Generation ghosts: every generation plays the SAME start at the same time,
each drawn in its own colour on a gradient (early = dark purple, late = bright
yellow). Early generations scatter and get caught; later ones converge.

Unlike the pixel clips, this is a smooth, anti-aliased "data art" style:
dozens of overlapping trails need fine lines to stay readable.

    # animation (9:16) + final-frame poster
    uv run python -m render.ghosts --ckpts experiments/EXP-004-fine-gens-runner/checkpoints \
        --side runner --opponent angle --play-index 5 --format shorts \
        --title "AI 러너 60세대의 진화" --out render/out/ghosts.mp4 --poster

    # static grid of 9 different starts
    uv run python -m render.ghosts --ckpts ... --side runner --opponent angle --grid 9 --out render/out/grid.png
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

from sim.core import FieldConfig, SimConfig
from .clip import add_sim_args, make_policy, sim_config, write_mp4
from .frames import _font
from .record import Play, record_plays

SS = 2                      # supersampling factor for anti-aliasing
BG = (14, 16, 22)
GRASS = (22, 44, 31)
GRASS_ALT = (25, 49, 35)
LINE = (70, 92, 78)
ENDZONE = (30, 40, 70)
TEXT = (235, 235, 230)
SUBTEXT = (150, 160, 170)
SIZES = {"shorts": (1080, 1920), "episode": (1920, 1080), "square": (1080, 1350)}


def gen_colors(n: int, cmap: str = "plasma") -> list[tuple[int, int, int]]:
    cm = colormaps[cmap]
    return [tuple(int(255 * c) for c in cm(0.12 + 0.86 * i / max(n - 1, 1))[:3]) for i in range(n)]


@dataclass
class Box:
    """Where the field sits on the canvas, in supersampled px."""
    x0: float
    y0: float
    w: float
    h: float
    vertical: bool
    field: FieldConfig
    endzone: float = 6.0     # yd drawn past the goal line
    y_off: float = 0.0       # zoomed view: field y shown at the box's left/top edge

    def xy(self, x: float, y: float) -> tuple[float, float]:
        L, W = self.field.length + self.endzone, self.field.width
        y = y - self.y_off
        if self.vertical:   # downfield is up
            return self.x0 + y / W * self.w, self.y0 + (L - x) / L * self.h
        return self.x0 + x / L * self.w, self.y0 + y / W * self.h

    @classmethod
    def fit(cls, field: FieldConfig, x0, y0, w, h, vertical: bool, endzone: float = 6.0, top: bool = False,
            y_off: float = 0.0) -> "Box":
        L, W = field.length + endzone, field.width
        aspect = (W / L) if vertical else (L / W)    # field width / height on screen
        if w / h > aspect:
            nw, nh = h * aspect, h
        else:
            nw, nh = w, w / aspect
        return cls(x0 + (w - nw) / 2, y0 + (0 if top else (h - nh) / 2), nw, nh, vertical, field, endzone, y_off)


def draw_field(d: ImageDraw.ImageDraw, box: Box) -> None:
    f = box.field
    y_lo, y_hi = box.y_off, box.y_off + f.width
    for yd in range(0, int(f.length), 5):
        a, b = box.xy(yd, y_lo), box.xy(min(yd + 5, f.length), y_hi)
        d.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])],
                    fill=GRASS if (yd // 5) % 2 == 0 else GRASS_ALT)
    if box.endzone > 0:
        a, b = box.xy(f.length, y_lo), box.xy(f.length + box.endzone, y_hi)
        d.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], fill=ENDZONE)
    for yd in range(0, int(f.length) + 1, 5):
        d.line([box.xy(yd, y_lo), box.xy(yd, y_hi)], fill=LINE, width=SS * (3 if yd == f.length else 1))
    for sy in (0.0, 53.3):     # real sidelines; skipped when outside a zoomed view
        if box.y_off <= sy <= box.y_off + f.width:
            d.line([box.xy(0, sy), box.xy(f.length + box.endzone, sy)], fill=LINE, width=SS * 2)


def draw_ghosts(d: ImageDraw.ImageDraw, box: Box, plays: list[Play], colors, s: float,
                scale: float = 1.0, show_tackler: bool = True, focus: str = "runner") -> None:
    """Draw every generation's trails up to (fractional) step s, oldest first.
    `focus` is the side that evolves: its trails are bold, the opponent's faint."""
    n = len(plays)
    for g, pl in enumerate(plays):
        c = colors[g]
        late = g / max(n - 1, 1)
        k = int(min(np.floor(s), pl.steps))
        frac = min(s, pl.steps) - k
        def at(arr):
            k1 = min(k + 1, pl.steps)
            return arr[k] * (1 - frac) + arr[k1] * frac
        r_now, t_now = at(pl.r_pos), at(pl.t_pos)
        a_trail = int(60 + 150 * late)
        bold_w, faint_w = int(SS * (2 + 2 * late) * scale), int(SS * 2 * scale)
        r_style = (a_trail, bold_w) if focus == "runner" else (int(a_trail * 0.35), faint_w)
        t_style = (a_trail, bold_w) if focus == "tackler" else (int(a_trail * 0.35), faint_w)
        if show_tackler and k > 0:
            pts = [box.xy(*p) for p in pl.t_pos[: k + 1]] + [box.xy(*t_now)]
            d.line(pts, fill=c + (t_style[0],), width=t_style[1], joint="curve")
        if k > 0:
            pts = [box.xy(*p) for p in pl.r_pos[: k + 1]] + [box.xy(*r_now)]
            d.line(pts, fill=c + (r_style[0],), width=r_style[1], joint="curve")
        # Tackler: hollow ring. Runner: filled dot; ended plays get an outcome mark.
        rr = SS * (5 + 3 * late) * scale
        if show_tackler:
            tx, ty = box.xy(*t_now)
            tr = SS * (5 + (3 * late if focus == "tackler" else 0)) * scale
            d.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], outline=c + (int(90 + 100 * late),),
                      width=int(SS * (2.5 if focus == "tackler" else 1.5) * scale))
        x, y = box.xy(*r_now)
        done = s >= pl.steps
        if done and pl.outcome == "tackled":
            w = rr * 0.9
            d.line([x - w, y - w, x + w, y + w], fill=c + (230,), width=int(SS * 2.5 * scale))
            d.line([x - w, y + w, x + w, y - w], fill=c + (230,), width=int(SS * 2.5 * scale))
        else:
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=c + (int(150 + 100 * late),))


def draw_legend(d: ImageDraw.ImageDraw, x, y, w, h, colors, labels: tuple[str, str], font) -> None:
    n = len(colors)
    for i, c in enumerate(colors):
        d.rectangle([x + w * i / n, y, x + w * (i + 1) / n + 1, y + h], fill=c)
    d.text((x, y + h + SS * 8), labels[0], font=font, fill=SUBTEXT)
    tw = d.textlength(labels[1], font=font)
    d.text((x + w - tw, y + h + SS * 8), labels[1], font=font, fill=SUBTEXT)


def gen_number(path: Path) -> int:
    m = re.findall(r"\d+", path.stem)
    return int(m[-1]) if m else 0


def load_plays(ckpts: list[Path], side: str, opponent: str, cfg: SimConfig, seed: int) -> list[list[Play]]:
    """For each checkpoint, record 32 plays from the same seeded starts."""
    other = make_policy(opponent, seed)
    out = []
    for p in ckpts:
        me = make_policy(("ckpt:" if side == "runner" else "tckpt:") + str(p), seed)
        runner, tackler = (me, other) if side == "runner" else (other, me)
        out.append(record_plays(runner, tackler, n_plays=32, cfg=cfg, seed=seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=Path, required=True, help="checkpoint directory")
    ap.add_argument("--pattern", default="*.zip")
    ap.add_argument("--side", choices=["runner", "tackler"], required=True, help="which side the checkpoints play")
    ap.add_argument("--opponent", required=True, help="fixed other side: scripted name, ckpt:..., or tckpt:...")
    ap.add_argument("--every", type=int, default=1, help="use every k-th generation")
    ap.add_argument("--max-gen", type=int, default=None)
    ap.add_argument("--play-index", type=int, default=0)
    ap.add_argument("--grid", type=int, default=0, help="static poster of N starts instead of an animation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spread", type=float, default=8.0)
    ap.add_argument("--format", choices=list(SIZES), default="shorts")
    ap.add_argument("--speed", type=float, default=0.75, help="playback speed (1 = real time)")
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--view", type=float, default=None, help="zoom: show only the first N yards downfield")
    ap.add_argument("--view-width", type=float, default=None, help="zoom: show N yards across, centred on the start")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poster", action="store_true", help="also save the final frame as PNG")
    add_sim_args(ap)
    args = ap.parse_args()

    ckpts = sorted(args.ckpts.glob(args.pattern), key=gen_number)
    if args.max_gen is not None:
        ckpts = [c for c in ckpts if gen_number(c) <= args.max_gen]
    ckpts = ckpts[:: args.every] if ckpts[-1] in ckpts[:: args.every] else ckpts[:: args.every] + [ckpts[-1]]
    gens = [gen_number(c) for c in ckpts]
    cfg = sim_config(args, args.spread)
    colors = gen_colors(len(ckpts), args.cmap)
    who = "러너" if args.side == "runner" else "수비"
    print(f"{len(ckpts)} generations: {gens[0]}..{gens[-1]} ({who})")
    plays_all = load_plays(ckpts, args.side, args.opponent, cfg, args.seed)

    W, H = SIZES[args.format]
    Ws, Hs = W * SS, H * SS
    f_title, f_sub, f_small = _font(SS * 46), _font(SS * 28), _font(SS * 24)
    labels = (f"{gens[0]}세대", f"{gens[-1]}세대")
    subtitle = args.subtitle or f"같은 시작, {who} {len(ckpts)}개 세대가 동시에"

    if args.grid:
        n = args.grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        img = Image.new("RGB", (Ws, Hs), BG)
        d = ImageDraw.Draw(img, "RGBA")
        head = SS * 170
        d.text((SS * 40, SS * 36), args.title, font=f_title, fill=TEXT)
        d.text((SS * 40, SS * 100), args.subtitle or f"시작 위치 {n}곳 · {who} {len(ckpts)}개 세대", font=f_sub, fill=SUBTEXT)
        legend_h = SS * 90
        cw, ch = (Ws - SS * 40) / cols, (Hs - head - legend_h - SS * 20) / rows
        for i in range(n):
            r, c = divmod(i, cols)
            box = Box.fit(cfg.field, SS * 20 + c * cw + SS * 8, head + r * ch + SS * 8, cw - SS * 16, ch - SS * 16,
                          vertical=True)
            draw_field(d, box)
            plays = [pa[i] for pa in plays_all]
            draw_ghosts(d, box, plays, colors, s=max(p.steps for p in plays), scale=0.55, focus=args.side)
            td = sum(p.outcome == "touchdown" for p in plays)
            d.text((box.x0 + SS * 6, box.y0 + SS * 4), f"TD {td}/{len(plays)}", font=f_small, fill=TEXT)
        draw_legend(d, SS * 60, Hs - legend_h, Ws - SS * 120, SS * 14, colors, labels, f_small)
        img.resize((W, H), Image.LANCZOS).save(args.out)
        print(f"-> {args.out}")
        return

    plays = [pa[args.play_index] for pa in plays_all]
    vertical = args.format != "episode"
    from dataclasses import replace
    view_field, y_off = cfg.field, 0.0
    if args.view is not None and args.view < cfg.field.length:
        view_field = replace(view_field, length=args.view)
    if args.view_width is not None:
        mid = float(np.mean([plays[0].r_pos[0, 1], plays[0].t_pos[0, 1]]))
        y_off = float(np.clip(mid - args.view_width / 2, 0, cfg.field.width - args.view_width))
        view_field = replace(view_field, width=args.view_width)
    head = SS * (250 if vertical else 150)
    foot = SS * (110 if vertical else 90)
    box = Box.fit(view_field, SS * 30, head, Ws - SS * 60, Hs - head - foot, vertical=vertical, top=True,
                  endzone=6.0 if view_field.length >= cfg.field.length else 0.0, y_off=y_off)
    clip_rect = tuple(int(v) for v in (box.x0, box.y0, box.x0 + box.w, box.y0 + box.h))
    legend_y = box.y0 + box.h + SS * 40
    base = Image.new("RGB", (Ws, Hs), BG)
    bd = ImageDraw.Draw(base, "RGBA")
    draw_field(bd, box)
    bd.text((SS * 40, SS * 40), args.title, font=f_title, fill=TEXT)
    bd.text((SS * 40, SS * 104), subtitle, font=f_sub, fill=SUBTEXT)
    draw_legend(bd, SS * 60, legend_y, Ws - SS * 120, SS * 14, colors, labels, f_small)
    if vertical:   # explain the marks in the space under the legend
        bd.text((SS * 60, legend_y + SS * 80), "● 공을 든 러너   ○ 수비   × 태클당한 지점", font=f_small, fill=SUBTEXT)

    sub = 30 / 10 / args.speed            # output frames per sim step
    longest = max(p.steps for p in plays)
    hold = int(2.5 * 30)

    def frames():
        for k in range(int(longest * sub) + 1 + hold):
            s = min(k / sub, longest)
            img = base.copy()
            # Draw trails on their own layer and keep only what falls inside the field box,
            # so a zoomed view doesn't spill trails over the header.
            layer = img.crop(clip_rect)
            ld = ImageDraw.Draw(layer, "RGBA")
            shifted = Box(box.x0 - clip_rect[0], box.y0 - clip_rect[1], box.w, box.h, box.vertical, box.field,
                          box.endzone, box.y_off)
            draw_ghosts(ld, shifted, plays, colors, s, focus=args.side)
            img.paste(layer, clip_rect[:2])
            d = ImageDraw.Draw(img, "RGBA")
            ended = [p for p in plays if s >= p.steps]
            td = sum(p.outcome == "touchdown" for p in ended)
            tk = sum(p.outcome == "tackled" for p in ended)
            stat = f"터치다운 {td}   태클 {tk}   진행 중 {len(plays) - len(ended)}"
            d.text((SS * 40, SS * (160 if vertical else 104) + (0 if vertical else SS * 0)) if vertical
                   else (Ws - SS * 40 - d.textlength(stat, font=f_sub), SS * 104), stat, font=f_sub, fill=TEXT)
            yield img.resize((W, H), Image.LANCZOS)

    last = None
    def tracked():
        nonlocal last
        for im in frames():
            last = im
            yield im

    n = write_mp4(tracked(), args.out, (W, H))
    print(f"{n} frames ({n / 30:.1f}s) {W}x{H} -> {args.out}")
    if args.poster and last is not None:
        png = args.out.with_suffix(".png")
        last.save(png)
        print(f"poster -> {png}")
    outs = [p.outcome for p in plays]
    print("outcomes by generation:", " ".join(f"{g}:{'T' if o == 'touchdown' else 'x' if o == 'tackled' else o[0]}"
                                               for g, o in zip(gens, outs)))


if __name__ == "__main__":
    main()

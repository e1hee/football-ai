"""Render pass plays (sim/passing.py) to MP4: top-down field, downfield is up.

Receiver orange, QB dark orange, CB blue (a stick shows the hips). The ball is
a brown dot: with the QB, in the air (with an X where it was thrown to), then
with the receiver. Panels side by side, each from the same start.

    uv run python -m render.pass_clip --run experiments/EXP-039-pass/s0 \
        --panel 1:1:"1세대" --panel 12:12:"12세대" --out clip.mp4
    (a panel is OFFENSE:DEFENSE:title; a generation number, or "script")
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.core import OUTCOME_NAMES
from sim.pass_policies import man_coverage, route_offense
from sim.passing import CB, QB, WR, END_ZONE, PassSim
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import config_from_run
from .clip import HOLD_END_S, OUT_FPS, SIM_HZ, write_mp4
from .frames import (BALL, BG, CAPTION_BG, CAPTION_FG, ENDZONE, GRASS, GRASS_ALT, LINE, RUNNER, RUNNER_DARK, TACKLER,
                     TACKLER_DARK, TAG_BG, _font)

QB_COLOR = (200, 110, 40)
LOS_COLOR = (90, 140, 230)
OUTCOME_KO = {"touchdown": "터치다운", "tackled": "태클", "out_of_bounds": "아웃", "timeout": "시간 초과",
              "incomplete": "패스 실패", "interception": "가로채기", "sack": "색"}


@dataclass
class PassPlay:
    pos: np.ndarray        # (T, 3, 2)
    facing: np.ndarray     # (T, 3, 2)
    ball: np.ndarray       # (T, 2)
    target: np.ndarray     # (T, 2)
    flying: np.ndarray     # (T,)
    outcome: str
    yards: float
    throw_t: float | None


def policy(spec: str, run: Path, side: str):
    if spec == "script":
        return (lambda sim, obs: route_offense(sim)) if side == "off" else (lambda sim, obs: man_coverage(sim))
    g = int(spec)
    ck = run / "checkpoints"
    return ppo_runner(ck / f"runner_g{g:02d}.zip") if side == "off" else ppo_tackler(ck / f"tackler_g{g:02d}.zip")


def record(cfg, off, dfn, n: int, seed: int) -> list[PassPlay]:
    """The first play of each of `n` slots (same seed = same starts across panels)."""
    sim = PassSim(n, cfg, seed=seed)
    obs = sim.observe()
    frames = [dict(pos=sim.pos.copy(), facing=sim.facing.copy(), ball=sim.ball_pos(), target=sim.target.copy(),
                   flying=np.zeros(n, bool))]
    fin = np.zeros(n, bool)
    end = np.zeros(n, np.int64)
    outcome = np.zeros(n, np.int8)
    yards = np.zeros(n, np.float32)
    throw_t = np.full(n, np.nan)
    for t in range(cfg.field.max_steps + 1):
        obs, _, done, info = sim.step(off(sim, obs), dfn(sim, obs))
        frames.append(dict(pos=info["pos"], facing=info["facing"], ball=info["ball"], target=info["target"],
                           flying=info["in_flight"]))
        live = ~fin
        tn = info["thrown_now"] & live
        throw_t[tn] = (t + 1) / SIM_HZ
        new = done & live
        end[new], outcome[new], yards[new] = t + 2, info["outcome"][new], info["yards"][new]
        fin |= done
        if fin.all():
            break
    stack = {k: np.stack([f[k] for f in frames], 1) for k in frames[0]}
    return [PassPlay(stack["pos"][i, :end[i]], stack["facing"][i, :end[i]], stack["ball"][i, :end[i]],
                     stack["target"][i, :end[i]], stack["flying"][i, :end[i]], OUTCOME_NAMES[outcome[i]],
                     float(yards[i]), None if np.isnan(throw_t[i]) else float(throw_t[i])) for i in range(n)]


class PassPanel:
    """One play on the whole field: from the back of the offense's side to the back of the end zone."""

    def __init__(self, cfg, width_px: int):
        f = cfg.field
        self.cfg = cfg
        self.x0, self.x1 = 0.0, f.length + END_ZONE
        self.s = width_px / f.width
        self.w = width_px
        self.h = int((self.x1 - self.x0) * self.s)
        self.base = self._field()

    def to_px(self, p) -> tuple[float, float]:
        return float(p[1] * self.s), float((self.x1 - p[0]) * self.s)

    def _field(self) -> Image.Image:
        img = Image.new("RGB", (self.w, self.h), GRASS)
        d = ImageDraw.Draw(img)
        f = self.cfg.field
        for k, x in enumerate(np.arange(self.x0, f.length, 5.0)):
            if k % 2:
                d.rectangle([0, self.to_px((x + 5, 0))[1], self.w, self.to_px((x, 0))[1]], fill=GRASS_ALT)
        d.rectangle([0, 0, self.w, self.to_px((f.length, 0))[1]], fill=ENDZONE)
        d.line([0, self.to_px((f.length, 0))[1], self.w, self.to_px((f.length, 0))[1]], fill=LINE, width=4)
        y = self.to_px((self.cfg.los, 0))[1]
        d.line([0, y, self.w, y], fill=LOS_COLOR, width=3)
        for x in np.arange(self.cfg.los + 10, f.length, 10.0):
            yy = self.to_px((x, 0))[1]
            d.line([0, yy, self.w, yy], fill=LINE, width=1)
        return img

    def draw(self, pos, facing, ball, target, flying: bool, has_ball_wr: bool, trail=None) -> Image.Image:
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        r = 0.9 * self.s                                   # drawn larger than the body so it reads on video
        if trail is not None and len(trail) > 1:
            for j, col in ((WR, (255, 200, 140)), (CB, (170, 210, 250))):
                pts = [self.to_px(q[j]) for q in trail]
                d.line(pts, fill=col, width=3)
        if flying:
            tx, ty = self.to_px(target)
            k = 0.6 * self.s
            d.line([tx - k, ty - k, tx + k, ty + k], fill=(250, 230, 120), width=3)
            d.line([tx - k, ty + k, tx + k, ty - k], fill=(250, 230, 120), width=3)
        for j, (col, dark) in ((QB, (QB_COLOR, RUNNER_DARK)), (WR, (RUNNER, RUNNER_DARK)), (CB, (TACKLER, TACKLER_DARK))):
            x, y = self.to_px(pos[j])
            d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=dark, width=2)
            fx, fy = facing[j][1] * r * 1.6, -facing[j][0] * r * 1.6
            d.line([x, y, x + fx, y + fy], fill=dark, width=3)
        bx, by = self.to_px(ball)
        br = 0.55 * self.s
        d.ellipse([bx - br, by - br * 0.7, bx + br, by + br * 0.7], fill=BALL, outline=(60, 30, 10))
        return img


def compose(panels: list[Image.Image], titles: list[str], captions: list[str], size=(1920, 1080)) -> Image.Image:
    W, H = size
    canvas = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(canvas)
    ft, fc = _font(40), _font(34)
    gap = 30
    pw = panels[0].width
    total = len(panels) * pw + (len(panels) - 1) * gap
    left = (W - total) // 2
    top = 70
    for i, (p, title, cap) in enumerate(zip(panels, titles, captions)):
        x = left + i * (pw + gap)
        canvas.paste(p, (x, top))
        tw = d.textlength(title, font=ft)
        d.text((x + (pw - tw) / 2, 18), title, fill=(240, 240, 240), font=ft)
        if cap:
            cw = d.textlength(cap, font=fc)
            d.rectangle([x, top + p.height + 14, x + pw, top + p.height + 70], fill=TAG_BG if "실패" in cap or "색" in cap else CAPTION_BG)
            d.text((x + (pw - cw) / 2, top + p.height + 22), cap, fill=CAPTION_FG, font=fc)
    return canvas


def frames_for(plays: list[PassPlay], renderers: list[PassPanel], titles: list[str]):
    steps = max(len(p.pos) for p in plays)
    sub = OUT_FPS // SIM_HZ
    for t in range(steps):
        for k in range(sub):
            a = k / sub
            imgs, caps = [], []
            for p, r in zip(plays, renderers):
                i = min(t, len(p.pos) - 1)
                j = min(t + 1, len(p.pos) - 1)
                lerp = lambda arr: arr[i] + (arr[j] - arr[i]) * a
                imgs.append(r.draw(lerp(p.pos), p.facing[i], lerp(p.ball), p.target[i], bool(p.flying[i]), False,
                                   trail=list(p.pos[:i + 1]) + [lerp(p.pos)]))
                done = t >= len(p.pos) - 1
                caps.append(f"{OUTCOME_KO.get(p.outcome, p.outcome)} {p.yards:+.0f}야드" if done else
                            f"{t / SIM_HZ:.1f}초" + (f" · 던짐 {p.throw_t:.1f}초" if p.throw_t and t / SIM_HZ >= p.throw_t else ""))
            yield compose(imgs, titles, caps)
    last = compose([r.draw(p.pos[-1], p.facing[-1], p.ball[-1], p.target[-1], False, False, trail=list(p.pos))
                    for p, r in zip(plays, renderers)],
                   titles, [f"{OUTCOME_KO.get(p.outcome, p.outcome)} {p.yards:+.0f}야드" for p in plays])
    for _ in range(int(HOLD_END_S * OUT_FPS)):
        yield last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--panel", action="append", required=True, help="OFFENSE:DEFENSE:title (generation or 'script')")
    ap.add_argument("--play-index", type=int, default=None)
    ap.add_argument("--pick", choices=["caught_last", "caught_all", "any"], default="caught_last",
                    help="caught_last / caught_all: first slot where the last / every panel's pass is completed")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = config_from_run(args.run)
    specs = [p.split(":", 2) for p in args.panel]
    recs = [record(cfg, policy(o, args.run, "off"), policy(d, args.run, "def"), args.n, args.seed) for o, d, _ in specs]
    idx = args.play_index
    if idx is None:
        caught = lambda p: p.outcome in ("touchdown", "tackled", "out_of_bounds")
        need = recs if args.pick == "caught_all" else recs[-1:]
        ok = [i for i in range(args.n) if all(caught(r[i]) for r in need)]
        idx = ok[0] if (ok and args.pick != "any") else 0
    plays = [r[idx] for r in recs]
    pw = min(880, (1920 - 30 * (len(plays) - 1)) // len(plays) - 20)
    pw = min(pw, int(840 * cfg.field.width / (cfg.field.length + END_ZONE)))   # fit the field's height
    renderers = [PassPanel(cfg, pw) for _ in plays]
    n = write_mp4(frames_for(plays, renderers, [s[2] for s in specs]), args.out, (1920, 1080))
    print(f"{args.out}: play {idx}, {n} frames | " + " | ".join(f"{s[2]}: {p.outcome} {p.yards:+.1f} yd throw {p.throw_t}"
                                                              for s, p in zip(specs, plays)))


if __name__ == "__main__":
    main()

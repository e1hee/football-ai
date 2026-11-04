"""Draw a play as chunky pixel-art frames.

Everything is drawn on a small canvas (a few px per yard) and upscaled with
nearest-neighbour so the result looks like the dot-style reels the project
is modelled on. Field is drawn vertically (goal line at the top) for 9:16
shorts and horizontally for 16:9 episodes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.core import FieldConfig
from .record import Play

# Palette: fictional team colours only.
GRASS = (62, 122, 70)
GRASS_ALT = (56, 112, 64)
LINE = (230, 230, 220)
ENDZONE = (48, 70, 110)
RUNNER = (240, 150, 60)       # "Comets" orange
RUNNER_DARK = (150, 80, 20)
TACKLER = (90, 160, 230)      # "Bolts" blue
TACKLER_DARK = (30, 80, 150)
BALL = (120, 70, 30)
BG = (24, 24, 32)
CAPTION_BG = (245, 240, 220)
CAPTION_FG = (30, 30, 30)
TAG_BG = (220, 70, 70)

_FONT_CANDIDATES = [
    os.path.expanduser("~/Library/Fonts/Galmuri11.ttf"),   # pixel font if the user installs it
    os.path.expanduser("~/.local/share/fonts/Galmuri11.ttf"),  # same font on Linux (cloud VM)
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",           # macOS Korean fallback
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


@dataclass
class Layout:
    vertical: bool
    px_per_yd: int = 3
    scale: int = 6                 # nearest-neighbour upscale factor
    margin: int = 6                # canvas px around the field
    caption_h: int = 26            # canvas px for the caption band
    trail: int = 12                # trail length in sim steps

    @classmethod
    def shorts(cls) -> "Layout":
        return cls(vertical=True)

    @classmethod
    def episode(cls) -> "Layout":
        return cls(vertical=False, px_per_yd=4, scale=5)


class PlayRenderer:
    def __init__(self, field: FieldConfig, layout: Layout):
        self.f = field
        self.L = layout
        L = layout
        self.field_w = int(round(field.width * L.px_per_yd))     # across the field
        self.field_l = int(round(field.length * L.px_per_yd))    # downfield
        self.endzone = 3 * L.px_per_yd
        if L.vertical:
            self.W = self.field_w + 2 * L.margin
            self.H = self.field_l + self.endzone + 2 * L.margin + L.caption_h
        else:
            self.W = self.field_l + self.endzone + 2 * L.margin
            self.H = self.field_w + 2 * L.margin + L.caption_h
        self.font = _font(10)
        self.font_small = _font(8)

    # yards -> canvas px. x is downfield, y across.
    def _xy(self, x: float, y: float) -> tuple[float, float]:
        L = self.L
        if L.vertical:
            # Downfield is up: goal line near the top, behind the caption band.
            px = L.margin + y * L.px_per_yd
            py = L.caption_h + L.margin + self.endzone + (self.f.length - x) * L.px_per_yd
        else:
            px = L.margin + x * L.px_per_yd
            py = L.caption_h + L.margin + y * L.px_per_yd
        return px, py

    def _field(self, d: ImageDraw.ImageDraw) -> None:
        f, L = self.f, self.L
        # Alternate 5-yard stripes, yard lines every 5 yards, endzone past the goal.
        for yd in range(0, int(f.length), 5):
            a = self._xy(yd, 0)
            b = self._xy(min(yd + 5, f.length), f.width)
            box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
            d.rectangle(box, fill=GRASS if (yd // 5) % 2 == 0 else GRASS_ALT)
        ez_a = self._xy(f.length, 0)
        ez_b = self._xy(f.length, f.width)
        if L.vertical:
            d.rectangle([ez_a[0], ez_a[1] - self.endzone, ez_b[0], ez_a[1]], fill=ENDZONE)
        else:
            d.rectangle([ez_a[0], ez_a[1], ez_a[0] + self.endzone, ez_b[1]], fill=ENDZONE)
        for yd in range(0, int(f.length) + 1, 5):
            a, b = self._xy(yd, 0), self._xy(yd, f.width)
            d.line([a, b], fill=LINE, width=1)
        # Hash marks every yard, NFL hashes are 18.5 ft apart (~6.2 yd) around centre.
        for yd in range(0, int(f.length)):
            for hy in (f.width / 2 - 3.1, f.width / 2 + 3.1):
                p = self._xy(yd, hy)
                d.point(p, fill=LINE)
        # Sidelines.
        for sy in (0.0, f.width):
            a, b = self._xy(0, sy), self._xy(f.length, sy)
            d.line([a, b], fill=LINE, width=1)

    def _player(self, d: ImageDraw.ImageDraw, pos, color, dark, has_ball: bool, face=None) -> None:
        x, y = self._xy(*pos)
        r = max(2, self.L.px_per_yd * (0.7 if self.L.px_per_yd < 8 else 0.5))
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=dark)
        if face is not None:
            # Hips: a short stick from the centre the way the body faces.
            fx, fy = self._xy(*(np.asarray(pos) + face * 1.6))
            d.line([(x, y), (fx, fy)], fill=dark, width=1)
        if has_ball:
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=BALL)

    def _trail(self, d: ImageDraw.ImageDraw, pts: np.ndarray, color) -> None:
        n = len(pts)
        for i, p in enumerate(pts):
            if i == n - 1:
                continue
            # Fade older points toward the grass colour.
            k = (i + 1) / n
            c = tuple(int(GRASS[j] * (1 - k) + color[j] * k) for j in range(3))
            d.point(self._xy(*p), fill=c)

    def _caption(self, d: ImageDraw.ImageDraw, title: str, tag: str | None) -> None:
        d.rectangle([0, 0, self.W, self.L.caption_h], fill=CAPTION_BG)
        d.text((4, 3), title, font=self.font, fill=CAPTION_FG)
        if tag:
            tw = d.textlength(tag, font=self.font_small)
            d.rectangle([4, 15, 8 + tw, 25], fill=TAG_BG)
            d.text((6, 15), tag, font=self.font_small, fill=(255, 255, 255))

    EVENT_TEXT = {"made": ("태클!", TAG_BG), "missed": ("놓쳤다!", (230, 180, 40))}
    EVENT_SHOW_STEPS = 7   # 0.7 s on screen

    def _events(self, d: ImageDraw.ImageDraw, play: Play, step: float, at) -> None:
        """Pop a small label above the runner for a moment after each tackle event."""
        for s, kind in play.events:
            if s <= step < s + self.EVENT_SHOW_STEPS:
                text, color = self.EVENT_TEXT[kind]
                x, y = self._xy(*at)
                y -= 4 * self.L.px_per_yd   # above the runner on screen
                w = d.textlength(text, font=self.font_small)
                d.rectangle([x - w / 2 - 2, y - 6, x + w / 2 + 2, y + 5], fill=color)
                d.text((x - w / 2, y - 6), text, font=self.font_small, fill=(255, 255, 255))

    def frame(self, play: Play, step: int, title: str, tag: str | None = None) -> Image.Image:
        """Draw the state at (fractional) `step`; positions are interpolated."""
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        self._field(d)

        i0 = int(np.floor(step))
        i1 = min(i0 + 1, play.steps)
        a = step - i0
        r = play.r_pos[i0] * (1 - a) + play.r_pos[i1] * a
        t = play.t_pos[i0] * (1 - a) + play.t_pos[i1] * a

        lo = max(0, i0 - self.L.trail)
        self._trail(d, play.r_pos[lo:i0 + 1], RUNNER)
        self._trail(d, play.t_pos[lo:i0 + 1], TACKLER)
        rf = tf = None
        if play.r_face is not None:
            rf, tf = play.r_face[i0], play.t_face[i0]
        self._player(d, t, TACKLER, TACKLER_DARK, has_ball=False, face=tf)
        self._player(d, r, RUNNER, RUNNER_DARK, has_ball=True, face=rf)
        self._events(d, play, step, r)
        self._caption(d, title, tag)

        s = self.L.scale
        return img.resize((self.W * s, self.H * s), Image.NEAREST)


def fit_canvas(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Letterbox onto an exact output size (1080x1920 shorts, 1920x1080 episode),
    shrinking with nearest-neighbour if needed so pixels stay crisp."""
    W, H = size
    k = min(W / img.width, H / img.height, 1.0)
    if k < 1.0:
        img = img.resize((int(img.width * k), int(img.height * k)), Image.NEAREST)
    out = Image.new("RGB", size, BG)
    out.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return out


def stack(frames: list[Image.Image], vertical: bool, gap: int = 12) -> Image.Image:
    """Put several rendered frames side by side (16:9) or one above another (9:16)."""
    if vertical:
        w = max(f.width for f in frames)
        h = sum(f.height for f in frames) + gap * (len(frames) - 1)
    else:
        w = sum(f.width for f in frames) + gap * (len(frames) - 1)
        h = max(f.height for f in frames)
    out = Image.new("RGB", (w, h), BG)
    off = 0
    for f in frames:
        out.paste(f, (0, off) if vertical else (off, 0))
        off += (f.height if vertical else f.width) + gap
    return out

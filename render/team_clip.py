"""Render team plays (3v3 and up) to MP4, same pixel style as `render.clip`.

Offense in orange (the ball carrier has the ball dot), defense in blue, a
short stick shows each player's hips, and a light line joins a blocker to
the defender he is engaged with.

    uv run python -m render.team_clip --pair screen::read --format shorts --out render/out/screen.mp4
    uv run python -m render.team_clip --pair ckpt:EXP/.../runner_g12.zip::tckpt:EXP/.../tackler_g12.zip \
        --pair screen::tckpt:... --event missed --out cmp.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.core import OUTCOME_NAMES, FieldConfig, SimConfig, TackleConfig
from sim.team import CARRIER, SIMS, TeamConfig, TeamSim, config_for
from sim.team_policies import DEFENSE, OFFENSE
from .clip import HOLD_END_S, OUT_FPS, OUT_SIZE, SIM_HZ, outcome_tag, play_frames, write_mp4
from .frames import (RUNNER, RUNNER_DARK, TACKLER, TACKLER_DARK, Layout, PlayRenderer, fit_canvas, stack)

ENGAGE_LINE = (250, 245, 200)
BLOCKER = (215, 120, 45)          # a shade off the carrier so the ball carrier stands out
LABELS = {"screen": "스크린", "carrier_only": "블로킹 없음", "pursuit": "추격 수비", "read": "읽고 반응 수비",
          "lane": "읽고 꺾는 RB", "gap": "갭 지키는 DL"}


@dataclass
class TeamPlay:
    pos: np.ndarray          # (T, P, 2)
    facing: np.ndarray       # (T, P, 2)
    partner: np.ndarray      # (T, P) engaged-with index or -1
    n_off: int
    outcome: str
    yards: float
    events: list[tuple[int, str]] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.pos) - 1

    @property
    def r_pos(self) -> np.ndarray:   # PlayRenderer._events places pop-ups above the carrier
        return self.pos[:, CARRIER]


def make_team_policy(name: str):
    """Scripted names from sim.team_policies, 'ckpt:path' (offense) or 'tckpt:path' (defense)."""
    if name.startswith("ckpt:"):
        from train.evaluate import ppo_runner
        return ppo_runner(name[5:])
    if name.startswith("tckpt:"):
        from train.evaluate import ppo_tackler
        return ppo_tackler(name[6:])
    fn = OFFENSE.get(name) or DEFENSE.get(name)
    if fn is None:
        raise ValueError(f"unknown team policy {name!r}")
    return lambda sim, obs=None: fn(sim)


def team_label(name: str) -> str:
    if name.startswith(("ckpt:", "tckpt:")):
        stem = Path(name.split(":", 1)[1]).stem
        num = int("".join(ch for ch in stem.split("_")[-1] if ch.isdigit()) or 0)
        return f"{'AI 수비' if name.startswith('tckpt:') else 'AI 공격'} {num}세대"
    return LABELS.get(name, name)


def record_team_plays(offense, defense, cfg: TeamConfig, n_plays: int = 32, seed: int = 0) -> list[TeamPlay]:
    sim = TeamSim(n_plays, cfg, seed=seed)
    obs = sim.reset()
    pos, face, part = [sim.pos.copy()], [sim.facing.copy()], [sim.partner.copy()]
    finished = np.zeros(n_plays, bool)
    end = np.full(n_plays, -1)
    outcome = np.zeros(n_plays, np.int8)
    yards = np.zeros(n_plays, np.float32)
    events: list[list[tuple[int, str]]] = [[] for _ in range(n_plays)]
    step = 0
    while not finished.all():
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        step += 1
        for kind, key in (("made", "tackle_made"), ("missed", "tackle_missed")):
            for i in np.nonzero(info[key] & ~finished)[0]:
                events[i].append((step, kind))
        pos.append(info["pos"])
        face.append(info["facing"])
        part.append(info["partner"])
        newly = done & ~finished
        end[newly] = step
        outcome[newly] = info["outcome"][newly]
        yards[newly] = info["yards"][newly]
        finished |= done
    P, F, E = np.stack(pos), np.stack(face), np.stack(part)
    return [TeamPlay(P[: end[i] + 1, i], F[: end[i] + 1, i], E[: end[i] + 1, i], sim.n_off,
                     OUTCOME_NAMES[outcome[i]], float(yards[i]), events[i]) for i in range(n_plays)]


BAG = (235, 200, 60)
LOS_LINE = (110, 160, 230)


class TeamRenderer(PlayRenderer):
    """`lane` = (left edge y, right edge y, line of scrimmage x) draws the drill's bags."""

    def __init__(self, field: FieldConfig, layout: Layout, lane: tuple[float, float, float] | None = None):
        super().__init__(field, layout)
        self.lane = lane

    def _lane(self, d: ImageDraw.ImageDraw) -> None:
        y0, y1, los = self.lane
        d.line([self._xy(los, y0), self._xy(los, y1)], fill=LOS_LINE, width=1)
        for y in (y0, y1):
            a, b = self._xy(los - 2.0, y), self._xy(los + 2.0, y)
            d.rectangle([min(a[0], b[0]) - 1, min(a[1], b[1]), max(a[0], b[0]) + 1, max(a[1], b[1])], fill=BAG)
            for x in range(0, int(self.f.length) + 1, 2):
                d.point(self._xy(x, y), fill=BAG)

    def frame(self, play: TeamPlay, step: float, title: str, tag: str | None = None) -> Image.Image:
        img = Image.new("RGB", (self.W, self.H), (24, 24, 32))
        d = ImageDraw.Draw(img)
        self._field(d)
        if self.lane:
            self._lane(d)
        i0 = int(np.floor(step))
        i1 = min(i0 + 1, play.steps)
        a = step - i0
        cur = play.pos[i0] * (1 - a) + play.pos[i1] * a
        lo = max(0, i0 - self.L.trail)
        n = cur.shape[0]
        colors = [(RUNNER if j == CARRIER else BLOCKER, RUNNER_DARK) if j < play.n_off else (TACKLER, TACKLER_DARK)
                  for j in range(n)]
        for j in range(n):
            self._trail(d, play.pos[lo:i0 + 1, j], colors[j][0])
        for j in range(play.n_off):            # blocker -> engaged defender
            k = play.partner[i0, j]
            if k >= 0:
                d.line([self._xy(*cur[j]), self._xy(*cur[k])], fill=ENGAGE_LINE, width=1)
        order = [j for j in range(n) if j != CARRIER] + [CARRIER]   # carrier on top
        for j in order:
            self._player(d, cur[j], colors[j][0], colors[j][1], has_ball=(j == CARRIER), face=play.facing[i0, j])
        self._events(d, play, step, cur[CARRIER])
        self._caption(d, title, tag)
        s = self.L.scale
        return img.resize((self.W * s, self.H * s), Image.NEAREST)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", action="append", required=True, help="offense::defense; repeat to compare side by side")
    p.add_argument("--format", choices=["shorts", "episode"], default="shorts")
    p.add_argument("--title", default="")
    p.add_argument("--outcome", default=None)
    p.add_argument("--event", default=None, choices=["made", "missed"])
    p.add_argument("--play-index", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--tackle-base", type=float, default=2.3)
    p.add_argument("--overspeed", type=float, default=0.25)
    p.add_argument("--sidestep", type=float, default=0.2)
    p.add_argument("--sim", choices=[s for s in SIMS if s != "1v1"], default="3v3")
    p.add_argument("--lane-width", type=float, default=5.0)
    p.add_argument("--stumble", type=float, default=0.0)
    p.add_argument("--square-ahead", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", action="store_true")
    args = p.parse_args()

    cfg = config_for(args.sim, SimConfig(field=FieldConfig(reaction_delay=args.delay), facing=True,
                                         tackle=TackleConfig(base_logit=args.tackle_base,
                                                             overspeed_weight=args.overspeed,
                                                             sidestep_weight=args.sidestep)),
                     lane_width=args.lane_width, stumble=args.stumble, carrier_square_ahead=args.square_ahead)
    layout = Layout.shorts() if args.format == "shorts" else Layout.episode()
    view, lane, shift = cfg.field, None, 0.0
    if cfg.formation.kind == "lane":
        # Zoom in, and show the lane inside a wider strip of practice field.
        layout = Layout(vertical=layout.vertical, px_per_yd=10, scale=4)
        view = dataclasses.replace(cfg.field, width=max(15.0, cfg.field.width + 6.0))
        shift = (view.width - cfg.field.width) / 2
        lane = (shift, shift + cfg.field.width, cfg.field.runner_start_x + cfg.formation.los_ahead)
    renderer = TeamRenderer(view, layout, lane)
    size = OUT_SIZE[args.format]
    pairs = [tuple(x.split("::", 1)) for x in args.pair]
    recorded = {pr: record_team_plays(make_team_policy(pr[0]), make_team_policy(pr[1]), cfg, seed=args.seed)
                for pr in pairs}
    for plays in recorded.values():
        for pl in plays:
            pl.pos[..., 1] += shift

    first = recorded[pairs[0]]
    if args.play_index is not None:
        idx = args.play_index
    else:
        pool = [i for i, pl in enumerate(first)
                if (args.outcome is None or pl.outcome == args.outcome)
                and (args.event is None or any(k == args.event for _, k in pl.events))]
        idx = pool[0] if pool else 0
    chosen = [(pr, recorded[pr][idx]) for pr in pairs]

    def title_of(pr):
        return args.title if (args.title and len(pairs) == 1) else f"{team_label(pr[0])} vs {team_label(pr[1])}"

    def frames():
        if len(chosen) == 1:
            pr, pl = chosen[0]
            for f in play_frames(renderer, pl, title_of(pr), None, outcome_tag(pl)):
                yield fit_canvas(f, size)
            return
        longest = max(pl.steps for _, pl in chosen)
        sub = OUT_FPS // SIM_HZ
        for k in range(longest * sub + 1 + int(HOLD_END_S * OUT_FPS)):
            parts = []
            for pr, pl in chosen:
                s = min(k / sub, pl.steps)
                parts.append(renderer.frame(pl, s, title_of(pr), outcome_tag(pl) if s >= pl.steps else None))
            yield fit_canvas(stack(parts, vertical=layout.vertical), size)

    if args.png:
        pr, pl = chosen[0]
        mid = pl.events[0][0] if pl.events else pl.steps // 2
        png = args.out.with_suffix(".png")
        fit_canvas(renderer.frame(pl, mid, title_of(pr)), size).save(png)
        print(f"preview -> {png}")
    n = write_mp4(frames(), args.out, size)
    print(f"{n} frames ({n / OUT_FPS:.1f}s) -> {args.out}  play #{idx}")
    for pr, plays in recorded.items():
        outs = {o: sum(pl.outcome == o for pl in plays) for o in sorted({pl.outcome for pl in plays})}
        print(f"  {team_label(pr[0])} vs {team_label(pr[1])}: {outs}")


if __name__ == "__main__":
    main()

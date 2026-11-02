"""Render recorded plays to MP4 (and a preview PNG) through ffmpeg.

    uv run python -m render.clip --runner straight --tackler pursuit \
        --title "EXP-000 · 직진 러너" --tag "80% 득점" --format shorts --out render/out/exp000.mp4

    uv run python -m render.clip --compare random,straight --format episode --out render/out/cmp.mp4

Learned policies plug in later via --runner ckpt:path (see train/).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from sim.core import SimConfig
from sim.policies import pursuit_tackler, random_policy, straight_runner
from .frames import Layout, PlayRenderer, stack
from .record import Play, Policy, pick, record_plays

SIM_HZ = 10
OUT_FPS = 30               # interpolate 10 Hz sim to 30 fps
HOLD_END_S = 1.2           # freeze on the final frame so the outcome registers


def make_policy(name: str, seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    if name == "random":
        return lambda sim, obs: random_policy(sim, rng)
    if name == "straight":
        return lambda sim, obs: straight_runner(sim)
    if name == "pursuit":
        return lambda sim, obs: pursuit_tackler(sim)
    if name == "still":
        return lambda sim, obs: np.zeros((sim.n, 2), np.float32)
    raise ValueError(f"unknown policy {name!r}")


def play_frames(renderer: PlayRenderer, play: Play, title: str, tag: str | None):
    """Yield one upscaled frame per output tick, including the end hold."""
    sub = OUT_FPS // SIM_HZ
    for k in range(play.steps * sub + 1):
        yield renderer.frame(play, k / sub, title, tag)
    last = renderer.frame(play, play.steps, title, tag)
    for _ in range(int(HOLD_END_S * OUT_FPS)):
        yield last


def write_mp4(frames, out: Path, size: tuple[int, int]) -> int:
    """Pipe raw RGB frames into ffmpeg. Returns the frame count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    w -= w % 2
    h -= h % 2  # yuv420p needs even dimensions
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(OUT_FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
        str(out),
    ]
    n = 0
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for img in frames:
            if img.size != (w, h):
                img = img.crop((0, 0, w, h))
            proc.stdin.write(img.tobytes())
            n += 1
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runner", default="straight", help="random|straight (ckpt:... later)")
    p.add_argument("--tackler", default="pursuit", help="pursuit|still")
    p.add_argument("--compare", default=None, help="comma list of runner policies to show together")
    p.add_argument("--format", choices=["shorts", "episode"], default="shorts")
    p.add_argument("--title", default="")
    p.add_argument("--tag", default=None)
    p.add_argument("--outcome", default=None, help="prefer a play with this outcome")
    p.add_argument("--plays", type=int, default=1, help="how many plays to chain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", action="store_true", help="also save a mid-play preview PNG")
    args = p.parse_args()

    cfg = SimConfig()
    layout = Layout.shorts() if args.format == "shorts" else Layout.episode()
    renderer = PlayRenderer(cfg.field, layout)
    tackler = make_policy(args.tackler, args.seed)

    runners = args.compare.split(",") if args.compare else [args.runner]
    # Same seed for every runner so compared plays start from identical setups.
    recorded = {
        name: record_plays(make_policy(name, args.seed), tackler, n_plays=16, cfg=cfg, seed=args.seed)
        for name in runners
    }

    def frames():
        for i in range(args.plays):
            chosen = []
            for name in runners:
                plays = recorded[name]
                play = pick(plays, args.outcome) if args.plays == 1 else plays[i % len(plays)]
                chosen.append((name, play))
            if len(chosen) == 1:
                name, play = chosen[0]
                title = args.title or f"{name} runner"
                tag = args.tag or f"{play.outcome} · {play.yards:.0f} yd"
                yield from play_frames(renderer, play, title, tag)
            else:
                # Compare: advance all plays in lockstep, holding finished ones.
                longest = max(pl.steps for _, pl in chosen)
                sub = OUT_FPS // SIM_HZ
                total = longest * sub + 1 + int(HOLD_END_S * OUT_FPS)
                for k in range(total):
                    parts = []
                    for name, pl in chosen:
                        s = min(k / sub, pl.steps)
                        tag = f"{pl.outcome} · {pl.yards:.0f} yd" if s >= pl.steps else None
                        parts.append(renderer.frame(pl, s, f"{name} runner", tag))
                    yield stack(parts, vertical=layout.vertical)

    gen = frames()
    first = next(gen)
    size = first.size

    def all_frames():
        yield first
        yield from gen

    if args.png:
        png = args.out.with_suffix(".png")
        # Grab a frame roughly a third of the way through the first play for the preview.
        preview_play = recorded[runners[0]][0]
        renderer.frame(preview_play, preview_play.steps // 3, args.title or runners[0], args.tag).save(png)
        print(f"preview -> {png}")

    n = write_mp4(all_frames(), args.out, size)
    print(f"{n} frames @ {OUT_FPS} fps ({n / OUT_FPS:.1f}s) {size[0]}x{size[1]} -> {args.out}")
    for name, plays in recorded.items():
        outcomes = {o: sum(pl.outcome == o for pl in plays) for o in set(pl.outcome for pl in plays)}
        print(f"  {name:>9}: {outcomes}")


if __name__ == "__main__":
    main()

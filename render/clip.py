"""Render recorded plays to MP4 (and a preview PNG) through ffmpeg.

    # one matchup, 9:16 shorts
    uv run python -m render.clip --runner evasive --tackler angle --spread 8 \
        --title "EXP-001" --format shorts --out render/out/a.mp4

    # side-by-side matchups (runner/tackler pairs), same seed = same starting setup
    uv run python -m render.clip --compare straight/pursuit,straight/angle --spread 8 \
        --format episode --out render/out/cmp.mp4

Learned policies plug in later via --runner ckpt:path (see train/).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from sim.core import FieldConfig, SimConfig, TackleConfig


def add_sim_args(p: argparse.ArgumentParser) -> None:
    """Sim settings a checkpoint was trained with (EXP-005+: delay and tackle control)."""
    p.add_argument("--delay", type=int, default=0, help="reaction delay in steps")
    p.add_argument("--tackle-base", type=float, default=1.2)
    p.add_argument("--overspeed", type=float, default=0.0)
    p.add_argument("--sidestep", type=float, default=0.0)
    p.add_argument("--facing", action="store_true", help="body facing (hips), EXP-008+")


def sim_config(args, spread: float | None) -> SimConfig:
    return SimConfig(field=FieldConfig(tackler_lateral_spread=spread, reaction_delay=args.delay),
                     tackle=TackleConfig(base_logit=args.tackle_base, overspeed_weight=args.overspeed,
                                         sidestep_weight=args.sidestep), facing=args.facing)
from sim.policies import (angle_tackler, evasive_runner, pursuit_tackler, random_policy,
                          read_react_tackler, straight_runner)
from .frames import Layout, PlayRenderer, fit_canvas, stack
from .record import Play, Policy, pick, record_plays

SIM_HZ = 10
OUT_FPS = 30               # interpolate 10 Hz sim to 30 fps
HOLD_END_S = 1.2           # freeze on the final frame so the outcome registers
OUT_SIZE = {"shorts": (1080, 1920), "episode": (1920, 1080)}
LABEL = {"random": "무작위", "straight": "직진", "evasive": "회피",
         "pursuit": "추격", "angle": "각도+브레이크다운", "read": "읽고 반응", "still": "정지"}


def make_policy(name: str, seed: int = 0) -> Policy:
    """Scripted policy by name, a saved PPO runner via 'ckpt:path.zip', or a
    saved PPO tackler via 'tckpt:path.zip' (it sees the tackler's observation)."""
    if name.startswith("ckpt:"):
        from train.evaluate import ppo_runner
        return ppo_runner(name[5:])
    if name.startswith("tckpt:"):
        from train.evaluate import ppo_tackler
        return ppo_tackler(name[6:])
    rng = np.random.default_rng(seed)
    table = {
        "random": lambda sim, obs: random_policy(sim, rng),
        "straight": lambda sim, obs: straight_runner(sim),
        "evasive": lambda sim, obs: evasive_runner(sim),
        "pursuit": lambda sim, obs: pursuit_tackler(sim),
        "angle": lambda sim, obs: angle_tackler(sim),
        "read": lambda sim, obs: read_react_tackler(sim),
        "still": lambda sim, obs: np.zeros((sim.n, 2), np.float32),
    }
    if name not in table:
        raise ValueError(f"unknown policy {name!r}")
    return table[name]


def play_frames(renderer: PlayRenderer, play: Play, title: str, tag: str | None, end_tag: str | None = None):
    """Yield one upscaled frame per output tick, then hold the last frame.
    `end_tag` (the outcome) only appears on the hold, so the clip doesn't spoil itself."""
    sub = OUT_FPS // SIM_HZ
    for k in range(play.steps * sub):
        yield renderer.frame(play, k / sub, title, tag)
    last = renderer.frame(play, play.steps, title, end_tag or tag)
    for _ in range(int(HOLD_END_S * OUT_FPS)):
        yield last


def write_mp4(frames, out: Path, size: tuple[int, int]) -> int:
    """Pipe raw RGB frames (already `size`) into ffmpeg. Returns the frame count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
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
            proc.stdin.write(img.tobytes())
            n += 1
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    return n


def label(name: str) -> str:
    if name.startswith(("ckpt:", "tckpt:")):
        stem = Path(name.split(":", 1)[1]).stem       # gen_010, runner_g03, tackler_g07
        num = int("".join(ch for ch in stem.split("_")[-1] if ch.isdigit()) or 0)
        who = "AI 수비" if name.startswith("tckpt:") or stem.startswith("tackler") else "AI 러너"
        return f"{who} {num}세대"
    return LABEL.get(name, name)


def outcome_tag(pl: Play) -> str:
    word = {"touchdown": "터치다운", "tackled": "태클", "out_of_bounds": "아웃", "timeout": "시간 초과"}
    return f"{word.get(pl.outcome, pl.outcome)} · {pl.yards:.0f}야드"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runner", default="straight", help="random|straight|evasive (ckpt:... later)")
    p.add_argument("--tackler", default="angle", help="pursuit|angle|still")
    p.add_argument("--compare", default=None,
                   help="comma list of pairs shown together: straight/pursuit,straight/angle or ckpt:a.zip::tckpt:b.zip")
    p.add_argument("--sequence", default=None,
                   help="comma list of runners played one after another on the SAME start "
                        "(generation evolution), e.g. ckpt:.../gen_000.zip,ckpt:.../gen_025.zip")
    p.add_argument("--play-index", type=int, default=None, help="force which recorded start (0-31) to use")
    p.add_argument("--spread", type=float, default=None, help="tackler lateral spread in yd (default: whole field)")
    p.add_argument("--format", choices=["shorts", "episode"], default="shorts")
    p.add_argument("--title", default="")
    p.add_argument("--tag", default=None)
    p.add_argument("--outcome", default=None, help="prefer a play with this outcome")
    p.add_argument("--event", default=None, choices=["made", "missed"], help="prefer a play with this tackle event")
    p.add_argument("--plays", type=int, default=1, help="how many plays to chain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", action="store_true", help="also save a preview PNG")
    add_sim_args(p)
    args = p.parse_args()

    cfg = sim_config(args, args.spread)
    layout = Layout.shorts() if args.format == "shorts" else Layout.episode()
    renderer = PlayRenderer(cfg.field, layout)
    size = OUT_SIZE[args.format]

    if args.sequence:
        pairs = [(r, args.tackler) for r in args.sequence.split(",")]
    elif args.compare:
        # "runner::tackler" (any names/paths), or the short "runner/tackler" for scripted names.
        pairs = [tuple(x.split("::", 1)) if "::" in x else tuple(x.rsplit("/", 1)) for x in args.compare.split(",")]
    else:
        pairs = [(args.runner, args.tackler)]
    # Same seed for every pair so compared plays start from identical setups.
    recorded = {
        pr: record_plays(make_policy(pr[0], args.seed), make_policy(pr[1], args.seed),
                         n_plays=32, cfg=cfg, seed=args.seed)
        for pr in pairs
    }

    def choose(plays: list[Play], i: int) -> Play:
        if args.plays > 1:
            return plays[i % len(plays)]
        pool = [pl for pl in plays if args.event is None or any(k == args.event for _, k in pl.events)]
        return pick(pool or plays, args.outcome)

    def frames():
        if args.sequence:
            idx = args.play_index if args.play_index is not None else 0
            for pr in pairs:
                pl = recorded[pr][idx]
                for f in play_frames(renderer, pl, args.title or label(pr[0]), label(pr[0]), outcome_tag(pl)):
                    yield fit_canvas(f, size)
            return
        for i in range(args.plays):
            if len(pairs) == 1:
                pl = choose(recorded[pairs[0]], i)
                title = args.title or f"러너: {label(pairs[0][0])} vs 태클러: {label(pairs[0][1])}"
                for f in play_frames(renderer, pl, title, args.tag, outcome_tag(pl)):
                    yield fit_canvas(f, size)
                continue
            # Compare: use the same play index (= same start) for every pair.
            idx = 0
            if args.play_index is not None:
                idx = args.play_index + i
            elif args.plays == 1 and (args.outcome or args.event):
                first = recorded[pairs[0]]
                idx = first.index(choose(first, 0))
            else:
                idx = i
            chosen = [(pr, recorded[pr][idx % 32]) for pr in pairs]
            longest = max(pl.steps for _, pl in chosen)
            sub = OUT_FPS // SIM_HZ
            for k in range(longest * sub + 1 + int(HOLD_END_S * OUT_FPS)):
                parts = []
                for pr, pl in chosen:
                    s = min(k / sub, pl.steps)
                    tag = outcome_tag(pl) if s >= pl.steps else None
                    parts.append(renderer.frame(pl, s, f"{label(pr[0])} vs {label(pr[1])}", tag))
                yield fit_canvas(stack(parts, vertical=layout.vertical), size)

    if args.png:
        pl = choose(recorded[pairs[0]], 0)
        mid = pl.events[0][0] if pl.events else pl.steps // 2
        png = args.out.with_suffix(".png")
        fit_canvas(renderer.frame(pl, mid, args.title or "", args.tag), size).save(png)
        print(f"preview -> {png}")

    n = write_mp4(frames(), args.out, size)
    print(f"{n} frames @ {OUT_FPS} fps ({n / OUT_FPS:.1f}s) {size[0]}x{size[1]} -> {args.out}")
    for pr, plays in recorded.items():
        outcomes = {o: sum(pl.outcome == o for pl in plays) for o in sorted(set(pl.outcome for pl in plays))}
        ev = sum(len(pl.events) for pl in plays)
        print(f"  {label(pr[0]):>10}/{label(pr[1]):<8} {outcomes}  tackle events: {ev}")


if __name__ == "__main__":
    main()

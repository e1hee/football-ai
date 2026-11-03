"""Does the line hold? Over the first seconds of a box play, per offense-vs-defense pair:

- blocks alive: share of the DL still blocked at each time (of plays still live)
- how blocks ended before the runner got to the line: shed, OL let go, tackle
- where the line is: mean OL and DL x past the line of scrimmage at each time
  (negative = in the offense's backfield)
- DL who got 1+ yd into the backfield, and the widest gap between neighbouring OL

    uv run python -m scripts.line_hold experiments/EXP-017-box --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.core import FieldConfig, TackleConfig
from sim.team import CARRIER, TeamSim, box_config
from sim.team_policies import box_offense, gap_defense
from train.evaluate import ppo_runner, ppo_tackler

TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def line_stats(offense, defense, cfg, n: int = 1024, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    roles = np.array(sim.roles)
    dls = sim.dfn[roles[sim.dfn] == "DL"]
    live = np.ones(n, bool)
    steps = int(max(TIMES) * 10) + 1
    alive, dl_x, ol_x, live_n, gap_max = (np.zeros(steps) for _ in range(5))
    ended = {"shed": 0, "ol_let_go": 0, "tackle": 0}
    t_block_end = []
    block_age = np.zeros((n, len(dls)))
    in_backfield = np.zeros(n, bool)
    for t in range(steps):
        o, d = offense(sim, obs), defense(sim, obs)
        before = sim.partner[:, dls] >= 0
        crossed_before = sim.pos[:, CARRIER, 0] >= sim.los
        los = sim.los.copy()
        obs, _, done, info = sim.step(o, d)
        pos, part = info["pos"], info["partner"]
        after = part[:, dls] >= 0
        block_age = np.where(before, block_age + 0.1, block_age)
        gone = before & ~after & live[:, None]
        t_block_end.extend(block_age[gone].tolist())
        # A DL shed this step has a fresh re-block cooldown (plays that ended were reset: skip those).
        cd = cfg.block.reengage_cooldown - cfg.field.dt
        shed = gone & ~done[:, None] & (sim.shed_cd[:, dls - sim.n_off] >= cd - 1e-4)
        tackle = gone & info["tackle_made"][:, None]
        ended["shed"] += int(shed.sum())
        ended["tackle"] += int(tackle.sum())
        ended["ol_let_go"] += int((gone & ~done[:, None] & ~shed & ~tackle).sum())
        ok = live & ~done
        if ok.any():
            alive[t] = after[ok].mean()
            dl_x[t] = (pos[ok][:, dls, 0] - los[ok, None]).mean()
            ol_x[t] = (pos[ok][:, sim.blk, 0] - los[ok, None]).mean()
            ol_y = np.sort(pos[ok][:, sim.blk, 1], axis=1)
            gap_max[t] = np.diff(ol_y, axis=1).max(1).mean()
            live_n[t] = ok.mean()
            in_backfield[ok] |= (pos[ok][:, dls, 0] - los[ok, None] < -1.0).any(1)
        live &= ~done
    at = lambda v: {f"{s:.1f}": float(v[int(s * 10) - 1]) for s in TIMES}
    total = max(sum(ended.values()), 1)
    return {"alive": at(alive), "dl_x": at(dl_x), "ol_x": at(ol_x), "ol_gap": at(gap_max), "live": at(live_n),
            "ended": {k: v / total for k, v in ended.items()},
            "block_median_s": float(np.median(t_block_end)) if t_block_end else float("nan"),
            "dl_backfield_1yd": float(in_backfield.mean())}


def show(name: str, s: dict) -> None:
    fmt = lambda d, pct=False: " ".join(f"{v:6.0%}" if pct else f"{v:6.2f}" for v in d.values())
    print(name)
    print(f"  시간            {' '.join(f'{k:>6s}' for k in s['alive'])}")
    print(f"  살아있는 플레이 {fmt(s['live'], True)}")
    print(f"  블록 유지 DL    {fmt(s['alive'], True)}")
    print(f"  DL x(라인 기준) {fmt(s['dl_x'])}")
    print(f"  OL x(라인 기준) {fmt(s['ol_x'])}")
    print(f"  OL 최대 간격    {fmt(s['ol_gap'])}")
    e = s["ended"]
    print(f"  블록 끝난 이유: 벗어남 {e['shed']:.0%} / OL이 놓음 {e['ol_let_go']:.0%} / 태클 {e['tackle']:.0%}"
          f" · 중앙값 {s['block_median_s']:.1f}s · DL 백필드 1야드+ {s['dl_backfield_1yd']:.0%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path, nargs="?", default=None)
    p.add_argument("--gen", type=int, default=None)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--snap", action="store_true")
    p.add_argument("--mix", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = box_config(snap=args.snap, mix=args.mix, stumble=0.2, carrier_square_ahead=True, timeout_ends=True, field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    results = {}
    r = line_stats(lambda s, o=None: box_offense(s), lambda s, o=None: gap_defense(s), cfg, n=args.n)
    results["scripted"] = r
    show("스크립트 공격 / 스크립트 갭 수비", r)
    if args.exp:
        runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir())
        for r in runs:   # these checks run with hips on: a run trained without them is a different game
            if "facing: false" in (r / "config.yaml").read_text():
                raise SystemExit(f"{r} was trained without --facing")

        def ckpt(run, kind):
            files = sorted((run / "checkpoints").glob(f"{kind}_g*.zip"))
            if args.gen is not None:
                files = [f for f in files if f.stem.endswith(f"g{args.gen:02d}")] or files
            return files[-1]

        for a in runs:
            for b in runs:
                r = line_stats(ppo_runner(ckpt(a, "runner")), ppo_tackler(ckpt(b, "tackler")), cfg, n=args.n)
                results[f"{a.name} vs {b.name}"] = r
                show(f"AI 공격 {a.name} / AI 수비 {b.name}", r)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

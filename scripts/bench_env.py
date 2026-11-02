"""Throughput benchmark + scripted baselines. Target: >= 10,000 env steps/s.

    uv run python -m scripts.bench_env --n 4096 --steps 200 --runner straight --spread 8
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sim.core import OUTCOME_NAMES, FieldConfig, OneVOneSim, SimConfig
from sim.policies import angle_tackler, evasive_runner, pursuit_tackler, random_policy, straight_runner


def run(n: int, steps: int, runner: str, spread: float | None, tackler: str = "angle", seed: int = 0) -> dict:
    sim = OneVOneSim(n, SimConfig(field=FieldConfig(tackler_lateral_spread=spread)), seed=seed)
    rng = np.random.default_rng(seed + 1)
    sim.reset()
    pick = {"random": lambda: random_policy(sim, rng), "straight": lambda: straight_runner(sim),
            "evasive": lambda: evasive_runner(sim)}[runner]

    tpick = {"pursuit": lambda: pursuit_tackler(sim), "angle": lambda: angle_tackler(sim)}[tackler]
    outcomes = np.zeros(len(OUTCOME_NAMES), np.int64)
    yards: list[float] = []
    attempts = made = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, done, info = sim.step(pick(), tpick())
        attempts += int(info["tackle_attempt"].sum())
        made += int(info["tackle_made"].sum())
        if done.any():
            np.add.at(outcomes, info["outcome"][done], 1)
            yards.extend(info["yards"][done].tolist())
    dt = time.perf_counter() - t0
    fin = int(outcomes.sum())
    return {
        "runner": runner, "tackler": tackler, "spread": spread, "steps_per_s": round(n * steps / dt),
        "plays": fin, "outcomes": {k: round(c / fin, 3) for k, c in zip(OUTCOME_NAMES, outcomes) if c},
        "mean_yards": round(float(np.mean(yards)), 2),
        "tackle_attempts_per_play": round(attempts / fin, 2),
        "tackle_success": round(made / max(attempts, 1), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--runner", choices=["random", "straight", "evasive"], default="straight")
    p.add_argument("--tackler", choices=["pursuit", "angle"], default="angle")
    p.add_argument("--spread", type=float, default=None, help="tackler lateral spread (None = whole field)")
    p.add_argument("--all", action="store_true", help="run every runner x {wide, 8 yd} and print JSON")
    args = p.parse_args()
    if args.all:
        rows = [run(args.n, args.steps, r, s, t) for t in ("pursuit", "angle") for s in (None, 8.0)
                for r in ("random", "straight", "evasive")]
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return
    r = run(args.n, args.steps, args.runner, args.spread, args.tackler)
    print(f"{r['steps_per_s']:,} steps/s | {r['plays']:,} plays | {r['outcomes']} | "
          f"{r['mean_yards']} yd/play | {r['tackle_attempts_per_play']} attempts/play, "
          f"{r['tackle_success']:.0%} made")


if __name__ == "__main__":
    main()

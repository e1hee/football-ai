"""Throughput benchmark for the vectorized sim. Target: >= 10,000 env steps/s.

    uv run python -m scripts.bench_env --n 4096 --steps 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sim.core import OUTCOME_NAMES, OneVOneSim
from sim.policies import pursuit_tackler, random_policy, straight_runner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096, help="parallel plays")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--runner", choices=["random", "straight"], default="random")
    args = p.parse_args()

    sim = OneVOneSim(args.n, seed=0)
    rng = np.random.default_rng(1)
    sim.reset()

    outcomes = np.zeros(len(OUTCOME_NAMES), np.int64)
    yards_at_end: list[float] = []
    t0 = time.perf_counter()
    for _ in range(args.steps):
        ra = random_policy(sim, rng) if args.runner == "random" else straight_runner(sim)
        _, _, done, info = sim.step(ra, pursuit_tackler(sim))
        if done.any():
            np.add.at(outcomes, info["outcome"][done], 1)
            yards_at_end.extend(info["yards"][done].tolist())
    dt = time.perf_counter() - t0

    total = args.n * args.steps
    print(f"{total:,} env steps in {dt:.2f}s -> {total / dt:,.0f} steps/s (n={args.n})")
    finished = outcomes.sum()
    print(f"plays finished: {finished:,}")
    for name, c in zip(OUTCOME_NAMES, outcomes):
        if c:
            print(f"  {name:>14}: {c / finished:6.1%}")
    if yards_at_end:
        print(f"  mean yards/play: {np.mean(yards_at_end):.2f}")


if __name__ == "__main__":
    main()

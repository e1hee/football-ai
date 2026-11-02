"""Is a play decided once the runner gets past the tackler?

For every play, record the first moment the runner leads the tackler
(downfield, x) by at least L yards, then see how the play ends. If almost no
play with a lead >= L ends in a tackle, we can stop the sim there and count
it as a breakaway touchdown, which skips the dead running steps.

Also reports what fraction of all sim steps happen after that moment (the
compute the early end would save).

    uv run python -m scripts.breakaway_check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TACKLED, OUTCOME_TOUCHDOWN, FieldConfig, OneVOneSim, SimConfig
from sim.physics import PROFILES
from sim.policies import angle_tackler, evasive_runner, pursuit_tackler, straight_runner
from train.evaluate import ppo_runner, ppo_tackler

LEADS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


def check(runner, tackler, cfg: SimConfig, n: int = 4096, seed: int = 99) -> dict:
    sim = OneVOneSim(n, cfg, seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    first_lead_step = {L: np.full(n, -1) for L in LEADS}
    outcome = np.zeros(n, np.int8)
    steps = np.zeros(n, np.int32)
    t = 0
    while live.any():
        lead = sim.r_pos[:, 0] - sim.t_pos[:, 0]
        for L in LEADS:
            hit = live & (lead >= L) & (first_lead_step[L] < 0)
            first_lead_step[L][hit] = t
        obs, _, done, info = sim.step(runner(sim, obs), tackler(sim))
        t += 1
        newly = done & live
        outcome[newly] = info["outcome"][newly]
        steps[newly] = info["steps"][newly]
        live &= ~done
    res = {}
    total_steps = steps.sum()
    for L in LEADS:
        had = first_lead_step[L] >= 0
        caught = had & (outcome == OUTCOME_TACKLED)
        saved = np.where(had, steps - first_lead_step[L], 0).sum()
        res[L] = {
            "plays_with_lead": float(had.mean()),
            "caught_after_lead": float(caught.sum() / max(had.sum(), 1)),
            "td_after_lead": float((had & (outcome == OUTCOME_TOUCHDOWN)).sum() / max(had.sum(), 1)),
            "steps_saved": float(saved / total_steps),
        }
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp2", type=Path, default=Path("experiments/EXP-002-ppo-runner-v1/checkpoints/gen_025.zip"))
    p.add_argument("--exp3", type=Path, default=Path("experiments/EXP-003-selfplay-v1/checkpoints"))
    args = p.parse_args()
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=8.0))
    fast = SimConfig(field=FieldConfig(tackler_lateral_spread=8.0), tackler=PROFILES["WR"].scaled(max_speed=1.05, accel=1.05))
    S = lambda f: (lambda sim, obs: f(sim))
    matchups = [
        ("직진 vs 추격", S(straight_runner), pursuit_tackler, cfg),
        ("회피 vs 각도", S(evasive_runner), angle_tackler, cfg),
        ("AI러너(EXP-002) vs 각도", ppo_runner(args.exp2), angle_tackler, cfg),
        ("회피 vs 각도 (수비가 5% 더 빠름)", S(evasive_runner), angle_tackler, fast),
    ]
    ts = sorted(args.exp3.glob("tackler_g*.zip"))
    if len(ts) > 1:
        t_last = ppo_tackler(ts[-1])
        matchups.append((f"AI러너(EXP-002) vs AI수비 {ts[-1].stem}", ppo_runner(args.exp2), lambda sim: t_last(sim), cfg))
        matchups.append((f"회피 vs AI수비 {ts[-1].stem}", S(evasive_runner), lambda sim: t_last(sim), cfg))
    for name, r, t, c in matchups:
        res = check(r, t, c)
        print(f"\n{name}")
        print("  앞선 거리 | 그런 플레이 | 그 뒤 잡힘 | 그 뒤 TD | 아낄 수 있는 스텝")
        for L, v in res.items():
            print(f"  {L:5.1f}yd | {v['plays_with_lead']:10.1%} | {v['caught_after_lead']:9.2%} | "
                  f"{v['td_after_lead']:7.1%} | {v['steps_saved']:14.1%}")


if __name__ == "__main__":
    main()

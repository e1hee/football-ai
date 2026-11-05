"""Do the LBs chase a runner who goes around, and does the reward want them to?

wide: the runner is still behind the line and 3.6+ yd off the middle (first moment, per play).
For 1 s after that, per LB (near / far = closer / farther from the runner at that moment):
- chase: moving 2+ yd/s within about 45 deg of where the runner will be in 0.5 s.
- charged: chase steps that pay the LB rush cost (`TeamSim.lb_rushing`).
- engaged: steps held by a blocker.
tackles: share of made tackles by an LB, on wide plays.

What if: the same offense, with both LBs replaced by the scripted pursuit (always, or once the
runner goes wide). Yards and the defense's reward per play show whether the reward would pay
for the chase. A one-off swap: the offense doesn't get to adapt.

    PYTHONPATH=. uv run python -m scripts.lb_chase experiments/EXP-025-roles --gen 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TACKLED
from sim.team import CARRIER, TeamSim
from sim.team_policies import pursuit_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config

BOX = 3.6
LBS = [4, 5]


def is_wide(sim) -> np.ndarray:
    rb = sim.pos[:, CARRIER]
    return (rb[:, 0] < sim.los) & (np.abs(rb[:, 1] - sim.cfg.field.width / 2) > BOX)


def lb_pursuit(dfn, only_when_wide: bool):
    state = {}

    def f(sim, obs):
        a = np.array(dfn(sim, obs)).reshape(sim.n, sim.n_def, 3)
        s = pursuit_defense(sim).reshape(sim.n, sim.n_def, 3)
        if only_when_wide:
            state["w"] = state.get("w", np.zeros(sim.n, bool)) | is_wide(sim)
            use = state["w"]
        else:
            use = np.ones(sim.n, bool)
        for k in LBS:
            a[use, k] = s[use, k]
        return a.reshape(sim.n, -1)
    return f


def play(off, dfn, cfg, n: int, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    lb = np.asarray(sim.dfn)[LBS]
    fin = np.zeros(n, bool)
    yards, dfn_reward = np.zeros(n), np.zeros(n)
    t0 = np.full(n, -1)
    near = np.zeros(n, int)
    d0 = np.full(n, np.nan)
    d1 = np.full(n, np.nan)
    chase, charged, engaged, steps = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    by_lb = np.zeros(n, bool)
    by_dl = np.zeros(n, bool)
    tackled = np.zeros(n, bool)
    t = 0
    rows = np.arange(n)
    while not fin.all():
        obs, _, done, info = sim.step(off(sim, obs), dfn(sim, obs))
        t += 1
        live = ~fin & ~done
        dfn_reward[~fin] += info["tackler_reward"][~fin]
        made = info["tackle_made"] & ~fin & ~by_lb & ~by_dl
        if made.any():
            hit = info["tackle_by_def"] & ~info["tackle_missed_by_def"]
            by_lb |= made & hit[:, LBS].any(axis=1)
            by_dl |= made & hit[:, :4].any(axis=1)
        new_wide = live & (t0 < 0) & is_wide(sim)
        rb, v_rb = sim.pos[:, CARRIER], sim.vel[:, CARRIER]
        if new_wide.any():
            dist = np.linalg.norm(sim.pos[:, lb] - rb[:, None], axis=-1)
            t0[new_wide] = t
            near[new_wide] = dist.argmin(axis=1)[new_wide]
            d0[new_wide] = dist.min(axis=1)[new_wide]
        window = live & (t0 >= 0) & (t - t0 <= 10)
        if window.any():
            goal = rb + 0.5 * v_rb
            rel = goal[:, None] - sim.pos[:, lb]
            v = sim.vel[:, lb]
            speed = np.linalg.norm(v, axis=-1)
            cos = (v * rel).sum(-1) / np.maximum(speed * np.linalg.norm(rel, axis=-1), 1e-6)
            is_chase = (speed >= 2.0) & (cos >= 0.7)
            pay = sim.lb_rushing()
            held = sim.partner[:, lb] >= 0
            for which in (0, 1):            # 0 near, 1 far
                k = near if which == 0 else 1 - near
                r = rows[window]
                steps[which] += len(r)
                chase[which] += is_chase[r, k[r]].sum()
                charged[which] += (is_chase & pay)[r, k[r]].sum()
                engaged[which] += held[r, k[r]].sum()
        at1 = live & (t0 >= 0) & (t - t0 == 10)
        if at1.any():
            dist = np.linalg.norm(sim.pos[:, lb] - rb[:, None], axis=-1)
            d1[at1] = dist[at1, near[at1]]
        new = done & ~fin
        yards[new] = info["yards"][new]
        tackled[new] = info["outcome"][new] == OUTCOME_TACKLED
        fin |= done
    wide = t0 >= 0
    made_w = wide & (by_lb | by_dl)
    return {"yards": float(yards.mean()), "dfn_reward": float(dfn_reward.mean()), "wide": float(wide.mean()),
            "yards_wide": float(yards[wide].mean()) if wide.any() else float("nan"),
            "tackled_wide": float(tackled[wide].mean()) if wide.any() else float("nan"),
            "lb_tackle_share_wide": float(by_lb[made_w].mean()) if made_w.any() else float("nan"),
            "near_chase": float(chase[0] / max(steps[0], 1)), "far_chase": float(chase[1] / max(steps[1], 1)),
            "near_charged": float(charged[0] / max(chase[0], 1)), "far_charged": float(charged[1] / max(chase[1], 1)),
            "near_engaged": float(engaged[0] / max(steps[0], 1)), "far_engaged": float(engaged[1] / max(steps[1], 1)),
            "near_dist_0": float(np.nanmedian(d0)), "near_dist_1s": float(np.nanmedian(d1))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    out = {}
    for s in range(3):
        ck = args.exp / f"s{s}" / "checkpoints"
        o, d = ppo_runner(ck / f"runner_g{args.gen:02d}.zip"), ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip")
        base = play(o, d, cfg, args.n)
        wide = play(o, lb_pursuit(d, True), cfg, args.n)
        always = play(o, lb_pursuit(d, False), cfg, args.n)
        out[f"s{s}"] = {"base": base, "pursue_when_wide": wide, "pursue_always": always}
        print(f"s{s} wide {base['wide']*100:.0f}% yd_wide {base['yards_wide']:.2f} tackled_wide {base['tackled_wide']*100:.0f}% "
              f"LB share of tackles {base['lb_tackle_share_wide']*100:.0f}% | chase near {base['near_chase']*100:.0f}% far {base['far_chase']*100:.0f}% "
              f"charged near {base['near_charged']*100:.0f}% far {base['far_charged']*100:.0f}% engaged near {base['near_engaged']*100:.0f}% far {base['far_engaged']*100:.0f}% "
              f"dist {base['near_dist_0']:.1f}->{base['near_dist_1s']:.1f} | yards {base['yards']:.2f} when-wide {wide['yards']:.2f} always {always['yards']:.2f} "
              f"| dfn reward {base['dfn_reward']:.3f} when-wide {wide['dfn_reward']:.3f} always {always['dfn_reward']:.3f}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

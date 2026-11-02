"""Who does what in a team sim: per-role tackles, blocks, and who closed the gap.

For an offense/defense checkpoint pair, over many plays:
  - tackle attempts and made tackles by defender role,
  - share of play time each defender spends engaged with a blocker,
  - at the moment a block starts, who closed the distance (blocker vs defender
    speed toward the other): did the defender take on the blocker on purpose?
  - per blocker: plays where it ends 5+ yd behind its start without having
    blocked much ("lost blocker").

    uv run python -m scripts.team_roles experiments/EXP-009-3v3-screen --gen 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.core import FieldConfig, TackleConfig
from sim.team import CARRIER, TeamConfig, TeamSim
from train.evaluate import ppo_runner, ppo_tackler


def roles_report(offense, defense, cfg: TeamConfig, n: int = 2048, seed: int = 21) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    D = sim.n_def
    live = np.ones(n, bool)
    att = np.zeros(D)
    made = np.zeros(D)
    eng_time = np.zeros(D)
    steps = 0
    closers = []                      # (defender idx, defender closing speed, blocker closing speed)
    start = sim.pos[:, sim.blk].copy()
    blk_eng = np.zeros((n, len(sim.blk)))
    last_x = np.zeros((n, len(sim.blk)))
    prev_partner = sim.partner.copy()
    while live.any():
        pre_pos, pre_vel = sim.pos.copy(), sim.vel.copy()
        obs, _, done, info = sim.step(offense(sim, obs), defense(sim, obs))
        part = info["partner"]
        a = info["attempts"] > 0
        # attempts per defender: recompute from stun/miss is messy; use made + misses by nearest attempt.
        dist = np.linalg.norm(info["pos"][:, sim.dfn] - info["pos"][:, None, CARRIER], axis=-1)
        near = np.argmin(dist, axis=1)
        for i in np.nonzero(a & live)[0]:
            att[near[i]] += 1
        for i in np.nonzero(info["tackle_made"] & live)[0]:
            made[sim.wrapper[i] - sim.n_off] += 1
        eng_time += ((part[:, sim.dfn] >= 0) & live[:, None]).sum(0)
        steps += live.sum()
        for j, b in enumerate(sim.blk):
            newly = (part[:, b] >= 0) & (prev_partner[:, b] < 0) & live
            for i in np.nonzero(newly)[0]:
                d = part[i, b]
                u = pre_pos[i, d] - pre_pos[i, b]
                u /= max(np.linalg.norm(u), 1e-6)
                closers.append((d - sim.n_off, float(-pre_vel[i, d] @ u), float(pre_vel[i, b] @ u)))
            blk_eng[:, j] += (part[:, b] >= 0) & live
            last_x[live, j] = info["pos"][live, b, 0]
        prev_partner = part
        live &= ~done
    c = np.array(closers) if closers else np.zeros((0, 3))
    roles = sim.roles[sim.n_off:]
    out = {"defense": {}}
    for k in range(D):
        sel = c[c[:, 0] == k] if len(c) else c
        out["defense"][f"{roles[k]}{k}"] = {
            "tackle_attempt_share": float(att[k] / max(att.sum(), 1)),
            "made_tackle_share": float(made[k] / max(made.sum(), 1)),
            "engaged_time": float(eng_time[k] / steps),
            "blocks_taken": int(len(sel)),
            "def_closing_speed_at_block": float(sel[:, 1].mean()) if len(sel) else float("nan"),
            "blocker_closing_speed_at_block": float(sel[:, 2].mean()) if len(sel) else float("nan"),
        }
    lost = (last_x - start[..., 0] < -5) & (blk_eng < 5)
    out["lost_blocker_share"] = float(lost.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--gen", type=int, default=12)
    args = ap.parse_args()
    cfg = TeamConfig(field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    ck = args.exp / "checkpoints"
    res = roles_report(ppo_runner(ck / f"runner_g{args.gen:02d}.zip"), ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip"), cfg)
    for role, r in res["defense"].items():
        print(f"{role:4s} attempts {r['tackle_attempt_share']:5.0%}  made {r['made_tackle_share']:5.0%}  "
              f"engaged {r['engaged_time']:5.0%}  blocks {r['blocks_taken']:5d}  "
              f"closing speed at block: defender {r['def_closing_speed_at_block']:4.1f} / blocker {r['blocker_closing_speed_at_block']:4.1f} yd/s")
    print(f"lost blocker share {res['lost_blocker_share']:.0%}")


if __name__ == "__main__":
    main()

"""Compare two self-play settings over several seeds, league style (EXP-012).

Every final offense (runner_g12) plays every final defense (tackler_g12) of
both settings and all seeds, so a setting's offense is judged against the
same opponents as the other's, not just its own co-evolved rival. Per
offense: mean TD rate vs all defenses, plus blocking behaviour (lost blocker,
carrier behind a block) vs all defenses; per defense: mean TD allowed.

    uv run python -m scripts.seed_compare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.selfplay_report import team_style
from scripts.team_roles import roles_report
from sim.core import FieldConfig, TackleConfig
from sim.team import TeamConfig
from train.evaluate import evaluate, ppo_runner, ppo_tackler

ROOT = Path(__file__).resolve().parents[1] / "experiments"
RUNS = {
    "base": ["EXP-009-3v3-screen", "EXP-012-seed-repeats/base-s1", "EXP-012-seed-repeats/base-s2"],
    "shaping": ["EXP-011-blocker-shaping-fixed", "EXP-012-seed-repeats/shaping-s1", "EXP-012-seed-repeats/shaping-s2"],
}


def main() -> None:
    cfg_b = TeamConfig(field=FieldConfig(reaction_delay=2, breakaway_lead=2.0),
                       tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    cfg = TeamConfig(field=FieldConfig(reaction_delay=2),
                     tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))
    offs = {(s, r): ppo_runner(ROOT / r / "checkpoints/runner_g12.zip") for s, rs in RUNS.items() for r in rs}
    defs = {(s, r): ppo_tackler(ROOT / r / "checkpoints/tackler_g12.zip") for s, rs in RUNS.items() for r in rs}
    td = np.zeros((len(offs), len(defs)))
    for i, o in enumerate(offs.values()):
        for j, d in enumerate(defs.values()):
            td[i, j] = evaluate(o, d, cfg_b, n_plays=2048)["rate/touchdown"]
    beh = {}
    for k, o in offs.items():
        lost, behind = [], []
        for d in defs.values():
            lost.append(roles_report(o, d, cfg, n=512)["lost_blocker_share"])
            behind.append(team_style(o, d, cfg, n=512)["carrier_behind_block_share"])
        beh[k] = (float(np.mean(lost)), float(np.mean(behind)))
    keys_o, keys_d = list(offs), list(defs)
    out = {"offense": {}, "defense": {}}
    print("offense (mean TD vs all 6 defenses | lost blocker | behind block)")
    for i, k in enumerate(keys_o):
        out["offense"][f"{k[0]}/{k[1]}"] = {"td_vs_all": float(td[i].mean()), "lost_blocker": beh[k][0], "behind_block": beh[k][1]}
        print(f"  {k[0]:8s} {k[1]:40s} {td[i].mean():6.1%} | {beh[k][0]:5.1%} | {beh[k][1]:5.1%}")
    print("defense (mean TD allowed vs all 6 offenses)")
    for j, k in enumerate(keys_d):
        out["defense"][f"{k[0]}/{k[1]}"] = {"td_allowed": float(td[:, j].mean())}
        print(f"  {k[0]:8s} {k[1]:40s} {td[:, j].mean():6.1%}")
    for s in RUNS:
        oi = [i for i, k in enumerate(keys_o) if k[0] == s]
        dj = [j for j, k in enumerate(keys_d) if k[0] == s]
        o_scores = td[oi].mean(1)
        d_scores = td[:, dj].mean(0)
        lost = [beh[keys_o[i]][0] for i in oi]
        behind = [beh[keys_o[i]][1] for i in oi]
        print(f"{s:8s} offense TD {o_scores.mean():.1%} (seeds {', '.join(f'{x:.0%}' for x in o_scores)}) | "
              f"defense allows {d_scores.mean():.1%} (seeds {', '.join(f'{x:.0%}' for x in d_scores)}) | "
              f"lost blocker {np.mean(lost):.1%} ({', '.join(f'{x:.0%}' for x in lost)}) | "
              f"behind block {np.mean(behind):.1%} ({', '.join(f'{x:.0%}' for x in behind)})")
    out["td_matrix"] = td.tolist()
    (ROOT / "EXP-012-seed-repeats/league.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""A new offense (trained 2M steps vs a frozen defense) walks through that defense up the middle.
Is it the new runner or the new line? Swap the runner and the OL between the trained offense and
the new one, against the same frozen defense.

    PYTHONPATH=. uv run python -m scripts.offense_mix experiments/EXP-025-roles experiments/EXP-025-roles/results/g12 --gen 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.lane_check import dumb_runner, middle
from scripts.role_report import report
from sim.team import CARRIER
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config


def mixed(runner_from, line_from):
    def f(sim, obs):
        a = np.array(line_from(sim, obs)).reshape(sim.n, sim.n_off, 3)
        a[:, CARRIER] = np.array(runner_from(sim, obs)).reshape(sim.n, sim.n_off, 3)[:, CARRIER]
        return a.reshape(sim.n, -1)
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("fresh", type=Path, help="folder holding checkpoints/fresh_r_sK.zip")
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    keys = ["yards", "td", "reach_line", "box", "path_ratio", "backfield_contact", "ol_free_run", "reblock_1s"]
    print(f"{'':24}" + "".join(f"{k[:10]:>11}" for k in keys))
    out = {}
    for s in range(3):
        ck = args.exp / f"s{s}" / "checkpoints"
        old = ppo_runner(ck / f"runner_g{args.gen:02d}.zip")
        new = ppo_runner(args.fresh / "checkpoints" / f"fresh_r_s{s}.zip")
        dfn = ppo_tackler(ck / f"tackler_g{args.gen:02d}.zip")
        arms = {"old RB + old OL": old, "new RB + new OL": new, "new RB + old OL": mixed(new, old),
                "old RB + new OL": mixed(old, new), "middle RB + new OL": dumb_runner(new, middle)}
        for name, off in arms.items():
            row = report(off, dfn, cfg, args.n)
            out[f"s{s} {name}"] = row
            print(f"{f's{s} {name}':24}" + "".join(f"{row[k]:11.2f}" for k in keys), flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""What does an AI runner do when the defense doesn't come at him?

Fresh defenses beat EXP-024's runners by hanging back: the runner stepped out of
bounds behind the line with nobody near. This checks the same thing with no
learning involved: all six defenders standing still (zero action).

    uv run python -m scripts.passive_defense experiments/EXP-024-explore/B --gen 24 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_NAMES
from sim.team import TeamSim
from train.evaluate import ppo_runner
from train.run_config import exp_config


def against_still_defense(offense, cfg, n: int = 1024, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    still = np.zeros((n, sim.n_def * 3), np.float32)
    fin, yards, outcome = np.zeros(n, bool), np.zeros(n), np.zeros(n, int)
    while not fin.all():
        obs, _, done, info = sim.step(offense(sim, obs), still)
        new = done & ~fin
        yards[new], outcome[new] = info["yards"][new], info["outcome"][new]
        fin |= done
    return {"yards": float(yards.mean()), **{OUTCOME_NAMES[k]: float((outcome == k).mean()) for k in np.unique(outcome)}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=24)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.exp, None)
    out = {}
    for r in sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()):
        out[r.name] = against_still_defense(ppo_runner(r / "checkpoints" / f"runner_g{args.gen:02d}.zip"), cfg)
        print(r.name, {k: round(v, 3) for k, v in out[r.name].items()}, flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

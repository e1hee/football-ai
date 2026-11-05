"""Is self-play learning anything general? Checks that don't need new training:

- mirror: the sim is left-right symmetric, so a runner flipped left-right is an
  equally legal runner. A defense that holds him one way and not the other only
  covers one side. `both` flips both teams and must match `normal`.
- line: mean yards / TD for the AI runner behind its own line, a line that holds
  still, random pushes, or the scripted line (same defense).
- fresh: optional policies from `scripts/recovery_test.py --save` vs every generation-36 policy.

    uv run python -m scripts.selfplay_audit experiments/EXP-022-long --gen 36 \
        --fresh-tackler t.zip --fresh-runner r.zip --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from sim.core import OUTCOME_OUT_OF_BOUNDS, OUTCOME_TIMEOUT, OUTCOME_TOUCHDOWN
from sim.team import TeamSim
from sim.team_policies import box_blockers, box_offense, gap_defense
from train.run_config import config_from_run

# Left-right swap of the box-mix slots: RB, then OL left to right; DL left to right, LBs.
OFF = [0, 5, 4, 3, 2, 1]
DEF = [3, 2, 1, 0, 5, 4]


def mirror_obs(o: np.ndarray, own: list[int], other: list[int], n_global: int = 5) -> np.ndarray:
    o = o.copy()
    o[:, 1] = 1 - o[:, 1]
    per = o[:, n_global:].reshape(len(o), len(own) + len(other), 8)
    per[..., [1, 3, 5]] *= -1
    o[:, n_global:] = per[:, own + [i + len(own) for i in other]].reshape(len(o), -1)
    return o


def actor(model, side: str, mirror: bool = False):
    own, other = (OFF, DEF) if side == "off" else (DEF, OFF)

    def act(sim):
        ob = sim.observe() if side == "off" else sim.observe_tackler()
        if not mirror:
            return model.predict(ob, deterministic=True)[0]
        a = model.predict(mirror_obs(ob, own, other), deterministic=True)[0].reshape(sim.n, 6, 3).copy()
        a[..., 1] *= -1
        return a[:, own].reshape(sim.n, -1)
    return act


def play(offense, defense, cfg, n: int = 2048, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    sim.reset()
    live = np.ones(n, bool)
    out, yards, end_y, tries = np.zeros(n, np.int8), np.zeros(n), np.zeros(n), np.zeros(n)
    while live.any():
        _, _, done, info = sim.step(offense(sim), defense(sim))
        tries += info["attempts"] * live
        new = done & live
        out[new], yards[new], end_y[new] = info["outcome"][new], info["yards"][new], info["r_pos"][new, 1]
        live &= ~done
    return {"td": float((out == OUTCOME_TOUCHDOWN).mean()), "yards": float(yards.mean()),
            "timeout": float((out == OUTCOME_TIMEOUT).mean()), "oob": float((out == OUTCOME_OUT_OF_BOUNDS).mean()),
            "end_right": float((end_y > cfg.field.width / 2).mean()), "any_try": float((tries > 0).mean())}


def line_variant(model, mode: str):
    base = actor(model, "off")

    def act(sim):
        a = np.array(base(sim)).reshape(sim.n, 6, 3)
        if mode == "still":
            a[:, 1:] = 0
            a[:, 1:, 2] = 1
        elif mode == "noisy":
            a[:, 1:, :2] = np.random.default_rng(int(sim.t.sum())).normal(size=a[:, 1:, :2].shape)
        elif mode == "script":
            a[:, sim.blk] = box_blockers(sim, sim.perceived("offense")[0], 2.5)
        return a.reshape(sim.n, -1)
    return act


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--gen", type=int, default=36)
    p.add_argument("--fresh-tackler", type=Path, default=None)
    p.add_argument("--fresh-runner", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir())
    cfg = config_from_run(runs[0])

    def load(run, kind, gen=args.gen):
        return PPO.load(str(run / "checkpoints" / f"{kind}_g{gen:02d}.zip"), device="cpu")

    res: dict = {}
    for r in runs:
        off, dfn = load(r, "runner"), load(r, "tackler")
        res[r.name] = {"normal": play(actor(off, "off"), actor(dfn, "def"), cfg),
                       "runner_mirrored": play(actor(off, "off", True), actor(dfn, "def"), cfg),
                       "both_mirrored": play(actor(off, "off", True), actor(dfn, "def", True), cfg)}
        res[r.name]["line"] = {m: play(line_variant(off, m), actor(dfn, "def"), cfg) for m in ("still", "noisy", "script")}
        print(r.name, json.dumps(res[r.name], indent=None, default=float), flush=True)
    script_off, script_def = (lambda s: box_offense(s)), (lambda s: gap_defense(s))
    if args.fresh_tackler:
        t = actor(PPO.load(str(args.fresh_tackler), device="cpu"), "def")
        res["fresh_tackler"] = {"scripted": play(script_off, t, cfg),
                                **{r.name: play(actor(load(r, "runner"), "off"), t, cfg) for r in runs}}
        print("fresh_tackler", res["fresh_tackler"], flush=True)
    if args.fresh_runner:
        o = actor(PPO.load(str(args.fresh_runner), device="cpu"), "off")
        res["fresh_runner"] = {"scripted": play(o, script_def, cfg),
                               **{r.name: play(o, actor(load(r, "tackler"), "def"), cfg) for r in runs}}
        print("fresh_runner", res["fresh_runner"], flush=True)
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

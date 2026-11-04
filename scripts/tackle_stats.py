"""Tackle odds in the box sim, next to the NFL.

Counts every tackle attempt (a defender in arm's reach, once per 0.9 s
cooldown) and splits them by whether the defender was free or still engaged
with a blocker. Each attempt's success probability is recorded, so the report
can also say what the same attempts would give if the base logit moved by
some amount, and which shift reaches a target success rate for free defenders
(the "what if" assumes players don't change how they play; retrain to know).

NFL: PFF-charted missed tackle rate about 13% (league average, see
research/2026-11-02). A charter counts a defender who had a real shot and
whiffed; we count every reach, including arm tackles out of a block, so only
the free-defender rate is compared.

    uv run python -m scripts.tackle_stats                       # scripted players, EXP-019 rules
    uv run python -m scripts.tackle_stats experiments/EXP-020-tackle --gen 12
    uv run python -m scripts.tackle_stats --tackle-base 3.2 --engaged-tackle -2.4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from sim.core import OUTCOME_TACKLED
from sim.team import TeamSim
from sim.team_policies import box_offense, gap_defense, pursuit_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import build_config, config_from_run, run_args

EXP019 = Path("experiments/EXP-019-mix/s0")


def logit(p):
    return np.log(p / (1.0 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def tackle_stats(offense, defense, cfg, n: int = 2048, seed: int = 7) -> dict:
    sim = TeamSim(n, cfg, seed=seed)
    obs = sim.reset()
    live = np.ones(n, bool)
    any_attempt = np.zeros(n, bool)
    outcome = np.zeros(n, np.int8)
    yards = np.zeros(n)
    p_free, p_eng, ok_free, ok_eng = [], [], [], []
    while live.any():
        o, d = offense(sim, obs), defense(sim, obs)
        obs, _, done, info = sim.step(o, d)
        att = info["tackle_by_def"] & live[:, None]
        if att.any():
            eng = info["tackle_engaged"]
            ok = att & ~info["tackle_missed_by_def"]
            p = info["tackle_p"]
            p_free.append(p[att & ~eng]); ok_free.append(ok[att & ~eng])
            p_eng.append(p[att & eng]); ok_eng.append(ok[att & eng])
            any_attempt |= att.any(1)
        newly = done & live
        outcome[newly] = info["outcome"][newly]
        yards[newly] = info["yards"][newly]
        live &= ~done
    # A runner can go a whole sample untouched, so the lists may be empty.
    pf, pe = (np.concatenate(x) if x else np.empty(0) for x in (p_free, p_eng))
    of, oe = (np.concatenate(x) if x else np.empty(0, bool) for x in (ok_free, ok_eng))
    n_att = len(pf) + len(pe)
    return {
        "plays": n, "attempts": n_att,
        "success": float((of.sum() + oe.sum()) / max(n_att, 1)),
        "success_free": float(of.mean()) if len(of) else float("nan"),
        "success_engaged": float(oe.mean()) if len(oe) else float("nan"),
        "engaged_share": float(len(pe) / max(n_att, 1)),
        "no_attempt": float((~any_attempt).mean()),
        "tackled": float((outcome == OUTCOME_TACKLED).mean()),
        "yards_mean": float(yards.mean()),
        "stuffed": float((yards <= 0).mean()),
        "_p_free": pf, "_p_eng": pe,
    }


def shift_for(p: np.ndarray, target: float) -> float:
    """Logit shift that makes the mean odds of these attempts equal target (bisection)."""
    lo, hi = -6.0, 6.0
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sig(lp + mid).mean() < target else (lo, mid)
    return (lo + hi) / 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path, nargs="?", default=None, help="folder with s0/s1/... runs (else scripted only)")
    p.add_argument("--gen", type=int, default=None)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--tackle-base", type=float, default=None, help="scripted only: override the run's base logit")
    p.add_argument("--engaged-tackle", type=float, default=None, help="scripted only: override the engaged offset")
    p.add_argument("--target", type=float, default=0.87, help="free-defender success to solve for")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    runs = sorted(d for d in args.exp.iterdir() if (d / "checkpoints").is_dir()) if args.exp else []
    if runs:
        cfg = config_from_run(runs[0])
        for r in runs[1:]:
            if config_from_run(r) != cfg:
                raise SystemExit(f"{r} has different rules from {runs[0]}")
    else:
        a = run_args(args.exp / "s0" if args.exp else EXP019)
        if args.tackle_base is not None:
            a.tackle_base = args.tackle_base
        if args.engaged_tackle is not None:
            a.engaged_tackle = args.engaged_tackle
        cfg = build_config(a)
    print(f"rules: base {cfg.tackle.base_logit}, engaged {cfg.block.engaged_tackle_logit}, "
          f"overspeed {cfg.tackle.overspeed_weight}, sidestep {cfg.tackle.sidestep_weight}")

    def ckpt(run, kind):
        files = sorted((run / "checkpoints").glob(f"{kind}_g*.zip"))
        if args.gen is not None:
            files = [f for f in files if f.stem.endswith(f"g{args.gen:02d}")] or files
        return files[-1]

    off = {"script read": lambda s, o=None: box_offense(s),
           "script snap-decide": lambda s, o=None: box_offense(s, read_until=99.0)}
    dfn = {"script gap": lambda s, o=None: gap_defense(s),
           "script pursuit": lambda s, o=None: pursuit_defense(s)}
    for r in runs:
        off[f"AI off {r.name}"] = ppo_runner(ckpt(r, "runner"))
        dfn[f"AI def {r.name}"] = ppo_tackler(ckpt(r, "tackler"))

    cols = ["success", "success_free", "success_engaged", "engaged_share", "no_attempt", "stuffed"]
    print(f"{'offense / defense':34s} " + " ".join(f"{c[:12]:>12s}" for c in cols) + f" {'yards':>6s} {'attempts':>8s}")
    results, pf_all = {}, []
    for on, of in off.items():
        for dn, df in dfn.items():
            s = tackle_stats(of, df, cfg, n=args.n)
            pf_all.append(s.pop("_p_free")); s.pop("_p_eng")
            results[f"{on} | {dn}"] = s
            print(f"{on + ' | ' + dn:34s} " + " ".join(f"{s[c]:12.0%}" for c in cols)
                  + f" {s['yards_mean']:6.2f} {s['attempts']:8d}")
    pf = np.concatenate(pf_all)
    shift = shift_for(pf, args.target)
    print(f"\nfree-defender success, all matchups: {np.mean(sig(logit(np.clip(pf, 1e-6, 1 - 1e-6)))):.0%} expected "
          f"-> {args.target:.0%} needs base logit {shift:+.2f} "
          f"({cfg.tackle.base_logit} -> {cfg.tackle.base_logit + shift:.2f}), same attempts, players unchanged")
    if args.json:
        args.json.write_text(json.dumps({"rules": {"tackle": dataclasses.asdict(cfg.tackle),
                                                   "engaged_tackle_logit": cfg.block.engaged_tackle_logit},
                                         "matchups": results, "free_shift_to_target": shift,
                                         "target": args.target}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

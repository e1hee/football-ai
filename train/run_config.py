"""One place that turns training args into a sim config.

`train.selfplay` builds its config here, and reports rebuild a run's exact
config from its `config.yaml` with `config_from_run`, instead of hard-coding
rule values that go stale when a rule changes (EXP-020 changed the tackle odds).
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
from pathlib import Path

import yaml

from sim.core import FieldConfig, RewardConfig, SimConfig, TackleConfig
from sim.team import config_for

# Args added after older runs were trained, with the value those runs used.
LATER_ARGS = {"engaged_tackle": None, "shed_rate": None, "reengage_cooldown": None,
              "ol_run_cost": 0.0, "lb_rush_cost": 0.0, "lane_bonus": 0.0}


def build_config(args):
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=args.spread, breakaway_lead=args.breakaway_lead,
                                      full_play_share=args.full_play_share, reaction_delay=args.delay),
                    tackle=TackleConfig(base_logit=args.tackle_base, overspeed_weight=args.overspeed,
                                        sidestep_weight=args.sidestep),
                    reward=RewardConfig(yards_scale=args.yards_scale, blocker_shaping=args.blocker_shaping,
                                        ol_run_cost=getattr(args, "ol_run_cost", 0.0),
                                        lb_rush_cost=getattr(args, "lb_rush_cost", 0.0),
                                        lane_bonus=getattr(args, "lane_bonus", 0.0)),
                    facing=args.facing)
    team = (dict(stumble=args.stumble, carrier_square_ahead=args.square_ahead, timeout_ends=args.timeout_ends)
            if args.sim != "1v1" else {})
    cfg = config_for(args.sim, cfg, lane_width=args.lane_width, **team)
    block = {k: v for k, v in (("engaged_tackle_logit", getattr(args, "engaged_tackle", None)),
                               ("shed_rate", getattr(args, "shed_rate", None)),
                               ("reengage_cooldown", getattr(args, "reengage_cooldown", None))) if v is not None}
    if block:
        cfg = dataclasses.replace(cfg, block=dataclasses.replace(cfg.block, **block))
    return cfg


def _value(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def run_args(run: Path) -> argparse.Namespace:
    raw = yaml.safe_load((Path(run) / "config.yaml").read_text())["args"]
    return argparse.Namespace(**{**LATER_ARGS, **{k: _value(v) for k, v in raw.items()}})


def config_from_run(run: Path):
    """The sim config a run was trained with (rebuilt from its config.yaml args)."""
    return build_config(run_args(run))


def exp_config(exp: Path | None, fallback):
    """The rules the runs under `exp` (s0/s1/...) were trained with, or `fallback`
    when there are no runs. Runs with different rules are an error: a report over
    them would mix two games."""
    runs = sorted(d for d in Path(exp).iterdir() if (d / "config.yaml").is_file()) if exp else []
    if not runs:
        return fallback
    cfg = config_from_run(runs[0])
    for r in runs[1:]:
        if config_from_run(r) != cfg:
            raise SystemExit(f"{r} has different rules from {runs[0]}")
    return cfg

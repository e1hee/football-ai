import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from sim.core import RewardConfig
from sim.team import TeamSim
from sim.team_policies import box_offense, gap_defense
from train.run_config import build_config, config_from_run, run_args

EXP019 = Path(__file__).resolve().parents[1] / "experiments" / "EXP-019-mix" / "s0"


def test_rebuilds_a_saved_run_exactly():
    saved = yaml.safe_load((EXP019 / "config.yaml").read_text())["sim"]
    now = json.loads(json.dumps(dataclasses.asdict(config_from_run(EXP019))))
    for later in ("ol_run_cost", "lb_rush_cost", "lb_chase_free", "lane_bonus", "lane_cap", "lane_box"):    # added after EXP-019
        assert now["reward"].pop(later) == dataclasses.asdict(RewardConfig())[later]
    assert now.pop("brake") == 1        # added in EXP-038: older runs keep the old brake curve
    assert now == saved


def test_old_runs_keep_the_old_brake_and_new_runs_get_the_new_one():
    old = TeamSim(1, config_from_run(EXP019), seed=0)
    new = TeamSim(1, build_config(argparse.Namespace(**{**vars(run_args(EXP019)), "brake": 2})), seed=0)
    dl = old.cfg.offense.__len__()      # first defender is a DL
    assert old.bodies.brake_drop[0, dl] == 1.0
    assert new.bodies.brake_drop[0, dl] < 0.5


def test_engaged_tackle_arg_only_moves_the_engaged_offset():
    a = run_args(EXP019)
    base = build_config(a)
    b = build_config(argparse.Namespace(**{**vars(a), "engaged_tackle": -2.4}))
    assert b.block.engaged_tackle_logit == -2.4
    assert dataclasses.replace(b, block=dataclasses.replace(b.block, engaged_tackle_logit=-1.5)) == base


def test_tackle_odds_are_reported_per_attempt():
    sim = TeamSim(256, config_from_run(EXP019), seed=1)
    sim.reset()
    seen = 0
    for _ in range(70):
        _, _, _, info = sim.step(box_offense(sim), gap_defense(sim))
        att, p = info["tackle_by_def"], info["tackle_p"]
        assert np.isnan(p[~att]).all() and ((p[att] > 0) & (p[att] < 1)).all()
        assert not (info["tackle_engaged"] & ~att).any()
        assert not (info["tackle_missed_by_def"] & ~att).any()
        seen += att.sum()
    assert seen > 0


def test_block_args_only_move_their_fields():
    a = run_args(EXP019)
    base = build_config(a)
    b = build_config(argparse.Namespace(**{**vars(a), "shed_rate": 0.15, "reengage_cooldown": 0.3}))
    assert (b.block.shed_rate, b.block.reengage_cooldown) == (0.15, 0.3)
    assert dataclasses.replace(b, block=dataclasses.replace(b.block, shed_rate=base.block.shed_rate,
                                                            reengage_cooldown=base.block.reengage_cooldown)) == base

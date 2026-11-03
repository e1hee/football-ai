"""Self-play: runner and tackler take turns learning against each other.

Phase 0 trains a fresh tackler against the EXP-002 champion runner; then the
sides alternate. A phase's learner faces an opponent *pool* so it can't just
overfit the latest rival (which is exactly how EXP-002 broke the scripted
tackler): 60% of plays vs the newest opponent, the rest split over older
versions and a scripted baseline.

Every phase saves `runner_gNN.zip` / `tackler_gNN.zip` (generation = how many
times that side has trained) and evaluates the newest pair plus fixed
references, so progress is measured against something that doesn't move.

    uv run python -m train.selfplay --exp EXP-003-selfplay-v1 --phases 24 --phase-steps 4_000_000 \
        --runner-init experiments/EXP-002-ppo-runner-v1/checkpoints/gen_025.zip
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from sim.core import FieldConfig, OneVOneSim, RewardConfig, SimConfig, TackleConfig
from sim.policies import angle_tackler, evasive_runner, read_react_tackler
from sim.team import SIMS, config_for
from sim.team_policies import (box_offense, combo_offense, gap_defense, lane_offense, pursuit_defense, read_defense,
                                screen_offense)
from train.evaluate import evaluate
from train.vec_env import SideVecEnv

ROOT = Path(__file__).resolve().parents[1]


class Frozen:
    """A saved PPO policy acting for one side, as `fn(sim) -> (N, act)`."""

    def __init__(self, path: Path, side: str):
        self.path, self.side = path, side
        self.model = PPO.load(str(path), device="cpu")

    def observe(self, sim: OneVOneSim) -> np.ndarray:
        return sim.observe() if self.side == "runner" else sim.observe_tackler()

    def act(self, obs: np.ndarray) -> np.ndarray:
        a, _ = self.model.predict(obs, deterministic=False)
        return a.astype(np.float32)

    def __call__(self, sim: OneVOneSim) -> np.ndarray:
        return self.act(self.observe(sim))


class Pool:
    """Assign each play slot to one opponent; recomputed whenever the pool changes.

    Frozen opponents only run on the slots assigned to them, from one shared
    observation: late in self-play the pool holds 12+ networks, and running
    each on every slot made the pool ~3/4 of the rollout time."""

    def __init__(self, n: int, opponents: list, weights: list[float], seed: int):
        rng = np.random.default_rng(seed)
        w = np.asarray(weights, float)
        self.assign = rng.choice(len(opponents), size=n, p=w / w.sum())
        self.opponents = opponents
        self.masks = [self.assign == k for k in range(len(opponents))]

    def __call__(self, sim: OneVOneSim) -> np.ndarray:
        obs: dict[str, np.ndarray] = {}
        acts = {}
        for k, opp in enumerate(self.opponents):
            m = self.masks[k]
            if not m.any():
                continue
            if isinstance(opp, Frozen):
                if opp.side not in obs:
                    obs[opp.side] = opp.observe(sim)
                acts[k] = opp.act(obs[opp.side][m])
            else:
                acts[k] = opp(sim)[m]
        out = np.zeros((sim.n, max(a.shape[1] for a in acts.values())), np.float32)
        for k, a in acts.items():   # 1v1 scripted 2-column actions get square-up 0
            out[self.masks[k], :a.shape[1]] = a
        return out


def make_pool(n: int, history: list, scripted, seed: int, latest_share: float = 0.6,
              scripted_share: float | None = None, skip_untrained: bool = False) -> Pool:
    """`scripted_share=None`: EXP-003 behaviour, the scripted opponent shares the
    leftover with the history (its weight shrinks as generations pile up).
    A number keeps it fixed, so learners never forget how to handle it.
    `skip_untrained`: generation 0 is a random policy; facing it teaches nothing
    useful (EXP-006: a runner that barely moves taught tacklers to flee), so it is
    left out, and with no trained opponent yet the pool is all scripted."""
    if skip_untrained:
        history = history[1:]
        if not history:
            return Pool(n, [scripted], [1.0], seed)
    if scripted_share is None:
        if len(history) == 1:
            return Pool(n, [history[-1], scripted], [0.8, 0.2], seed)
        older = history[:-1]
        rest = (1 - latest_share) / (len(older) + 1)
        return Pool(n, [history[-1], *older, scripted], [latest_share, *[rest] * len(older), rest], seed)
    older = history[:-1]
    if not older:
        return Pool(n, [history[-1], scripted], [1 - scripted_share, scripted_share], seed)
    rest = (1 - latest_share - scripted_share) / len(older)
    return Pool(n, [history[-1], *older, scripted], [latest_share, *[rest] * len(older), scripted_share], seed)


def as_runner(fn):
    return lambda sim, obs: fn(sim)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--phases", type=int, default=24, help="total phases; even = tackler learns, odd = runner")
    p.add_argument("--phase-steps", type=int, default=4_000_000)
    p.add_argument("--runner-init", type=Path, default=None, help="start the runner from this checkpoint (else fresh)")
    p.add_argument("--first", choices=["tackler", "runner"], default="tackler", help="which side learns in phase 0")
    p.add_argument("--scripted-share", type=float, default=None, help="fixed share of scripted opponents in every pool")
    p.add_argument("--delay", type=int, default=0, help="reaction delay in steps")
    p.add_argument("--scripted-tackler", choices=["angle", "read"], default="angle",
                   help="scripted tackler mixed into the runner's opponent pool")
    p.add_argument("--tackle-base", type=float, default=1.2)
    p.add_argument("--overspeed", type=float, default=0.0)
    p.add_argument("--sidestep", type=float, default=0.0)
    p.add_argument("--facing", action="store_true", help="body facing (hips), EXP-008+")
    p.add_argument("--n-envs", type=int, default=512)
    p.add_argument("--spread", type=float, default=8.0)
    p.add_argument("--breakaway-lead", type=float, default=None,
                   help="end plays once the runner leads every defender by this many yd (see scripts/breakaway_check.py)")
    p.add_argument("--full-play-share", type=float, default=0.1)
    p.add_argument("--yards-scale", type=float, default=0.1)
    p.add_argument("--blocker-shaping", type=float, default=0.0, help="team sim: OL shaping per yard (EXP-010)")
    p.add_argument("--sim", choices=SIMS, default="1v1",
                   help="3v3 = RB screen, lane(-lb) = Oklahoma drill (sim/team.py)")
    p.add_argument("--lane-width", type=float, default=5.0, help="lane drill: yd between the bags")
    p.add_argument("--stumble", type=float, default=0.0, help="team sim: s off balance per yd/s of hard collision")
    p.add_argument("--square-ahead", action="store_true", help="team sim: carrier squares up only to defenders ahead")
    p.add_argument("--timeout-ends", action="store_true", help="team sim: the play clock is a real end (no bootstrap)")
    p.add_argument("--net", type=int, default=128, help="hidden units per layer (2 layers)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    out = ROOT / "experiments" / args.exp
    ck = out / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=args.spread, breakaway_lead=args.breakaway_lead,
                                      full_play_share=args.full_play_share, reaction_delay=args.delay),
                    tackle=TackleConfig(base_logit=args.tackle_base, overspeed_weight=args.overspeed,
                                        sidestep_weight=args.sidestep),
                    reward=RewardConfig(yards_scale=args.yards_scale, blocker_shaping=args.blocker_shaping),
                    facing=args.facing)
    team = (dict(stumble=args.stumble, carrier_square_ahead=args.square_ahead, timeout_ends=args.timeout_ends)
            if args.sim != "1v1" else {})
    cfg = config_for(args.sim, cfg, lane_width=args.lane_width, **team)
    if args.sim == "3v3":
        script_runner = screen_offense
        script_tackler = {"angle": pursuit_defense, "read": read_defense}[args.scripted_tackler]
    elif args.sim.startswith("lane") or args.sim in ("combo", "box", "box-snap"):
        script_runner = {"combo": combo_offense, "box": box_offense, "box-snap": box_offense}.get(args.sim, lane_offense)
        script_tackler = gap_defense
    else:
        script_runner = evasive_runner
        script_tackler = {"angle": angle_tackler, "read": read_react_tackler}[args.scripted_tackler]
    (out / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(
        {"experiment": args.exp, "args": {k: str(v) for k, v in vars(args).items()}, "sim": dataclasses.asdict(cfg)})),
        allow_unicode=True, sort_keys=False))

    ppo_kw = dict(n_steps=64, batch_size=8192, n_epochs=5, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
                  clip_range=0.2, policy_kwargs={"net_arch": [args.net, args.net]}, device="cpu", verbose=0)

    # Generation 0 of each side: a given runner (e.g. the EXP-002 champion) or a fresh one,
    # and a fresh tackler.
    runner_path = ck / "runner_g00.zip"
    if args.runner_init is not None:
        runner_path.write_bytes(args.runner_init.read_bytes())
        runner_model = None
    else:
        venv_r = SideVecEnv(args.n_envs, cfg, script_tackler, side="runner", seed=args.seed)
        runner_model = PPO("MlpPolicy", venv_r, seed=args.seed, tensorboard_log=str(out / "tb"), **ppo_kw)
        runner_model.save(runner_path)
    venv_t = SideVecEnv(args.n_envs, cfg, script_tackler, side="tackler", seed=args.seed)
    tackler_model = PPO("MlpPolicy", venv_t, seed=args.seed, tensorboard_log=str(out / "tb"), **ppo_kw)
    tackler_path = ck / "tackler_g00.zip"
    tackler_model.save(tackler_path)
    runners = [Frozen(runner_path, "runner")]
    tacklers = [Frozen(tackler_path, "tackler")]

    log = open(out / "metrics.csv", "w", newline="")
    writer = None
    t0 = time.time()
    for phase in range(args.phases):
        first_is_tackler = args.first == "tackler"
        side = "tackler" if (phase % 2 == 0) == first_is_tackler else "runner"
        if side == "tackler":
            venv = SideVecEnv(args.n_envs, cfg, make_pool(args.n_envs, runners, script_runner, args.seed + phase,
                                                              scripted_share=args.scripted_share,
                                                              skip_untrained=args.runner_init is None),
                              side="tackler", seed=args.seed + phase)
            tackler_model.set_env(venv)
            tackler_model.learn(args.phase_steps, reset_num_timesteps=False, tb_log_name="tackler")
            gen = len(tacklers)
            path = ck / f"tackler_g{gen:02d}.zip"
            tackler_model.save(path)
            tacklers.append(Frozen(path, "tackler"))
        else:
            venv = SideVecEnv(args.n_envs, cfg, make_pool(args.n_envs, tacklers, script_tackler,
                                                              args.seed + phase,
                                                              scripted_share=args.scripted_share,
                                                              skip_untrained=True),
                              side="runner", seed=args.seed + phase)
            if runner_model is None:
                runner_model = PPO.load(str(runner_path), env=venv, device="cpu",
                                        tensorboard_log=str(out / "tb"), **{k: v for k, v in ppo_kw.items()
                                                                           if k not in ("policy_kwargs", "device")})
            else:
                runner_model.set_env(venv)
            runner_model.learn(args.phase_steps, reset_num_timesteps=False, tb_log_name="runner")
            gen = len(runners)
            path = ck / f"runner_g{gen:02d}.zip"
            runner_model.save(path)
            runners.append(Frozen(path, "runner"))

        # Deterministic evals on fixed starts.
        r_new = Frozen(ck / f"runner_g{len(runners) - 1:02d}.zip", "runner")
        t_new = Frozen(ck / f"tackler_g{len(tacklers) - 1:02d}.zip", "tackler")
        det = lambda f: (lambda sim, *a: f.model.predict(sim.observe() if f.side == "runner" else sim.observe_tackler(),
                                                         deterministic=True)[0])
        head = evaluate(det(r_new), det(t_new), cfg, n_plays=2048)
        r_vs_script = evaluate(det(r_new), script_tackler, cfg, n_plays=2048)
        t_vs_champ = evaluate(det(runners[0]), det(t_new), cfg, n_plays=2048)
        t_vs_script = evaluate(as_runner(script_runner), det(t_new), cfg, n_plays=2048)
        row = {
            "phase": phase, "learner": side, "runner_gen": len(runners) - 1, "tackler_gen": len(tacklers) - 1,
            "wall_s": round(time.time() - t0, 1),
            "td_latest_vs_latest": head["rate/touchdown"], "oob_latest_vs_latest": head["rate/out_of_bounds"],
            "yards_latest_vs_latest": head["yards/mean"],
            "td_runner_vs_scripted_tackler": r_vs_script["rate/touchdown"],
            "td_champion_runner_vs_tackler": t_vs_champ["rate/touchdown"],
            "td_scripted_runner_vs_tackler": t_vs_script["rate/touchdown"],
        }
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row))
            writer.writeheader()
        writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        log.flush()
        print(f"[phase {phase:2d} {side:>7} learned] R{row['runner_gen']:02d} vs T{row['tackler_gen']:02d}: "
              f"TD {head['rate/touchdown']:.1%} oob {head['rate/out_of_bounds']:.1%} yd {head['yards/mean']:.1f} | "
              f"R vs scripted T {r_vs_script['rate/touchdown']:.1%} | champion R vs T {t_vs_champ['rate/touchdown']:.1%} | "
              f"scripted R vs T {t_vs_script['rate/touchdown']:.1%} | {row['wall_s']:.0f}s", flush=True)
    log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

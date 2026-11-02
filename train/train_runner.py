"""Train one side with PPO against a fixed opponent (scripted or a frozen checkpoint).

Default: the runner learns against the scripted angle tackler. With
`--side tackler --opponent ckpt:path/to/runner.zip` the tackler learns instead.

Each "generation" is a checkpoint saved at an even fraction of training
(gen 0 = untrained). Every generation is evaluated on the same 4096 starting
setups, so curves and side-by-side clips compare like with like.

    uv run python -m train.train_runner --exp EXP-002-ppo-runner-v1 --timesteps 20_000_000 --gens 20
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from sim.core import FieldConfig, RewardConfig, SimConfig, TackleConfig
from sim.policies import angle_tackler, pursuit_tackler, read_react_tackler
from sim.policies import evasive_runner, straight_runner
from train.evaluate import evaluate, ppo_runner
from train.vec_env import SideVecEnv

ROOT = Path(__file__).resolve().parents[1]
TACKLERS = {"angle": angle_tackler, "pursuit": pursuit_tackler, "read": read_react_tackler}


def make_runner(name: str):
    """Fixed runner opponent as fn(sim, obs) -> actions."""
    if name.startswith("ckpt:"):
        return ppo_runner(name[5:])
    return {"evasive": lambda sim, obs: evasive_runner(sim), "straight": lambda sim, obs: straight_runner(sim)}[name]


class GenerationCallback(BaseCallback):
    """Save + evaluate every `every` env steps; log rollout stats each rollout."""

    def __init__(self, out: Path, every: int, eval_fn, venv: RunnerVecEnv):
        super().__init__()
        self.out, self.every, self.eval_fn, self.venv = out, every, eval_fn, venv
        self.next_gen_at = every
        self.gen = 0
        self.t0 = time.time()
        self.csv_path = out / "metrics.csv"
        self._writer = None

    def save_generation(self) -> None:
        path = self.out / "checkpoints" / f"gen_{self.gen:03d}.zip"
        self.model.save(path)
        res = self.eval_fn(self.model)
        row = {"gen": self.gen, "timesteps": self.num_timesteps, "wall_s": round(time.time() - self.t0, 1), **res}
        for k, v in res.items():
            self.logger.record(f"eval/{k}", v)
        if self._writer is None:
            f = open(self.csv_path, "w", newline="")
            self._writer = (f, csv.DictWriter(f, fieldnames=list(row)))
            self._writer[1].writeheader()
        self._writer[1].writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        self._writer[0].flush()
        print(f"[gen {self.gen:3d}] {self.num_timesteps:>11,} steps  {row['wall_s']:7.0f}s  "
              f"TD {res['rate/touchdown']:.1%}  tackled {res['rate/tackled']:.1%}  "
              f"oob {res['rate/out_of_bounds']:.1%}  timeout {res['rate/timeout']:.1%}  "
              f"yards {res['yards/mean']:.1f}  broken {res['tackle/broken_per_play']:.2f}", flush=True)
        self.gen += 1

    def _on_training_start(self) -> None:
        self.save_generation()   # gen 0: untrained

    def _on_rollout_end(self) -> None:
        for k, v in self.venv.pop_stats().items():
            self.logger.record(f"train/{k}", v)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_gen_at:
            self.save_generation()
            self.next_gen_at += self.every
        return True

    def _on_training_end(self) -> None:
        if self._writer:
            self._writer[0].close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=512)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent", type=float, default=0.0)
    p.add_argument("--net", type=int, nargs="+", default=[128, 128])
    p.add_argument("--tackler", choices=list(TACKLERS), default="angle", help="scripted opponent when --side runner")
    p.add_argument("--side", choices=["runner", "tackler"], default="runner")
    p.add_argument("--opponent", default=None,
                   help="when --side tackler: evasive | straight | ckpt:path/to/runner.zip")
    p.add_argument("--spread", type=float, default=8.0)
    p.add_argument("--breakaway-lead", type=float, default=None,
                   help="end plays once the runner leads every defender by this many yd (see scripts/breakaway_check.py)")
    p.add_argument("--full-play-share", type=float, default=0.1)
    p.add_argument("--delay", type=int, default=0, help="reaction delay in steps")
    p.add_argument("--tackle-base", type=float, default=1.2)
    p.add_argument("--overspeed", type=float, default=0.0)
    p.add_argument("--sidestep", type=float, default=0.0)
    p.add_argument("--facing", action="store_true", help="body facing (hips), EXP-008+")
    p.add_argument("--yards-scale", type=float, default=0.1, help="reward per yard (scaled for PPO)")
    p.add_argument("--td-bonus", type=float, default=0.0)
    p.add_argument("--tackle-penalty", type=float, default=0.0)
    p.add_argument("--oob-penalty", type=float, default=0.0)
    p.add_argument("--time-penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    out = ROOT / "experiments" / args.exp
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        field=FieldConfig(tackler_lateral_spread=args.spread, breakaway_lead=args.breakaway_lead,
                          full_play_share=args.full_play_share, reaction_delay=args.delay),
        tackle=TackleConfig(base_logit=args.tackle_base, overspeed_weight=args.overspeed,
                            sidestep_weight=args.sidestep),
        reward=RewardConfig(yards_scale=args.yards_scale, touchdown_bonus=args.td_bonus,
                            tackle_penalty=args.tackle_penalty, out_of_bounds_penalty=args.oob_penalty,
                            time_penalty=args.time_penalty),
        facing=args.facing,
    )
    if args.side == "runner":
        tackler = TACKLERS[args.tackler]
        venv = SideVecEnv(args.n_envs, cfg, tackler, side="runner", seed=args.seed)
    else:
        runner_eval = make_runner(args.opponent or "evasive")
        venv = SideVecEnv(args.n_envs, cfg, lambda sim: runner_eval(sim, sim.observe()), side="tackler", seed=args.seed)

    cfg_dump = {"experiment": args.exp, "args": vars(args), "sim": dataclasses.asdict(cfg)}
    (out / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg_dump)), allow_unicode=True, sort_keys=False))

    model = PPO(
        "MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs,
        learning_rate=args.lr, gamma=args.gamma, gae_lambda=0.95, ent_coef=args.ent, clip_range=0.2,
        policy_kwargs={"net_arch": args.net}, tensorboard_log=str(out / "tb"), seed=args.seed,
        device="cpu", verbose=0,
    )

    def eval_fn(m):
        if args.side == "runner":
            def act(sim, obs):
                return m.predict(obs, deterministic=True)[0]
            return evaluate(act, tackler, cfg, n_plays=4096)

        def t_act(sim, *_):
            return m.predict(sim.observe_tackler(), deterministic=True)[0]
        return evaluate(runner_eval, t_act, cfg, n_plays=4096)

    every = args.timesteps // args.gens
    cb = GenerationCallback(out, every, eval_fn, venv)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb, tb_log_name="ppo", progress_bar=False)
    if cb.gen <= args.gens:
        cb.save_generation()
    dt = time.time() - t0
    print(f"done: {model.num_timesteps:,} steps in {dt / 60:.1f} min ({model.num_timesteps / dt:,.0f} steps/s)")


if __name__ == "__main__":
    main()

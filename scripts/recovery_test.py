"""Can a side still learn? Train one side against one fixed opponent and watch it.

EXP-022's s2 defense spent 24M steps losing 98% of plays to one runner without
recovering. This trains a single arm against just that opponent, from the
existing weights or from scratch, with the normal reward or with potential-based
pursuit shaping (the total over a play doesn't depend on the path, so the best
policy stays the same; it only adds a signal on plays with no contact).

    uv run python -m scripts.recovery_test --name warm --learner tackler --run experiments/EXP-022-long/s2 \
        --opp-gen 36 --init-gen 36 --out experiments/EXP-023-audit/warm.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from sim.core import OUTCOME_TOUCHDOWN
from sim.team import CARRIER, TeamSim
from train.run_config import config_from_run
from train.selfplay import Frozen
from train.vec_env import SideVecEnv

PPO_KW = dict(n_steps=64, batch_size=8192, n_epochs=5, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, clip_range=0.2)


class ShapedTackler(SideVecEnv):
    """Tackler env with shaping F = gamma * Phi(s') - Phi(s), Phi = -scale * yards to
    the carrier from the nearest defender, and Phi = 0 once the play is over."""

    def __init__(self, *a, scale: float, gamma: float, **k):
        super().__init__(*a, **k)
        self.scale, self.gamma = scale, gamma

    def _phi(self) -> np.ndarray:
        s = self.sim
        return -self.scale * np.linalg.norm(s.pos[:, s.dfn] - s.pos[:, CARRIER, None], axis=-1).min(1)

    def step_wait(self):
        before = self._phi()
        obs, rew, done, infos = super().step_wait()
        after = np.where(done, 0.0, self._phi())
        return obs, rew + (self.gamma * after - before).astype(np.float32), done, infos


def measure(runner, tackler, cfg, n: int = 2048, seed: int = 7) -> dict:
    """Deterministic both ways, as the box report does."""
    sim = TeamSim(n, cfg, seed=seed)
    sim.reset()
    live = np.ones(n, bool)
    out, yards, end_y, tries = np.zeros(n, np.int8), np.zeros(n), np.zeros(n), np.zeros(n)
    while live.any():
        a = runner.predict(sim.observe(), deterministic=True)[0]
        b = tackler.predict(sim.observe_tackler(), deterministic=True)[0]
        _, _, done, info = sim.step(a, b)
        tries += info["attempts"] * live
        new = done & live
        out[new], yards[new], end_y[new] = info["outcome"][new], info["yards"][new], info["r_pos"][new, 1]
        live &= ~done
    return {"td": float((out == OUTCOME_TOUCHDOWN).mean()), "yards": float(yards.mean()),
            "end_right": float((end_y > cfg.field.width / 2).mean()), "any_try": float((tries > 0).mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--learner", choices=["tackler", "runner"], required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--opp-gen", type=int, default=36)
    p.add_argument("--init-gen", type=int, default=None, help="start from this generation; omit for a fresh network")
    p.add_argument("--shaping", type=float, default=0.0, help="reward per yard closer (tackler only)")
    p.add_argument("--steps", type=int, default=12_000_000)
    p.add_argument("--every", type=int, default=2_000_000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-std", action="store_true", help="keep the weights, put exploration noise back to std 1")
    p.add_argument("--save", type=Path, default=None, help="save the final policy here")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    cfg = config_from_run(args.run)
    ck = args.run / "checkpoints"
    other = "runner" if args.learner == "tackler" else "tackler"
    opp = Frozen(ck / f"{other}_g{args.opp_gen:02d}.zip", other)
    if args.shaping:
        venv = ShapedTackler(512, cfg, opp, side=args.learner, seed=args.seed, scale=args.shaping, gamma=PPO_KW["gamma"])
    else:
        venv = SideVecEnv(512, cfg, opp, side=args.learner, seed=args.seed)
    if args.init_gen is None:
        net = int(json.loads(json.dumps(__import__("yaml").safe_load((args.run / "config.yaml").read_text())))["args"]["net"])
        model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0, policy_kwargs={"net_arch": [net, net]}, **PPO_KW)
    else:
        model = PPO.load(str(ck / f"{args.learner}_g{args.init_gen:02d}.zip"), env=venv, device="cpu", **PPO_KW)
    if args.reset_std:
        with torch.no_grad():
            model.policy.log_std.zero_()

    def row(done_steps: int) -> dict:
        r, t = (model, opp.model) if args.learner == "runner" else (opp.model, model)
        return {"name": args.name, "steps": done_steps, **measure(r, t, cfg),
                "std": float(model.policy.log_std.detach().exp().mean())}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(json.dumps(row(0)) + "\n")
        f.flush()
        for done_steps in range(args.every, args.steps + 1, args.every):
            model.learn(args.every, reset_num_timesteps=False)
            f.write(json.dumps(row(done_steps)) + "\n")
            f.flush()
    if args.save:
        model.save(args.save)


if __name__ == "__main__":
    main()

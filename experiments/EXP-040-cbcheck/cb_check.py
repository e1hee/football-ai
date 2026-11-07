"""EXP-040: why the corner learns nothing from the yards reward (EXP-039). A corner alone
against the scripted route tree, PPO as in self-play, `--sim-steps` of experience.

Variants: action repeat K (the CB decides every K steps, the push held in between; the
learner gets the summed reward) and deterministic catches (`PassConfig.catch_sep`).
Every --eval-every sim steps: scripted routes vs the deterministic CB, 2048 plays, on the
variant's own rules and on EXP-039's rules.

    uv run python experiments/EXP-040-cbcheck/cb_check.py rep3 --repeat 3
"""
import argparse, dataclasses, json, sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.core import OUTCOME_NAMES, OUTCOME_TIMEOUT
from sim.pass_policies import route_offense
from train.evaluate import evaluate
from train.policies import policy_for
from train.run_config import config_from_run
from train.vec_env import SideVecEnv


class RepeatTackler(SideVecEnv):
    """The tackler's action is held for `k` sim steps. A slot whose play ends inside the hold
    reports that end (summed reward up to it, its terminal obs) and pushes 0 for the rest
    of the hold (the new play's first steps)."""

    def __init__(self, *a, k: int = 1, **kw):
        super().__init__(*a, side="tackler", **kw)
        self.k = k

    def step_wait(self):
        n = self.num_envs
        act = self._actions.copy()
        total = np.zeros(n, np.float32)
        ended = np.zeros(n, bool)
        term = np.zeros((n, self.sim.obs_dim), np.float32)
        out = np.zeros(n, np.int8)
        yards = np.zeros(n, np.float32)
        steps = np.zeros(n, np.int64)
        for _ in range(self.k):
            a = act.copy()
            a[ended] = 0.0
            _, _, done, info = self.sim.step(self.opponent(self.sim), a)
            total += info["tackler_reward"] * ~ended
            new = done & ~ended
            if new.any():
                term[new] = info["terminal_obs_tackler"][new]
                out[new], yards[new], steps[new] = info["outcome"][new], info["yards"][new], info["steps"][new]
            ended |= done
        obs = self._my_obs()
        self.ep_returns += total
        infos = [{} for _ in range(n)]
        for i in np.nonzero(ended)[0]:
            self._outcomes[out[i]] += 1
            self._yards.append(float(yards[i]))
            infos[i] = {"terminal_observation": term[i], "TimeLimit.truncated": False,
                        "episode": {"r": float(self.ep_returns[i]), "l": int(steps[i])}, "outcome": OUTCOME_NAMES[out[i]]}
            self.ep_returns[i] = 0
        return obs, total, ended, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--catch-sep", type=float, default=None)
    ap.add_argument("--sim-steps", type=float, default=3e6)
    ap.add_argument("--eval-every", type=float, default=1e6)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parent
    base = config_from_run(root.parent / "EXP-039-pass" / "s0")
    cfg = dataclasses.replace(base, catch_sep=args.catch_sep)
    env = RepeatTackler(512, cfg, route_offense, k=args.repeat, seed=0)
    pol, kw = policy_for(1.25, 256)
    m = PPO(pol, env, n_steps=64, batch_size=8192, n_epochs=5, learning_rate=3e-4, gamma=0.995 ** args.repeat,
            gae_lambda=0.95, policy_kwargs=kw, device="cpu", verbose=0, seed=0)
    ro = lambda sim, obs: route_offense(sim)

    def cb(sim, *_):
        # Deterministic CB, deciding every `repeat` steps like in training.
        if not hasattr(sim, "_held") or sim.t.max() == 0 and not hasattr(sim, "_init"):
            sim._held, sim._age, sim._init = np.zeros((sim.n, 3), np.float32), np.full(sim.n, 10 ** 6), True
        fresh = (sim._age >= args.repeat) | (sim.t == 0)
        if fresh.any():
            a = m.predict(sim.observe_tackler(), deterministic=True)[0]
            sim._held[fresh] = a[fresh]
            sim._age[fresh] = 0
        sim._age += 1
        return sim._held.copy()

    rows = []
    done_steps = 0
    while True:
        r_own = evaluate(ro, cb, cfg, n_plays=2048)
        r_039 = evaluate(ro, cb, base, n_plays=2048) if args.catch_sep is not None else r_own
        row = {"sim_steps": done_steps, "yards_own_rules": round(r_own["yards/mean"], 2),
               "yards_exp039_rules": round(r_039["yards/mean"], 2),
               "tackled_exp039_rules": round(r_039["rate/tackled"], 3), "std": round(float(m.policy.log_std.exp().mean()), 3)}
        rows.append(row)
        print(args.name, row, flush=True)
        if done_steps >= args.sim_steps:
            break
        m.learn(int(args.eval_every / args.repeat), reset_num_timesteps=False)
        done_steps += int(args.eval_every)
    (root / f"{args.name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

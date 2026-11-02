"""Expose the batched sim to Stable-Baselines3 as a VecEnv without copying
plays into Python objects: one `OneVOneSim` with N plays = N "envs".

Either side can be the learner. The other side is `opponent(sim) -> (N, 2)`:
a scripted policy, a frozen learned policy, or a mix (self-play pools).
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from sim.core import OBS_DIM, OUTCOME_NAMES, OUTCOME_TIMEOUT, OneVOneSim, SimConfig

Opponent = Callable[[OneVOneSim], np.ndarray]

# Action box half-width. SB3 clips each axis to the box *separately*; with a
# [-1, 1] box, PPO's Gaussian mean drifts outside and every clipped action lands
# on an edge or corner: in EXP-005 96% of tackler actions fell into 8 directions
# (a keyboard, not a joystick). With a wide box nothing gets clipped per axis,
# and the sim scales any push longer than 1 back onto the unit circle along its
# own direction (physics.apply_controls), so direction is always preserved.
ACTION_BOUND = 5.0


class SideVecEnv(VecEnv):
    """The learner controls `side` ("runner" or "tackler"); `opponent` the other."""

    def __init__(self, n_envs: int, cfg: SimConfig, opponent: Opponent, side: str = "runner", seed: int = 0):
        assert side in ("runner", "tackler")
        self.side = side
        self.sim = OneVOneSim(n_envs, cfg, seed=seed)
        self.opponent = opponent
        obs_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        act_space = gym.spaces.Box(-ACTION_BOUND, ACTION_BOUND, (2,), np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions: np.ndarray | None = None
        self.ep_returns = np.zeros(n_envs, np.float32)
        self._outcomes = np.zeros(len(OUTCOME_NAMES), np.int64)
        self._yards: list[float] = []

    def _my_obs(self) -> np.ndarray:
        return self.sim.observe() if self.side == "runner" else self.sim.observe_tackler()

    # SB3 VecEnv API ----------------------------------------------------------
    def reset(self):
        self.ep_returns[:] = 0
        self.sim.reset()
        return self._my_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, np.float32)

    def step_wait(self):
        other = self.opponent(self.sim)
        if self.side == "runner":
            _, rew, done, info = self.sim.step(self._actions, other)
            term = info["terminal_obs"]
        else:
            _, r_rew, done, info = self.sim.step(other, self._actions)
            rew = info["tackler_reward"]
            term = info["terminal_obs_tackler"]
        obs = self._my_obs()
        self.ep_returns += rew
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        if done.any():
            idx = np.nonzero(done)[0]
            np.add.at(self._outcomes, info["outcome"][idx], 1)
            self._yards.extend(info["yards"][idx].tolist())
            for i in idx:
                infos[i] = {
                    "terminal_observation": term[i],
                    # Running out the play clock is a cut-off, not a real end.
                    "TimeLimit.truncated": bool(info["outcome"][i] == OUTCOME_TIMEOUT),
                    "episode": {"r": float(self.ep_returns[i]), "l": int(info["steps"][i])},
                    "outcome": OUTCOME_NAMES[info["outcome"][i]],
                }
            self.ep_returns[idx] = 0
        return obs, rew, done, infos

    def pop_stats(self) -> dict:
        n = int(self._outcomes.sum())
        out = {f"rate/{k}": c / n for k, c in zip(OUTCOME_NAMES, self._outcomes) if k != "running"} if n else {}
        if self._yards:
            out["yards/mean"] = float(np.mean(self._yards))
        out["plays"] = n
        self._outcomes[:] = 0
        self._yards.clear()
        return out

    def close(self) -> None:
        pass

    def seed(self, seed: int | None = None):
        self.sim.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    # Unused parts of the VecEnv interface.
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class RunnerVecEnv(SideVecEnv):
    """Kept for EXP-002's train_runner.py: learner = runner."""

    def __init__(self, n_envs: int, cfg: SimConfig, opponent: Opponent, seed: int = 0):
        super().__init__(n_envs, cfg, opponent, side="runner", seed=seed)

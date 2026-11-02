"""Expose the batched sim to Stable-Baselines3 as a VecEnv without copying
plays into Python objects: one `OneVOneSim` with N plays = N "envs".

The opponent is a fixed function of the sim (a scripted policy now, a
frozen learned policy later for self-play).
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from sim.core import OBS_DIM, OUTCOME_NAMES, OUTCOME_TIMEOUT, OneVOneSim, SimConfig

Opponent = Callable[[OneVOneSim], np.ndarray]


class RunnerVecEnv(VecEnv):
    """The learner controls the runner; `opponent` controls the tackler."""

    def __init__(self, n_envs: int, cfg: SimConfig, opponent: Opponent, seed: int = 0):
        self.sim = OneVOneSim(n_envs, cfg, seed=seed)
        self.opponent = opponent
        obs_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions: np.ndarray | None = None
        self.ep_returns = np.zeros(n_envs, np.float32)
        # Rolling outcome counts since the last `pop_stats()` (for logging).
        self._outcomes = np.zeros(len(OUTCOME_NAMES), np.int64)
        self._yards: list[float] = []

    # SB3 VecEnv API ----------------------------------------------------------
    def reset(self):
        self.ep_returns[:] = 0
        return self.sim.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, np.float32)

    def step_wait(self):
        obs, rew, done, info = self.sim.step(self._actions, self.opponent(self.sim))
        self.ep_returns += rew
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        if done.any():
            term = info["terminal_obs"]
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

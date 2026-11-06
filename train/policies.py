"""Policies with a bounded mean push (EXP-030).

The team sim scales any push longer than 1 down to length 1, so a policy loses nothing by
making its mean push longer and longer. EXP-029 found the trained offense's mean push grew to
3-4 over 12 generations, and the same exploration noise then turned the runner's push by 10-15
degrees instead of ~30: the offense stopped trying other directions. The defense grew too.

Here each player's mean push (x, y) is squashed along its length to at most `bound`,
v * bound * tanh(|v| / bound) / |v|, so its direction is untouched and still learns. The third
action of each player (hips, clipped to [0, 1] by the sim) is squashed the same way on its own.
The noise is unchanged. Actions come in groups of 3 per player; any other size is squashed per
component.
"""
from __future__ import annotations

import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy


def squash(x: torch.Tensor, bound: float) -> torch.Tensor:
    if x.shape[-1] % 3:
        return bound * torch.tanh(x / bound)
    p = x.reshape(*x.shape[:-1], -1, 3)
    xy, hips = p[..., :2], p[..., 2:]
    r = xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    xy = xy * (bound * torch.tanh(r / bound) / r)
    return torch.cat([xy, bound * torch.tanh(hips / bound)], dim=-1).reshape(x.shape)


class BoundedPushPolicy(ActorCriticPolicy):
    def __init__(self, *args, push_bound: float = 1.25, **kwargs):
        self.push_bound = float(push_bound)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict:
        return {**super()._get_constructor_parameters(), "push_bound": self.push_bound}

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        return self.action_dist.proba_distribution(squash(self.action_net(latent_pi), self.push_bound), self.log_std)


def policy_for(push_bound: float | None, net: int) -> tuple:
    """(policy, policy_kwargs) for PPO: the plain MLP policy, or the bounded one."""
    if push_bound is None:
        return "MlpPolicy", {"net_arch": [net, net]}
    return BoundedPushPolicy, {"net_arch": [net, net], "push_bound": float(push_bound)}

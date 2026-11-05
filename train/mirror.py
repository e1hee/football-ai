"""Left-right mirror of the box-mix game (RB + 5 OL vs 4 DL + 2 LB).

The game is symmetric but a network isn't: EXP-023 found a defense that held its
own runner to 15% and gave up 100% to the same runner flipped. A flipped opponent
is an equally legal opponent, so a learner can be made to face both.
"""
from __future__ import annotations

import numpy as np

# New slot i takes old slot PERM[i]: RB, then OL left to right; DL left to right, LBs.
OFFENSE = [0, 5, 4, 3, 2, 1]
DEFENSE = [3, 2, 1, 0, 5, 4]
PLAYER_OBS = 8
N_GLOBAL = 5
FLIP_OBS = [1, 3, 5]      # rel y, vy, facing y


def mirror_obs(o: np.ndarray, side: str) -> np.ndarray:
    """(N, obs) as the flipped play looks to `side` ("runner"/"tackler"); own players come first."""
    own, other = (OFFENSE, DEFENSE) if side == "runner" else (DEFENSE, OFFENSE)
    out = o.copy()
    out[:, 1] = 1 - out[:, 1]
    per = o[:, N_GLOBAL:].reshape(len(o), len(own) + len(other), PLAYER_OBS).copy()
    per[..., FLIP_OBS] *= -1
    out[:, N_GLOBAL:] = per[:, own + [i + len(own) for i in other]].reshape(len(o), o.shape[1] - N_GLOBAL)
    return out


def mirror_action(a: np.ndarray, side: str) -> np.ndarray:
    """(N, players * 3) team action flipped back: push y changes sign, slots swap sides."""
    own = OFFENSE if side == "runner" else DEFENSE
    out = np.array(a).reshape(len(a), len(own), 3).copy()
    out[..., 1] *= -1
    return out[:, own].reshape(len(a), len(own) * 3)

"""Run plays and record every position so they can be drawn later.

A policy is `fn(sim, obs) -> actions (N,2)`; scripted policies ignore `obs`,
learned ones ignore `sim`. Both sides use the same signature so a clip can
pit any runner against any tackler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from sim.core import OUTCOME_NAMES, OneVOneSim, SimConfig

Policy = Callable[[OneVOneSim, np.ndarray], np.ndarray]


@dataclass
class Play:
    r_pos: np.ndarray   # (T, 2) runner positions, including the start
    t_pos: np.ndarray   # (T, 2) tackler positions
    outcome: str
    yards: float
    events: list[tuple[int, str]] = field(default_factory=list)  # (step, "made" | "missed")

    @property
    def steps(self) -> int:
        return len(self.r_pos) - 1


def record_plays(
    runner: Policy,
    tackler: Policy,
    n_plays: int = 8,
    cfg: SimConfig | None = None,
    seed: int = 0,
) -> list[Play]:
    """Run `n_plays` in parallel and return each one's full trajectory
    (only the first play per slot; auto-resets after that are ignored)."""
    sim = OneVOneSim(n_plays, cfg, seed=seed)
    obs = sim.reset()
    r_hist = [sim.r_pos.copy()]
    t_hist = [sim.t_pos.copy()]
    finished = np.zeros(n_plays, bool)
    end_step = np.full(n_plays, -1)
    outcome = np.zeros(n_plays, np.int8)
    yards = np.zeros(n_plays, np.float32)

    events: list[list[tuple[int, str]]] = [[] for _ in range(n_plays)]
    step = 0
    while not finished.all():
        obs, _, done, info = sim.step(runner(sim, obs), tackler(sim, obs))
        step += 1
        for kind, key in (("made", "tackle_made"), ("missed", "tackle_missed")):
            for i in np.nonzero(info[key] & ~finished)[0]:
                events[i].append((step, kind))
        # info holds positions *before* auto-reset, which is what we want.
        r_hist.append(info["r_pos"])
        t_hist.append(info["t_pos"])
        newly = done & ~finished
        end_step[newly] = step
        outcome[newly] = info["outcome"][newly]
        yards[newly] = info["yards"][newly]
        finished |= done

    r_all = np.stack(r_hist)  # (T+1, N, 2)
    t_all = np.stack(t_hist)
    return [
        Play(r_all[: end_step[i] + 1, i], t_all[: end_step[i] + 1, i],
             OUTCOME_NAMES[outcome[i]], float(yards[i]), events[i])
        for i in range(n_plays)
    ]


def pick(plays: list[Play], outcome: str | None = None, longest: bool = False) -> Play:
    """Choose a play that tells the story: a given outcome, or the longest."""
    pool = [p for p in plays if outcome is None or p.outcome == outcome] or plays
    return max(pool, key=lambda p: p.steps) if longest else pool[0]

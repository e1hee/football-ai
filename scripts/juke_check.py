"""Does braking before a cut help the runner? (user question after EXP-007)

A scripted juke runner: run like `evasive_runner`; when the tackler is ahead
and within `trigger` yards, brake for `brake_steps`, then cut hard away from
him, then sprint again. Compared with the learned runner and plain evasion,
under the EXP-007 sim settings.

    uv run python -m scripts.juke_check
"""
import glob
import numpy as np
from sim.core import RUNNER, TACKLER, FieldConfig, SimConfig, TackleConfig
from sim.policies import evasive_runner, angle_tackler, read_react_tackler
from train.evaluate import evaluate, ppo_runner, ppo_tackler

cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=8.0, reaction_delay=2),
                tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2))

def juke_runner(trigger, brake_steps, cut_steps=4):
    st = {}
    def act(sim, obs):
        n = sim.n
        if "phase" not in st or st["phase"].shape[0] != n:
            st.update(phase=np.zeros(n, int), k=np.zeros(n, int), side=np.zeros(n), lastx=sim.r_pos[:, 0].copy())
        new = sim.r_pos[:, 0] < st["lastx"] - 5           # slot restarted a play
        st["phase"][new] = 0; st["lastx"] = sim.r_pos[:, 0].copy()
        t_pos, _ = sim.seen(TACKLER)
        rel = t_pos - sim.r_pos; dist = np.linalg.norm(rel, axis=1)
        a = evasive_runner(sim)
        ph, k = st["phase"], st["k"]
        go = (ph == 0) & (rel[:, 0] > 0.5) & (dist < trigger)
        mid = sim.cfg.field.width / 2
        away = -np.sign(rel[:, 1] + 1e-3)
        near_edge = np.minimum(sim.r_pos[:, 1], sim.cfg.field.width - sim.r_pos[:, 1]) < 5
        st["side"] = np.where(go, np.where(near_edge, np.sign(mid - sim.r_pos[:, 1]), away), st["side"])
        ph[go] = 1 if brake_steps else 2; k[go] = 0
        v = sim.vel[:, RUNNER]; sp = np.linalg.norm(v, axis=1, keepdims=True)
        brake = -v / np.maximum(sp, 1e-6)
        cut = np.stack([0.25 * np.ones(n), st["side"]], 1)
        a = np.where((ph == 1)[:, None], brake, a)
        a = np.where((ph == 2)[:, None], cut, a)
        k[(ph == 1) | (ph == 2)] += 1
        to_cut = (ph == 1) & (k >= brake_steps); ph[to_cut] = 2; k[to_cut] = 0
        done = (ph == 2) & (k >= cut_steps); ph[done] = 3
        return a.astype(np.float32)
    return act

exp = glob.glob("experiments/EXP-007*")[0]
tacklers = {"T12(AI)": ppo_tackler(f"{exp}/checkpoints/tackler_g12.zip"),
            "angle": angle_tackler, "read": read_react_tackler}
runners = {"R12(AI)": ppo_runner(f"{exp}/checkpoints/runner_g12.zip"),
           "evasive": lambda s, o: evasive_runner(s)}
for trig in (2.5, 4.0, 5.5):
    for b in (0, 2, 4):
        runners[f"juke d{trig} brake{b*0.1:.1f}s"] = juke_runner(trig, b)
print(f"{'runner':24s}" + "".join(f"{t:>16s}" for t in tacklers))
for rn, r in runners.items():
    row = []
    for tn, t in tacklers.items():
        res = evaluate(r, t, cfg, n_plays=4096)
        row.append(f"{res['rate/touchdown']*100:5.1f}% {res['yards/mean']:4.1f}y")
    print(f"{rn:24s}" + "".join(f"{c:>16s}" for c in row), flush=True)

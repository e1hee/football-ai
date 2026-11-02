"""Where does pre-contact deceleration come from (EXP-008)? For each learned
player step slower by 2+ yd/s^2 (moving > 2 yd/s, not stunned, no tackle
under way), was it pushing against its own motion (a real brake) or still
pushing forward (slowed by the hips speed cap)?

    uv run python -m scripts.decel_source experiments/EXP-008-facing
"""
import sys

import numpy as np
from stable_baselines3 import PPO
from sim.core import RUNNER, TACKLER, FieldConfig, OneVOneSim, SimConfig, TackleConfig
E = sys.argv[1]
cfg = SimConfig(field=FieldConfig(tackler_lateral_spread=8.0, reaction_delay=2),
                tackle=TackleConfig(base_logit=2.3, overspeed_weight=0.25, sidestep_weight=0.2), facing=True)
R = PPO.load(f"{E}/checkpoints/runner_g12.zip", device="cpu"); T = PPO.load(f"{E}/checkpoints/tackler_g12.zip", device="cpu")
sim = OneVOneSim(2048, cfg, seed=31); obs = sim.reset(); live = np.ones(sim.n, bool)
rows = {RUNNER: [], TACKLER: []}
while live.any():
    ra, _ = R.predict(obs, deterministic=True); ta, _ = T.predict(sim.observe_tackler(), deterministic=True)
    v0 = sim.vel.copy(); sp0 = np.linalg.norm(v0, axis=-1); stun = sim.stun.copy()
    acts = {RUNNER: ra, TACKLER: ta}
    obs, _, done, info = sim.step(ra, ta)
    pre_contact = ~(info["tackle_attempt"] | info["wrapped"])
    sp1 = np.linalg.norm(sim.vel, axis=-1)
    for who in (RUNNER, TACKLER):
        a = np.clip(acts[who][:, :2], -5, 5); a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0)
        along = np.sum(a * v0[:, who], axis=1) / np.maximum(sp0[:, who], 1e-6)
        dec = (sp0[:, who] - sp1[:, who]) / 0.1
        ok = live & ~done & pre_contact & (sp0[:, who] > 2) & (stun[:, who] <= 0)
        rows[who].append(np.stack([dec[ok], along[ok]], 1))
    live &= ~done
for who, name in ((RUNNER, "runner"), (TACKLER, "tackler")):
    d = np.concatenate(rows[who]); hard = d[d[:, 0] >= 2.0]
    print(f"{name}: steps {len(d)}, decel>=2 share {len(hard)/len(d):.1%}; of those, push against motion (along<-0.2) {np.mean(hard[:,1] < -0.2):.0%}, "
          f"push still forward (along>0.2) {np.mean(hard[:,1] > 0.2):.0%}")

import numpy as np, torch
from stable_baselines3 import PPO
from pathlib import Path
from sim.pass_policies import route_offense
from train.vec_env import SideVecEnv
from train.policies import policy_for
from train.evaluate import evaluate
from train.run_config import config_from_run
torch.set_num_threads(2)
cfg = config_from_run(Path("experiments/EXP-039-pass/s0"))
env = SideVecEnv(512, cfg, route_offense, side="tackler", seed=0)
pol, kw = policy_for(1.25, 256)
m = PPO(pol, env, n_steps=64, batch_size=8192, n_epochs=5, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, policy_kwargs=kw, device="cpu", verbose=1, seed=0)
ro = lambda sim, obs: route_offense(sim)
for k in range(1):
    det = lambda sim, *a: m.predict(sim.observe_tackler(), deterministic=True)[0]
    r = evaluate(ro, det, cfg, n_plays=2048)
    print(k*0.5, 'M steps: comp', round(r['rate/touchdown']+r['rate/tackled']+r['rate/out_of_bounds'],3), 'yd', round(r['yards/mean'],2), 'td', round(r['rate/touchdown'],3), flush=True)
    m.learn(500_000, reset_num_timesteps=False)
det = lambda sim, *a: m.predict(sim.observe_tackler(), deterministic=True)[0]
r = evaluate(ro, det, cfg, n_plays=2048); print('2M: yd', round(r['yards/mean'],2), 'td', round(r['rate/touchdown'],3))
print(env.pop_stats())

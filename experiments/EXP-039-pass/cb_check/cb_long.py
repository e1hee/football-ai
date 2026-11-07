import sys, dataclasses, numpy as np, torch
from stable_baselines3 import PPO
from pathlib import Path
from sim.pass_policies import route_offense
from train.vec_env import SideVecEnv
from train.policies import policy_for, calibrate_sde
from train.evaluate import evaluate
from train.run_config import config_from_run
torch.set_num_threads(6)
v = sys.argv[1]
cfg = config_from_run(Path("experiments/EXP-039-pass/s0"))
if v == "press":
    cfg = dataclasses.replace(cfg, press_share=1.0)
pol, kw = policy_for(1.25, 256)
if v == "lownoise":
    kw = {**kw, "log_std_init": float(np.log(0.4))}
ppo = dict(n_steps=64, batch_size=8192, n_epochs=5, learning_rate=1e-3 if v == "lr" else 3e-4,
           gamma=0.98 if v == "gamma" else 0.995, gae_lambda=0.95, policy_kwargs=kw, device="cpu", verbose=0, seed=0)
env = SideVecEnv(512, cfg, route_offense, side="tackler", seed=0)
if v.startswith("sde"):
    ppo.update(use_sde=True, sde_sample_freq=40)
m = PPO(pol, env, **ppo)
if v.startswith("sde"):
    calibrate_sde(m.policy, env.sim.observe_tackler(), 0.5 if v == "sde05" else 1.0)
ro = lambda sim, obs: route_offense(sim)
for k in range(6):
    det = lambda sim, *a: m.predict(sim.observe_tackler(), deterministic=True)[0]
    r = evaluate(ro, det, cfg, n_plays=2048)
    print(v, k * 2, "M: yd", round(r["yards/mean"], 2), "comp", round(r["rate/touchdown"] + r["rate/tackled"], 3), flush=True)
    if k < 5:
        m.learn(2_000_000, reset_num_timesteps=False)

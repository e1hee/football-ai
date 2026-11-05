"""Where a late-generation learning phase spends its time. usage: profile_train.py <threads> <steps>"""
import sys, time, json
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
from sim.team_policies import gap_defense, box_offense
from train.run_config import exp_config
from train.selfplay import Frozen, make_pool
from train.vec_env import SideVecEnv
th, steps = int(sys.argv[1]), int(sys.argv[2])
torch.set_num_threads(th)
R = Path("experiments/EXP-024-explore/B"); cfg = exp_config(R, None)
ck = R/"s0/checkpoints"
tacklers = [Frozen(ck/f"tackler_g{g:02d}.zip","tackler") for g in range(0,25)]
pool = make_pool(512, tacklers, gap_defense, 0, scripted_share=0.3, skip_untrained=True, mirror_share=0.5)
venv = SideVecEnv(512, cfg, pool, side="runner", seed=0)
model = PPO.load(str(ck/"runner_g24.zip"), env=venv, device="cpu", n_steps=64, batch_size=8192, n_epochs=5)
T = dict(sim=0.0, opp=0.0, obs=0.0, train=0.0)
def timed(obj, name, key):
    f = getattr(obj, name)
    def g(*a, **k):
        t=time.perf_counter(); r=f(*a,**k); T[key]+=time.perf_counter()-t; return r
    setattr(obj, name, g)
timed(venv.sim,"step","sim"); timed(venv,"opponent","opp"); timed(venv,"_my_obs","obs"); timed(model,"train","train")
t0=time.perf_counter(); model.learn(steps, reset_num_timesteps=True); wall=time.perf_counter()-t0
other=wall-sum(T.values())
print(json.dumps({"threads":th,"sps":round(model.num_timesteps/wall),"wall":round(wall,1),**{k:round(v/wall*100) for k,v in T.items()},"policy_rest":round(other/wall*100)}),flush=True)

import numpy as np, torch
from stable_baselines3 import PPO
from pathlib import Path
from sim.pass_policies import route_offense
from sim.passing import WR, CB
from train.vec_env import SideVecEnv
from train.policies import policy_for
from train.run_config import config_from_run
torch.set_num_threads(2)
cfg = config_from_run(Path("experiments/EXP-039-pass/s0"))

class DistEnv(SideVecEnv):
    def step_wait(self):
        obs, rew, done, infos = super().step_wait()
        d = np.linalg.norm(self.sim.pos[:, WR] - self.sim.pos[:, CB], axis=-1)
        return obs, (-0.05 * d).astype(np.float32), done, infos

env = DistEnv(512, cfg, route_offense, side="tackler", seed=0)
pol, kw = policy_for(1.25, 256)
m = PPO(pol, env, n_steps=64, batch_size=8192, n_epochs=5, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, policy_kwargs=kw, device="cpu", verbose=0, seed=0)
def mean_dist():
    from sim.passing import PassSim
    sim = PassSim(1024, cfg, seed=5); sim.reset(); ds = []
    for t in range(25):
        a = m.predict(sim.observe_tackler(), deterministic=True)[0]
        sim.step(route_offense(sim), a); ds.append(np.linalg.norm(sim.pos[:, WR] - sim.pos[:, CB], axis=-1).mean())
    return round(float(np.mean(ds)), 2)
for k in range(5):
    print(k * 0.4, "M: mean WR-CB distance first 2.5 s", mean_dist(), flush=True)
    if k < 4: m.learn(400_000, reset_num_timesteps=False)

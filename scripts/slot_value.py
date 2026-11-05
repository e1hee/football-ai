"""How much does one lineman's behavior move the only number he is rewarded on?
Replace a slot's action with standing still, mean yards vs the AI's own team, 8192 plays."""
import numpy as np
from pathlib import Path
from sim.team import TeamSim
from sim.team_policies import box_blockers
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config
R = Path("experiments/EXP-024-explore/B"); cfg = exp_config(R, None); N=8192
def play(off, dfn):
    sim=TeamSim(N,cfg,seed=11); obs=sim.reset(); fin=np.zeros(N,bool); y=np.zeros(N)
    while not fin.all():
        obs,_,done,info=sim.step(off(sim,obs),dfn(sim,obs)); new=done&~fin; y[new]=info["yards"][new]; fin|=done
    return y
def off_still(off, slots):
    def f(sim,obs):
        a=np.array(off(sim,obs)).reshape(sim.n,sim.n_off,3); a[:,slots]=0
        return a.reshape(sim.n,-1)
    return f
def off_script_one(off, j):
    def f(sim,obs):
        a=np.array(off(sim,obs)).reshape(sim.n,sim.n_off,3); pos,_,_=sim.perceived("offense")
        a[:,j]=box_blockers(sim,pos,2.5)[:,list(sim.blk).index(j)]
        return a.reshape(sim.n,-1)
    return f
def def_still(dfn, slots):
    def f(sim,obs):
        a=np.array(dfn(sim,obs)).reshape(sim.n,sim.n_def,3); a[:,slots]=0
        return a.reshape(sim.n,-1)
    return f
for s in range(3):
    ck=R/f"s{s}/checkpoints"; o=ppo_runner(ck/"runner_g24.zip"); d=ppo_tackler(ck/"tackler_g24.zip")
    base=play(o,d); m=base.mean(); se=base.std()*2**.5/N**.5
    print(f"s{s} base {m:.2f} play std {base.std():.2f} (se of a difference {se:.2f})")
    print("  OL one still  :",[round(float(play(off_still(o,[j]),d).mean()-m),2) for j in range(1,6)])
    print("  OL one script :",[round(play(off_script_one(o,j),d).mean()-m,2) for j in range(1,6)])
    print("  OL all still  :",round(play(off_still(o,list(range(1,6))),d).mean()-m,2))
    print("  RB still      :",round(play(off_still(o,[0]),d).mean()-m,2))
    print("  DF one still  :",[round(play(o,def_still(d,[j])).mean()-m,2) for j in range(6)], "(DL0-3, LB4-5)")
    print("  DL all still  :",round(play(o,def_still(d,[0,1,2,3])).mean()-m,2), " LB both still:",round(play(o,def_still(d,[4,5])).mean()-m,2),flush=True)

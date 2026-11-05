"""How hard do AI players actually speed up, brake and cut, against what the physics allows?"""
import numpy as np
from pathlib import Path
from sim.team import TeamSim
from sim.physics import _caps
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config
R=Path("experiments/EXP-024-explore/B"); cfg=exp_config(R,None); N=2048; dt=cfg.field.dt
groups={"RB":[0],"OL":[1,2,3,4,5],"DL":[6,7,8,9],"LB":[10,11]}
for s in (0,2):
    ck=R/f"s{s}/checkpoints"; o=ppo_runner(ck/"runner_g24.zip"); d=ppo_tackler(ck/"tackler_g24.zip")
    sim=TeamSim(N,cfg,seed=5); obs=sim.reset(); live=np.ones(N,bool)
    rec={g:dict(sp=[],brk=[],brk_frac=[],fwd=[],fwd_frac=[],lat=[],lat_frac=[],hip=[],push=[],free_brk=[],pair=[]) for g in groups}
    peak={g:np.zeros((N,len(v))) for g,v in groups.items()}; t=0
    while live.any() and t<70:
        off=np.array(o(sim,obs)); dfn=np.array(d(sim,obs))
        pushes=np.concatenate([off.reshape(N,6,3),dfn.reshape(N,6,3)],1)[...,:2]
        v0=sim.vel.copy(); f0=sim.facing.copy(); part0=sim.partner.copy()
        sp0=np.linalg.norm(v0,axis=-1); fc,bc,lc=_caps(sp0,sim.bodies)
        obs,_,done,info=sim.step(off,dfn); t+=1
        v1=sim.vel; ok=live&~done
        dv=(v1-v0)/dt; u=v0/np.maximum(sp0,1e-6)[...,None]; n_=np.stack([-u[...,1],u[...,0]],-1)
        at=(dv*u).sum(-1); al=np.abs((dv*n_).sum(-1)); mov=sp0>1.0
        for g,idx in groups.items():
            sel=np.zeros((N,12),bool); sel[:,idx]=True; sel&=ok[:,None]
            rec[g]["sp"].append(sp0[sel])
            mm=sel&mov
            rec[g]["brk"].append((-at[mm])); rec[g]["brk_frac"].append((-at[mm]/np.maximum(bc[mm],1e-6)))
            rec[g]["fwd_frac"].append(at[mm]/np.maximum(fc[mm],1e-6)); rec[g]["lat"].append(al[mm]); rec[g]["lat_frac"].append(al[mm]/np.maximum(lc[mm],1e-6))
            fm=sel&(sp0>2.0)
            cos=(u*f0).sum(-1)[fm]; rec[g]["hip"].append(cos)
            rec[g]["push"].append(np.linalg.norm(pushes,axis=-1)[sel])
            free=mm&(part0<0)
            rec[g]["free_brk"].append(-at[free]/np.maximum(bc[free],1e-6))
        live&=~done
    print(f"== s{s} ({t} steps max)")
    for g in groups:
        c=lambda k: np.concatenate(rec[g][k])
        print(f"{g}: speed mean {c('sp').mean():.1f} p95 {np.percentile(c('sp'),95):.1f} | "
              f"hard-brake steps (>=50% of cap) free {100*(c('free_brk')>=.5).mean():.1f}% | brake p99 {np.percentile(c('brk'),99):.1f} yd/s2 | "
              f"accel>=50% cap {100*(c('fwd_frac')>=.5).mean():.0f}% | lateral>=50% cap {100*(c('lat_frac')>=.5).mean():.0f}% | "
              f"hips>45deg from motion {100*(c('hip')<np.cos(np.radians(45))).mean():.0f}%, >90 {100*(c('hip')<0).mean():.0f}% | |push| med {np.median(c('push')):.2f}")

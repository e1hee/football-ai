"""Best possible 90 degree cut (steer: push along target velocity minus current), and what the AI does while moving."""
import numpy as np
from pathlib import Path
from sim.physics import apply_controls, Bodies, PROFILES
from sim.team import TeamSim
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config
pf=PROFILES["RB"]; b=Bodies(1,[pf])
for v in (4.,6.,8.):
    pos=np.zeros((1,1,2),np.float32); vel=np.array([[[v,0.]]],np.float32); tgt=np.array([0.,v],np.float32)
    for i in range(60):
        d=tgt-vel[0,0]; a=(d/max(np.linalg.norm(d),1e-6))[None,None].astype(np.float32); apply_controls(pos,vel,a,b,.1)
        if vel[0,0,0]<.15*np.linalg.norm(vel[0,0]): break
    print(f"RB best 90deg cut at {v}: {(i+1)*.1:.1f}s, carried {pos[0,0,0]:.1f} yd, speed {np.linalg.norm(vel[0,0]):.1f}, lowest not tracked")
R=Path("experiments/EXP-024-explore/B"); cfg=exp_config(R,None); N=2048; dt=cfg.field.dt
groups={"RB":[0],"OL":[1,2,3,4,5],"DL":[6,7,8,9],"LB":[10,11]}
for s in (0,2):
    ck=R/f"s{s}/checkpoints"; o=ppo_runner(ck/"runner_g24.zip"); d=ppo_tackler(ck/"tackler_g24.zip")
    sim=TeamSim(N,cfg,seed=5); obs=sim.reset(); live=np.ones(N,bool); t=0
    acc={g:dict(free=[],eng=[],low=[],engsp=[]) for g in groups}
    hist=[]; alive=[]
    while live.any() and t<70:
        off=np.array(o(sim,obs)); dfn=np.array(d(sim,obs))
        p=np.concatenate([off.reshape(N,6,3),dfn.reshape(N,6,3)],1)[...,:2]; pn=np.linalg.norm(p,axis=-1)
        part=sim.partner.copy(); obs,_,done,info=sim.step(off,dfn); t+=1; v=sim.vel.copy(); sp=np.linalg.norm(v,axis=-1); ok=live&~done
        cos=(p*v).sum(-1)/np.maximum(pn*sp,1e-6)
        for g,idx in groups.items():
            sel=np.zeros((N,12),bool); sel[:,idx]=True; sel&=ok[:,None]; mv=sel&(sp>1.0)
            acc[g]["free"].append(cos[mv&(part<0)]); acc[g]["eng"].append(cos[mv&(part>=0)]); acc[g]["low"].append(pn[mv]<.3); acc[g]["engsp"].append(sp[sel&(part>=0)])
        hist.append(v[:,0].copy()); alive.append(ok.copy()); live&=~done
    H=np.stack(hist); A=np.stack(alive); S=np.linalg.norm(H,axis=-1)
    # RB events in the same play: speed drop >=2 yd/s within 0.5 s, heading change >=45 deg within 0.5 s at >=4 yd/s
    k=5; drop=np.zeros(N,bool); cut=np.zeros(N,bool)
    for i in range(k,H.shape[0]):
        ok=A[i]&A[i-k]
        drop|=ok&(S[i-k]-S[i]>=2)
        c=(H[i]*H[i-k]).sum(-1)/np.maximum(S[i]*S[i-k],1e-6)
        cut|=ok&(S[i-k]>=4)&(c<np.cos(np.radians(45)))
    print(f"== s{s} RB plays with speed drop>=2 in 0.5s: {100*drop.mean():.0f}%, heading change>=45deg in 0.5s at 4+ yd/s: {100*cut.mean():.0f}%")
    for g in groups:
        c=lambda q:np.concatenate(acc[g][q]) if any(len(x) for x in acc[g][q]) else np.zeros(1)
        print(f"  {g}: free: moving >45deg away from push {100*(c('free')<np.cos(np.radians(45))).mean():.0f}% | engaged: same {100*(c('eng')<np.cos(np.radians(45))).mean():.0f}%, engaged speed mean {c('engsp').mean():.2f} | push<0.3 while moving {100*c('low').mean():.0f}%")

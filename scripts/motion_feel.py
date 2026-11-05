"""Physics feel: letting go, stopping, and cutting, for a simulated RB-sized body. No policy involved."""
import numpy as np
from sim.physics import apply_controls, Bodies, PROFILES
print("profiles", list(PROFILES))
name="RB" if "RB" in PROFILES else list(PROFILES)[0]
import inspect; print(inspect.signature(Bodies.__init__) if hasattr(Bodies,'__init__') else '')
for nm in ("RB","DL","LB"):
    if nm not in PROFILES: continue
    pf=PROFILES[nm]
    b=Bodies(1,[pf])
    if b is None: print("no ctor"); break
    def run(v0,act,steps=30):
        pos=np.zeros((1,1,2),np.float32);vel=np.array([[v0]],np.float32);out=[]
        for t in range(steps):
            a=act(vel,t); apply_controls(pos,vel,a,b,0.1); out.append((pos[0,0].copy(),vel[0,0].copy()))
        return out
    z=lambda v,t:np.zeros((1,1,2),np.float32)
    g=run([6.,0.],z); print(nm,"let go at 6 yd/s: speed after 1s",round(float(np.linalg.norm(g[9][1])),2),"3s",round(float(np.linalg.norm(g[29][1])),2))
    brk=lambda v,t:(-v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-6)).astype(np.float32)
    for v in (4.,6.,8.):
        s=run([v,0.],brk,40); sp=[np.linalg.norm(x[1]) for x in s]; stop=next((i for i,q in enumerate(sp) if q<.3),None); print(f"  full brake from {v}: stop in {None if stop is None else (stop+1)*.1:.1f}s, {s[stop][0][0]:.1f} yd")
    for v in (4.,6.,8.):
        side=lambda vv,t:np.array([[[0.,1.]]],np.float32)
        s=run([v,0.],side,40); sp=[np.linalg.norm(x[1]) for x in s]; 
        t90=next((i for i,q in enumerate(s) if q[1][0]<.15*np.linalg.norm(q[1])),None)
        print(f"  hard 90deg cut at {v}: heading done in {None if t90 is None else (t90+1)*.1:.1f}s, carried {None if t90 is None else round(float(s[t90][0][0]),1)} yd further forward, speed then {None if t90 is None else round(float(sp[t90]),1)}")

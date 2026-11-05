"""Aggregates only: how much speed real players shed / how far they turn in 0.5 s, vs the best the sim allows."""
import numpy as np, pandas as pd
from analysis.physics_from_bdb import load
from sim.physics import apply_controls, Bodies, PROFILES
df=load(2024)
key=df["game_id"].to_numpy()*10_000_000+df["play_id"].to_numpy()*100_000+df["nfl_id"].to_numpy()%100_000
s=df["s"].to_numpy(); d=df["dir"].to_numpy(); g=df["group"].to_numpy(); t=df["t"].to_numpy()
k=5; ok=(key[k:]==key[:-k])&(np.abs(t[k:]-t[:-k]-.5)<.03)
s0=s[:-k][ok]; drop=(s[:-k]-s[k:])[ok]; turn=np.abs((d[k:]-d[:-k]+180)%360-180)[ok]; gg=g[:-k][ok]; send=s[k:][ok]
# Sim: best speed shed and best heading change in 0.5 s from a given speed
def sim_best(pf,v):
    b=Bodies(1,[pf]); pos=np.zeros((1,1,2),np.float32); vel=np.array([[[v,0.]]],np.float32)
    for i in range(5):
        a=(-vel/np.maximum(np.linalg.norm(vel,axis=-1,keepdims=True),1e-6)).astype(np.float32); apply_controls(pos,vel,a,b,.1)
    dr=v-float(np.linalg.norm(vel))
    best=0.
    for ang in np.radians(np.arange(60,180,5)):
        pos=np.zeros((1,1,2),np.float32); vel=np.array([[[v,0.]]],np.float32); a=np.array([[[np.cos(ang),np.sin(ang)]]],np.float32)
        for i in range(5): apply_controls(pos,vel,a,b,.1)
        best=max(best,float(np.degrees(np.arctan2(vel[0,0,1],vel[0,0,0]))))
    return dr,best
for grp,prof in (("WR","RB"),("LB","LB"),("DL","DL"),("OL","OL")):
    for lo,hi in ((3,5),(5,7),(7,9)):
        m=(gg==grp)&(s0>=lo)&(s0<hi)
        if m.sum()<1000: continue
        q=lambda a,p:round(float(np.quantile(a,p)),1)
        mt=m&(send>=lo*0.6)      # still running: a real cut, not a stop
        dr,tu=sim_best(PROFILES[prof],(lo+hi)/2)
        print(f"{grp} {lo}-{hi} yd/s n={m.sum()}: NFL speed shed in 0.5s p99 {q(drop[m],.99)} p99.9 {q(drop[m],.999)} | sim best {dr:.1f} ({prof}) || NFL heading change in 0.5s p99 {q(turn[mt],.99)} p99.9 {q(turn[mt],.999)} deg | sim best {tu:.0f} deg")

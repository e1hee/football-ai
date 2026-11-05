from pathlib import Path
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config
from scripts.lane_check import paid_and_place
X=Path('experiments/EXP-025-roles'); O=Path('experiments/EXP-026-lane/continue'); cfg=exp_config(X,None)
for s in range(3):
    d=ppo_tackler(X/f"s{s}/checkpoints/tackler_g12.zip")
    r=paid_and_place(ppo_runner(O/f"checkpoints/exp025_s{s}.zip"),d,cfg,1024)
    print('EXP-025',s,'continued 2M vs its g12 defense',f"box {r['box']*100:.1f}% reach {r['reach']*100:.0f}% yd {r['yards']:.2f} box_yd {r['yards_box']:.2f} out_yd {r['yards_out']:.2f}",flush=True)

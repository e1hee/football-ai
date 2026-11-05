from pathlib import Path
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config
from scripts.lane_check import paid_and_place
for X,O,g in ((Path('experiments/EXP-025-roles'),Path('experiments/EXP-025-roles/results/g12'),12),(Path('experiments/EXP-024-explore/B'),Path('experiments/EXP-024-explore/results_B_g24'),24)):
    cfg=exp_config(X,None)
    for s in range(3):
        d=ppo_tackler(X/f"s{s}/checkpoints/tackler_g{g:02d}.zip")
        for name,ck in (("trained",X/f"s{s}/checkpoints/runner_g{g:02d}.zip"),("fresh",O/f"checkpoints/fresh_r_s{s}.zip")):
            r=paid_and_place(ppo_runner(ck),d,cfg,1024)
            print(X,s,name,f"box {r['box']*100:.1f}% reach {r['reach']*100:.0f}% yd {r['yards']:.2f} box_yd {r['yards_box']:.2f} out_yd {r['yards_out']:.2f}",flush=True)

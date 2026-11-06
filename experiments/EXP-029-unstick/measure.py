"""Where each saved offense crosses the line, against the defense it trained on (EXP-029).

    PYTHONPATH=. uv run python experiments/EXP-029-unstick/measure.py
"""
import json
from pathlib import Path

from scripts.lane_check import paid_and_place
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config

X = Path("experiments/EXP-028-older/A")
O = Path("experiments/EXP-029-unstick")
VARIANTS = ("warm", "noise", "head", "fresh")


def main() -> None:
    cfg = exp_config(X, None)
    rows = []
    for s in range(3):
        d = ppo_tackler(X / f"s{s}/checkpoints/tackler_g12.zip")
        for v in VARIANTS:
            cks = sorted(O.glob(f"checkpoints/{v}_s{s}_*k.zip"), key=lambda p: int(p.stem.rsplit("_", 1)[1][:-1]))
            pts = ([(0, X / f"s{s}/checkpoints/runner_g12.zip")] if v == "warm" else []) + [(int(p.stem.rsplit("_", 1)[1][:-1]) * 1000, p) for p in cks]
            for steps, path in pts:
                r = paid_and_place(ppo_runner(path), d, cfg, 2048)
                row = {"variant": v, "seed": s, "steps": steps, **{k: r[k] for k in ("box", "reach", "yards", "yards_box", "yards_out")}}
                rows.append(row)
                print(f"s{s} {v:5} {steps/1e6:4.1f}M box {r['box']*100:5.1f}% reach {r['reach']*100:3.0f}% "
                      f"yd {r['yards']:5.2f} (box {r['yards_box']:5.2f}, out {r['yards_out']:5.2f})", flush=True)
    (O / "measure.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

"""Hold one side fixed: did the offense change, or did the defense it played against?

Same-seed pairs at one generation, new offense/defense against the old ones and the scripted ones.

    uv run python -m scripts.role_cross experiments/EXP-025-roles experiments/EXP-024-explore/B --gen 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.role_report import report
from sim.team_policies import box_offense, gap_defense
from train.evaluate import ppo_runner, ppo_tackler
from train.run_config import exp_config

KEYS = ["yards", "td", "backfield_contact", "reblock_1s", "ol_free_run", "lb_early", "inside", "reach_line", "lane_ahead"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("new", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument("--gen", type=int, default=12)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    cfg = exp_config(args.new, None)
    script_off, script_def = (lambda s, o=None: box_offense(s)), (lambda s, o=None: gap_defense(s))
    out: dict = {}
    for seed in ("s0", "s1", "s2"):
        ck = lambda exp, side: exp / seed / "checkpoints" / f"{side}_g{args.gen:02d}.zip"
        r_new, t_new = ppo_runner(ck(args.new, "runner")), ppo_tackler(ck(args.new, "tackler"))
        r_old, t_old = ppo_runner(ck(args.old, "runner")), ppo_tackler(ck(args.old, "tackler"))
        pairs = {"old off / old def": (r_old, t_old), "new off / old def": (r_new, t_old),
                 "old off / new def": (r_old, t_new), "new off / new def": (r_new, t_new),
                 "old off / script def": (r_old, script_def), "new off / script def": (r_new, script_def),
                 "script off / old def": (script_off, t_old), "script off / new def": (script_off, t_new)}
        for name, (o, d) in pairs.items():
            out.setdefault(name, {})[seed] = report(o, d, cfg, args.n)
    print(f"{'':24}" + "".join(f"{k[:10]:>11}" for k in KEYS))
    for name, rows in out.items():
        print(f"{name:24}" + "".join(f"{sum(r[k] for r in rows.values()) / 3:11.2f}" for k in KEYS), flush=True)
    print("\nper seed (s0 / s1 / s2)")
    for name, rows in out.items():
        print(f"{name:24} yards " + " / ".join(f"{rows[s]['yards']:.2f}" for s in rows)
              + "  contact " + " / ".join(f"{rows[s]['backfield_contact']:.2f}" for s in rows)
              + "  reblock " + " / ".join(f"{rows[s]['reblock_1s']:.2f}" for s in rows))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

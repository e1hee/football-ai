"""Plot an experiment's per-generation evaluation as a video-ready chart.

    uv run python -m scripts.plot_curves experiments/EXP-002-ppo-runner-v1 \
        --baseline 회피=0.382 --baseline 직진=0.288
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()

COLORS = {"rate/touchdown": "#3fa34d", "rate/tackled": "#d64545",
          "rate/out_of_bounds": "#e0a526", "rate/timeout": "#8a8a8a"}
NAMES = {"rate/touchdown": "터치다운", "rate/tackled": "태클", "rate/out_of_bounds": "아웃", "rate/timeout": "시간 초과"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp", type=Path)
    p.add_argument("--baseline", action="append", default=[], help="name=touchdown_rate, drawn as dashed line")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.exp / "metrics.csv")
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)   # 16:9
    fig.patch.set_facecolor("#18181f")
    ax.set_facecolor("#18181f")
    for k, c in COLORS.items():
        if k in df:
            ax.plot(df["gen"], df[k] * 100, color=c, lw=3, marker="o", ms=4, label=NAMES[k])
    for b in args.baseline:
        name, v = b.split("=")
        ax.axhline(float(v) * 100, color="#9db4ff", ls="--", lw=1.5)
        ax.text(df["gen"].max(), float(v) * 100 + 1.5, f"손으로 짠 {name} 러너 {float(v):.0%}",
                color="#9db4ff", ha="right", fontsize=11)
    ax.set_xlabel("세대", color="white", fontsize=14)
    ax.set_ylabel("플레이 비율 (%)", color="white", fontsize=14)
    ax.set_ylim(-2, 102)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#555")
    ax.grid(color="#333", lw=0.6)
    ax.legend(facecolor="#22222b", edgecolor="#444", labelcolor="white", fontsize=12, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(args.exp.name, color="white", fontsize=15, loc="left")
    out = args.out or args.exp / "curves.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"-> {out}")


if __name__ == "__main__":
    main()

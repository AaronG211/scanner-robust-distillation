#!/usr/bin/env python3
"""Figure (exploratory): the layer ladder for four teachers. Input: data/pc_runs_2026-11-17/layer_ladder.csv
(scripts/layer_ladder.py). Output: figures/ladder_multi_teacher.png. New-scanner F1, single view, mean over folds and
seeds, 95% bootstrap intervals over images and seeds."""
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
d = pd.read_csv(R / "data" / "pc_runs_2026-11-17" / "layer_ladder.csv")
d = d[d.fold == "mean"]
STEPS = [("S0 KD", "KD only"), ("S1 +TGR", "+ TGR\n(cutoff)"), ("AdaBN only", "AdaBN\n(BN stats)"),
         ("S2 +AdaBN+TGR", "AdaBN\n+ TGR"), ("S3 KD+TGA", "KD + TGA\n(retrain)"), ("S4 teacher", "Teacher")]
TEACH = [("hopt0", "H-optimus-0", "#0072B2", "o"), ("virchow2", "Virchow2", "#E69F00", "s"),
         ("uni2h", "UNI2-h", "#009E73", "^"), ("uni", "UNI", "#D55E00", "D")]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
off = [-0.15, -0.05, 0.05, 0.15]
for (fm, name, col, mk), o in zip(TEACH, off):
    g = d[d.teacher == fm].set_index("step")
    y = [g.loc[s, "f1"] for s, _ in STEPS]; lo = [g.loc[s, "lo"] for s, _ in STEPS]; hi = [g.loc[s, "hi"] for s, _ in STEPS]
    x = [i + o for i in range(len(STEPS))]
    ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], fmt=mk, color=col,
                ms=6, lw=1.2, capsize=0, label=name, mec="white", mew=0.8)
ax.axhline(0.711, color="0.55", lw=1, ls=":", zorder=0)
ax.text(len(STEPS) - 0.5, 0.711, "no teacher 0.711", ha="right", va="bottom", fontsize=7.5, color="0.4")
ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([l for _, l in STEPS], fontsize=8)
ax.set_ylabel("New-scanner F1 (ResNet-18 student)", fontsize=9); ax.set_ylim(0.66, 0.84)
ax.tick_params(axis="y", labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.1))
ax.set_title("Where the student-teacher gap goes, four teachers (exploratory)", fontsize=9.5, loc="left", pad=22)
fig.tight_layout(); out = R / "figures" / "ladder_multi_teacher.png"; fig.savefig(out); print(out)

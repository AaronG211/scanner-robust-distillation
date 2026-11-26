"""Figures for the Demo II slides (ResNet add-ons).

Reads results/ablation5_summary.csv and results/ablation5_confirmatory.csv and writes
figures/demo2_per_scanner.png, figures/demo2_modes.png, figures/demo2_components.png.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

INK, MUTED, GRID = "#24313C", "#5F6B78", "#E3E7EB"
BLUE, ORANGE, GRAY = "#2D6CDF", "#D9822B", "#B8C0C8"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

S = pd.read_csv(RES / "ablation5_summary.csv")


def row(fold, tag, mode):
    r = S[(S.fold == fold) & (S.tag == tag) & (S["mode"] == mode)]
    assert len(r) == 1, (fold, tag, mode)
    return r.iloc[0]


def per_scanner():
    folds = ["CS2", "S360", "XR", "mean"]
    names = ["CS2", "S360", "XR", "Mean"]
    arms = [("baseline", "plain", "Baseline, as trained", GRAY),
            ("all", "adabncf_tta", "ResNet all, AdaBN + TTA", BLUE)]
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=220)
    w = 0.36
    for k, (tag, mode, label, color) in enumerate(arms):
        for i, f in enumerate(folds):
            r = row(f, tag, mode)
            x = i + (k - 0.5) * w
            ax.bar(x, r.f1_mean, w * 0.92, color=color, label=label if i == 0 else None, zorder=2)
            ax.plot([x, x], [r.f1_ci_lo, r.f1_ci_hi], color=INK, lw=1, zorder=3)
            ax.text(x, r.f1_ci_hi + 0.008, f"{r.f1_mean:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(folds)), names)
    ax.set_ylim(0.45, 0.86)
    ax.set_ylabel("F1 on the unseen scanner")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_title("Per held-out scanner (5 seeds, 95% CI)", fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(FIG / "demo2_per_scanner.png", facecolor="white")
    plt.close(fig)


def modes():
    tags = [("baseline", "Baseline", ORANGE, 2.2), ("stain", "+ Stain", GRAY, 1.4),
            ("adv", "+ GRL", GRAY, 1.4), ("stain_adv", "+ Stain + GRL", GRAY, 1.4),
            ("all", "ResNet all", BLUE, 2.6)]
    ms = [("plain", "As trained"), ("tta", "+ TTA"), ("adabncf_tta", "+ AdaBN\n+ TTA")]
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=220)
    ends = []
    for tag, label, color, lw in tags:
        y = [row("mean", tag, m).f1_mean for m, _ in ms]
        ax.plot(range(3), y, color=color, lw=lw, marker="o", ms=5 if lw > 2 else 3.5,
                zorder=3 if lw > 2 else 2)
        ends.append((y[0], label, color, lw))
    for y0, label, color, lw in ends:
        if label == "+ Stain":
            continue  # same start as + Stain + GRL (0.713 vs 0.714): one shared label
        if label == "+ Stain + GRL":
            label = "+ Stain, + Stain + GRL"
        ax.text(-0.08, y0, label, ha="right", va="center", fontsize=8.5,
                color=INK if lw > 2 else MUTED, fontweight="bold" if lw > 2 else "normal")
    b0, a0 = row("mean", "baseline", "plain").f1_mean, row("mean", "all", "plain").f1_mean
    b2, a2 = row("mean", "baseline", "adabncf_tta").f1_mean, row("mean", "all", "adabncf_tta").f1_mean
    for x, lo, hi in [(-0.0, b0, a0), (2.0, b2, a2)]:
        xb = x + 0.11
        ax.annotate("", xy=(xb, hi), xytext=(xb, lo),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1, shrinkA=0, shrinkB=0))
        ax.text(xb + 0.04, (lo + hi) / 2, f"{round(hi, 3) - round(lo, 3):+.3f}", va="center", fontsize=9.5,
                fontweight="bold", color=INK)
    ax.set_xticks(range(3), [m[1] for m in ms])
    ax.set_xlim(-1.05, 2.45)
    ax.set_ylim(0.655, 0.795)
    ax.set_ylabel("F1 (mean of 3 scanners × 5 seeds)")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_title("Add-on gains shrink after test-time adaptation", fontsize=11, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(FIG / "demo2_modes.png", facecolor="white")
    plt.close(fig)


def components():
    C = pd.read_csv(RES / "ablation5_confirmatory.csv")
    short = {
        "Stain augmentation": "Stain augmentation",
        "TTA": "TTA",
        "Cross-fitted AdaBN": "Cross-fitted AdaBN",
        "GRL scanner head": "GRL head",
        "Soft labels": "Soft labels",
        "Headline": "ResNet all vs baseline (as trained)",
    }
    order = ["Stain augmentation", "TTA", "Cross-fitted AdaBN", "GRL scanner head", "Soft labels", "Headline"]
    rows = []
    for key in order:
        r = C[C.component.str.startswith(key)]
        assert len(r) == 1, key
        rows.append(r.iloc[0])
    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=220)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        sig = bool(r.significant_holm)
        color = BLUE if sig else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r["diff"]], [y], s=55, color=color if sig else "white", edgecolor=color, lw=1.6, zorder=3)
        p = f"< {math.ceil(r.p_holm * 1000) / 1000:.3f}" if r.p_floor else f"{r.p_holm:.3f}"
        ax.text(r.ci_hi + 0.004, y, f"{r['diff']:+.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]   Holm p {p}",
                va="center", fontsize=8.8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_yticks(range(len(rows)), [short[k] for k in order][::-1])
    for t, r in zip(ax.get_yticklabels(), rows[::-1]):
        t.set_color(INK)
        if r.component.startswith("Headline"):
            t.set_fontweight("bold")
    ax.set_xlim(-0.015, 0.215)
    ax.set_ylim(-0.6, len(rows) - 0.5)
    ax.set_xlabel("ΔF1 when the component is added back (95% CI)")
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.0, -0.33, "filled = significant after Holm correction (6 tests)", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "demo2_components.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    per_scanner()
    modes()
    components()
    print("wrote", *[p.name for p in sorted(FIG.glob("demo2_*.png"))])

"""Figures for the cutoff-transfer study (reads results/cutoff_*.csv; writes figures/cutoff_*.png)."""
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
INK, MUTED, GRID_C = "#24313C", "#5F6B78", "#E3E7EB"
BLUE, ORANGE, GRAY = "#2D6CDF", "#D9822B", "#B8C0C8"
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def forest():
    c = pd.read_csv(RES / "cutoff_confirmatory.csv")
    labels = {"H1": "H1  PRM − SRC · non-stain models, as trained",
              "H2": "H2  PRM − SRC · stain models, as trained",
              "H3": "H3  PRM − SRC · on top of AdaBN-cf",
              "H4": "H4  PRM − SRC · Axel's models (no BatchNorm)",
              "H5": "H5  label-shift EM − SRC · grid, as trained",
              "H6": "H6  (as trained + PRM) − (AdaBN-cf + SRC)"}
    fig, ax = plt.subplots(figsize=(9.4, 3.3), dpi=220)
    for i, r in enumerate(c.itertuples()):
        y = len(c) - 1 - i; col = BLUE if r.significant_holm else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.scatter([r.dF1], [y], s=50, color=col if r.significant_holm else "white", edgecolor=col, lw=1.6, zorder=3)
        p = f"< {math.ceil(r.p_holm * 1000) / 1000:.3f}" if r.p_floor else f"{r.p_holm:.3f}"
        ax.annotate(f"{r.dF1:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]   Holm p {p}", xy=(1.02, y),
                    xycoords=("axes fraction", "data"), va="center", fontsize=8.3, annotation_clip=False)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(range(len(c)), [labels[h] for h in c.id][::-1], fontsize=8.6)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.23, 0.075); ax.set_ylim(-0.6, len(c) - 0.4)
    ax.set_xlabel("ΔF1 on the new scanner (95% CI, paired two-level bootstrap)")
    ax.xaxis.grid(True, color=GRID_C); ax.set_axisbelow(True)
    ax.set_title("Confirmatory family A (Holm over 6; filled = significant)", fontsize=10, loc="left")
    fig.tight_layout(); fig.subplots_adjust(right=0.70)
    fig.savefig(FIG / "cutoff_confirmatory.png", facecolor="white"); plt.close(fig)


def decomposition():
    d = pd.read_csv(RES / "cutoff_decomposition.csv").set_index("tag")
    tags = ["baseline", "adv", "stain", "stain_adv", "all", "all_no_grl", "all_no_soft", "all_no_stain", "all_stain_default"]
    names = {"baseline": "Baseline", "adv": "+ GRL", "stain": "+ Stain", "stain_adv": "+ Stain + GRL", "all": "ResNet all",
             "all_no_grl": "all − GRL", "all_no_soft": "all − soft labels", "all_no_stain": "all − stain",
             "all_stain_default": "all, default stain"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=220)
    y = np.arange(len(tags))[::-1]; h = 0.36
    cut, rank = d.loc[tags, "cutoff"].values, d.loc[tags, "ranking"].values
    ax.barh(y + h / 2, cut, h, color=BLUE, label="cutoff relocation")
    ax.barh(y - h / 2, rank, h, color=GRAY, label="better ranking (F1 at each mode's best cutoff)")
    for yi, c_, r_, g in zip(y, cut, rank, d.loc[tags, "gain"].values):
        ax.text(max(c_, 0) + 0.002, yi + h / 2, f"{c_:+.3f}", va="center", fontsize=7.5)
        ax.text(max(r_, 0) + 0.002, yi - h / 2, f"{r_:+.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, [names[t] for t in tags]); ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=MUTED, lw=0.9); ax.set_xlim(-0.01, 0.07)
    ax.set_xlabel("Part of AdaBN-cf's F1 gain over 'as trained' (mean of 3 folds × 5 seeds)")
    ax.xaxis.grid(True, color=GRID_C); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Most of AdaBN's gain is a cutoff shift, not better ranking", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "cutoff_adabn_decomposition.png", facecolor="white"); plt.close(fig)


def mechanism():
    m = pd.read_csv(RES / "cutoff_mechanism_runs.csv")
    m = m[m["mode"] == "plain"]
    rows = []
    for src, lab in (("ours", "ResNet-18 grid"), ("axel", "Axel's models")):
        for f in ("CS2", "S360", "XR"):
            x = m[(m.source == src) & (m.fold == f)].assign(dP=lambda z: z.F1_PRM - z.F1_SRC, dO=lambda z: z["F1_PRM-op"] - z.F1_SRC)
            x = x.groupby("name")[["dP", "dO", "prev_source_val", "prev_new_true"]].mean().mean()   # models weighted equally
            rows.append((f"{lab} · {f}", x.dP, x.dO, x.prev_source_val, x.prev_new_true))
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=220)
    for i, (lab, prm, op, ps, pt) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([prm, op], [y, y], color=GRID_C, lw=3, zorder=1)
        ax.scatter([prm], [y], s=45, color=ORANGE, zorder=3, label="PRM (label-free)" if i == 0 else None)
        ax.scatter([op], [y], s=45, color=BLUE, zorder=3, label="PRM given the true prevalence" if i == 0 else None)
        ax.text(0.068, y, f"mitoses: source {ps:.0%} → new {pt:.0%}", va="center", fontsize=7.8, color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False); ax.set_xlim(-0.04, 0.066)
    ax.set_xlabel("ΔF1 vs source cutoff (as trained; models weighted equally)")
    ax.xaxis.grid(True, color=GRID_C); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.tight_layout(); fig.subplots_adjust(right=0.74)
    fig.savefig(FIG / "cutoff_mechanism.png", facecolor="white"); plt.close(fig)


def audit():
    a = pd.read_csv(RES / "cutoff_audit.csv")
    x = a[(a.source == "ours") & (a["mode"] == "plain")].groupby("k")[["dF1_DIRECT_k", "dF1_PRM_k", "dF1_PRM_labelfree"]].mean()
    fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=220)
    ax.plot(x.index, x.dF1_PRM_k, color=BLUE, lw=2, marker="o", ms=5)
    ax.plot(x.index, x.dF1_DIRECT_k, color=ORANGE, lw=2, marker="o", ms=5)
    ax.axhline(x.dF1_PRM_labelfree.iloc[0], color=MUTED, lw=1, ls="--")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.text(205, x.dF1_PRM_k.iloc[-1], "  prevalence from k labels + PRM", color=BLUE, va="center", fontsize=8.3)
    ax.text(205, x.dF1_DIRECT_k.iloc[-1] - 0.004, "  cutoff picked on k labels", color=ORANGE, va="center", fontsize=8.3)
    ax.text(22, x.dF1_PRM_labelfree.iloc[0] + 0.002, "label-free PRM", color=MUTED, fontsize=8)
    ax.set_xscale("log"); ax.set_xticks([20, 50, 100, 200], ["20", "50", "100", "200"]); ax.set_xlim(17, 520)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("labelled new-scanner cells per estimation half (k)")
    ax.set_ylabel("ΔF1 vs source cutoff")
    ax.yaxis.grid(True, color=GRID_C); ax.set_axisbelow(True)
    ax.set_title("A small labelled audit: spend it on the prevalence", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "cutoff_audit.png", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    forest(); decomposition(); mechanism(); audit()
    print("wrote", sorted(p.name for p in FIG.glob("cutoff_*.png")))

#!/usr/bin/env python3
"""Draft paper figures (2026-11-25). Reads only result files in data/pc_runs_2026-11-17/; writes figures/paper/fig{1..4}.{pdf,png}.

  Fig 1  schematic: teacher, student, the three layers and what repairing each needs
  Fig 2  where the student-teacher gap sits: share of the teacher's advantage per layer (ladder_recovered.csv)
  Fig 3  the threshold layer: few-image budget (e02_*budget*.csv) and simulated prevalence shift (recal_baselines_table.csv)
  Fig 4  how close and how cheap: student-minus-teacher forest plot (noninferiority_*.csv) and throughput (efficiency.csv)

Colour = layer throughout (Okabe-Ito; colour-blind separation checked with an OKLab/Machado simulation):
threshold #0072B2, BN statistics #009E73, retraining #E69F00, distillation #C9C9C9.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "pc_runs_2026-11-17"
OUT = ROOT / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
C_THR, C_BN, C_RET, C_DIS = "#0072B2", "#009E73", "#E69F00", "#C9C9C9"
C_SRC, C_LS, INK, MUTED = "#D55E00", "#CC79A7", "#222222", "#6B6B6B"
NAME = {"hopt0": "H-optimus-0", "virchow2": "Virchow2", "uni": "UNI", "uni2h": "UNI2-h"}
plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
                     "axes.labelcolor": INK, "xtick.color": "#444444", "ytick.color": "#444444", "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})

def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E6E6E6", lw=0.5, zorder=0); ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", OUT / f"{name}.pdf")

# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(6.0, 2.35)); ax.set_xlim(0, 100); ax.set_ylim(0, 41); ax.axis("off")
    def box(x, y, w, h, fc, ec, lw=0.8, r=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    def head(x, t):
        ax.text(x, 40.2, t, fontsize=7.2, color=MUTED, va="top", weight="bold")
    head(0.5, "Source scanners (labelled)"); head(29, "Unseen scanner: what is lost"); head(60, "Repair: what it needs")
    box(0.5, 24, 23.5, 12.5, "#F2F2F2", "#888888")
    ax.text(12.25, 33.4, "Foundation-model\nteacher", ha="center", va="center", fontsize=7.6, weight="bold", color=INK, linespacing=1.0)
    ax.text(12.25, 28.6, "ViT + LoRA", ha="center", fontsize=6.6, color=INK)
    ax.text(12.25, 25.8, "0.3 to 1.2 B params", ha="center", fontsize=6.6, color=MUTED)
    box(0.5, 2.5, 23.5, 12.5, "white", INK)
    ax.text(12.25, 11.8, "CNN student", ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
    ax.text(12.25, 7.8, "ResNet-18 / ConvNeXt-T", ha="center", fontsize=6.6, color=INK)
    ax.text(12.25, 4.6, "11 to 28 M params", ha="center", fontsize=6.6, color=MUTED)
    ax.annotate("", xy=(12.25, 15.3), xytext=(12.25, 23.7), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    ax.text(13.3, 19.5, "distillation\n(soft labels)", fontsize=6.6, color=INK, va="center", linespacing=1.0)
    ax.annotate("", xy=(28.6, 8.75), xytext=(24.2, 8.75), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
    layers = [(26.2, C_THR, "Operating point", "decision threshold"),
              (14.4, C_BN, "Normalisation statistics", "BatchNorm statistics"),
              (2.6, C_RET, "Representation", "network weights")]
    for y, col, t1, t2 in layers:
        box(29, y, 26, 9.6, "white", col, lw=1.3)
        ax.add_patch(Rectangle((29, y), 1.5, 9.6, fc=col, ec="none"))
        ax.text(32, y + 6.2, t1, fontsize=7.6, weight="bold", color=INK, va="center")
        ax.text(32, y + 3.0, t2, fontsize=6.6, color=MUTED, va="center")
    fixes = [(26.2, C_THR, "TGR: one logit bias", "~5 unlabelled images + teacher", "no retraining"),
             (14.4, C_BN, "AdaBN (+ TENT)", "unlabelled images, no teacher", "no retraining"),
             (2.6, C_RET, "TGA: adapt while distilling", "unlabelled images; teacher optional", "retraining")]
    for y, col, t1, t2, t3 in fixes:
        ax.annotate("", xy=(59.6, y + 4.8), xytext=(55.3, y + 4.8), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1))
        box(60, y, 31, 9.6, "#FAFAFA", col, lw=0.8)
        ax.text(61.5, y + 7.0, t1, fontsize=7.4, weight="bold", color=INK, va="center")
        ax.text(61.5, y + 4.3, t2, fontsize=6.6, color=INK, va="center")
        ax.text(61.5, y + 1.8, t3, fontsize=6.4, color=MUTED, va="center", style="italic")
    ax.annotate("", xy=(95, 3.5), xytext=(95, 35), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(97.2, 19.3, "cost of repair", rotation=-90, fontsize=6.8, color=MUTED, va="center", ha="center")
    save(fig, "fig1_schematic")

# ---------------------------------------------------------------- Fig 2
def fig2():
    L = pd.read_csv(D / "ladder_recovered.csv")
    def q(st, fm, kind, name, col="value"):
        return L[(L.student == st) & (L.teacher == fm) & (L.kind == kind) & (L.quantity == name)][col].iloc[0]
    rows = [("resnet18", fm) for fm in ("hopt0", "virchow2", "uni2h")] + [("convnext_t", fm) for fm in ("hopt0", "virchow2")]
    fig, ax = plt.subplots(figsize=(6.0, 2.45))
    ypos = [4.6, 3.6, 2.6, 0.8, -0.2]
    for (st, fm), yy in zip(rows, ypos):
        if st == "resnet18":
            comps = [("distillation (kd - ctrl)", C_DIS), ("threshold, Shapley", C_THR), ("BN statistics, Shapley", C_BN),
                     ("retraining (kd_tga - kd_adabn_tgr)", C_RET)]
        else:
            comps = [("distillation (kd - ctrl)", C_DIS), ("threshold (kd_tgr - kd)", C_THR), ("retraining (kd_tga - kd_tgr)", C_RET)]
        c = 0.0
        for name, col in comps:
            v = 100 * q(st, fm, "component_share", name)
            left, w = (c, v) if v >= 0 else (c + v, -v)
            if v < 0:                                # a negative layer: thin hatched bar under the row, labelled
                neg = "#7A7A7A" if col == C_DIS else col
                ax.barh(yy - 0.42, w, left=left, height=0.13, color="white", edgecolor=neg, lw=0.7, hatch="//////", zorder=3)
                ax.text(left - 0.8, yy - 0.42, f"{name.split(' ')[0]} {v:+.0f}%", ha="right", va="center", fontsize=6.2, color=MUTED)
                c += v; continue
            ax.barh(yy, w, left=left, height=0.62, color=col, edgecolor="white", lw=1.0, zorder=3)
            if abs(v) >= 9:
                ax.text(left + w / 2, yy, f"{v:+.0f}%" if v < 0 else f"{v:.0f}%", ha="center", va="center", fontsize=6.8,
                        color="white" if col in (C_THR, C_BN) and v >= 0 else INK, zorder=4)
            c += v
        e, lo, hi = (100 * q(st, fm, "share", "kd_tga", k) for k in ("value", "lo", "hi"))
        ax.errorbar(e, yy + 0.43, xerr=[[e - lo], [hi - e]], fmt="v", ms=3.2, color=INK, lw=0.7, capsize=1.8, zorder=5)
        if c < 100:
            ax.barh(yy, 100 - c, left=c, height=0.62, color="white", edgecolor="#9A9A9A", lw=0.7, ls=(0, (2, 1.5)), zorder=2)
        adv = q(st, fm, "f1_diff", "teacher advantage (teacher - ctrl)")
        ax.text(141, yy, f"{adv:.3f}", ha="right", va="center", fontsize=7, color=INK)
        ax.text(-17, yy, NAME[fm], ha="right", va="center", fontsize=7.5, color=INK)
    ax.text(-17, 5.45, "ResNet-18 student", ha="right", va="center", fontsize=7.5, weight="bold", color=INK)
    ax.text(-17, 1.6, "ConvNeXt-T student", ha="right", va="center", fontsize=7.5, weight="bold", color=INK)
    ax.text(141, 5.45, "teacher's\nadvantage (F1)", ha="right", va="center", fontsize=6.5, color=MUTED, linespacing=0.95)
    for x in (0, 100):
        ax.axvline(x, color="#555555", lw=0.6, zorder=1)
    ax.text(1.2, 5.45, "= no-teacher student", ha="left", va="center", fontsize=6.4, color=MUTED)
    ax.text(98.8, 5.45, "teacher =", ha="right", va="center", fontsize=6.4, color=MUTED)
    ax.set_xlim(-15, 142); ax.set_ylim(-0.75, 5.85); ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120]); ax.set_xticklabels([f"{t}%" for t in (0, 20, 40, 60, 80, 100, 120)])
    ax.set_xlabel("share of the teacher's advantage on the unseen scanner (F1)")
    tidy(ax, grid="x"); ax.spines["left"].set_visible(False)
    hd = [Rectangle((0, 0), 1, 1, fc=C_DIS), Rectangle((0, 0), 1, 1, fc=C_THR), Rectangle((0, 0), 1, 1, fc=C_BN),
          Rectangle((0, 0), 1, 1, fc=C_RET), Rectangle((0, 0), 1, 1, fc="white", ec="#9A9A9A", ls=(0, (2, 1.5))),
          Line2D([], [], marker="v", ls="", color=INK, ms=3.5)]
    ax.legend(hd, ["distillation", "threshold (TGR)", "BN statistics (AdaBN)", "retraining (TGA)", "remaining gap",
                   "retrained student, 95% CI"], ncol=6, fontsize=6.6, loc="lower center", bbox_to_anchor=(0.43, 1.0),
              handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
    save(fig, "fig2_where_the_gap_sits")

# ---------------------------------------------------------------- Fig 3
def fig3():
    lab = pd.read_csv(D / "e02_labelled_budget.csv"); ada = pd.read_csv(D / "e02_adabn_budget.csv")
    tab = pd.read_csv(D / "recal_baselines_table.csv").set_index(["student", "pi", "method"])
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.6))
    ns = [1, 2, 5, 10, 25]
    def curve(df, arm, col):
        g = df[df.arm == arm]
        per = g.groupby(["n_images", "draw", "holdout"])[col].mean().groupby(["n_images", "draw"]).mean()
        m = per.groupby("n_images").mean().reindex(ns); lo = per.groupby("n_images").min().reindex(ns); hi = per.groupby("n_images").max().reindex(ns)
        return m.values, lo.values, hi.values
    for ax, arm, title in ((axes[0, 0], "r18_kd", "a  ResNet-18: few-image budget"), (axes[0, 1], "cnx_kd", "b  ConvNeXt-T: few-image budget")):
        base = lab[lab.arm == arm].groupby("holdout").f1_none.mean().mean()
        ax.axhline(base, color="#8A8A8A", lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.text(27, base + 0.002, "source cutoff, no adaptation", fontsize=6.3, color=MUTED, va="bottom", ha="right")
        series = [(lab, "f1_tgr", C_THR, "^", "TGR (n unlabelled images + teacher)"),
                  (lab, "f1_labelled", INK, "o", "labelled cutoff (n labelled images)")]
        if arm == "r18_kd":
            series.insert(1, (ada, "f1_adabn", C_BN, "s", "AdaBN (n unlabelled images)"))
        for df, col, c, mk, lbl in series:
            m, lo, hi = curve(df, arm, col)
            ax.fill_between(ns, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
            ax.plot(ns, m, color=c, marker=mk, ms=4, lw=1.4, label=lbl, zorder=3)
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
        ax.set_xlim(0.85, 30); ax.set_xlabel("images from the new scanner (per half)"); ax.set_title(title)
        tidy(ax)
    axes[0, 0].set_ylim(0.60, 0.79); axes[0, 1].set_ylim(0.68, 0.82)
    axes[0, 0].set_ylabel("new-scanner F1")
    axes[0, 0].legend(fontsize=6.4, loc="lower right", handlelength=1.6)
    axes[0, 1].legend(fontsize=6.4, loc="lower right", handlelength=1.6)
    xs = [0.2, 0.3, 0.4, 0.5, 0.6]
    meths = [("none", "#8A8A8A", "", (0, (3, 2)), "source cutoff"),
             ("upa", C_SRC, "s", "-", "UPA (source reference)"),
             ("bbse", C_LS, "D", "-", "BBSE (label-shift estimate)"),
             ("tgr", C_THR, "^", "-", "TGR (teacher reference)"),
             ("oracle", INK, "", (0, (1, 1.2)), "labelled cutoff (upper bound)")]
    for ax, st, title in ((axes[1, 0], "r18_kd", "c  ResNet-18: prevalence shift"), (axes[1, 1], "cnx_kd", "d  ConvNeXt-T: prevalence shift")):
        ax.axvspan(0.32, 0.45, color="#F0F0F0", zorder=0, lw=0)
        for m, c, mk, ls, lbl in meths:
            ax.plot(xs, [tab.loc[(st, str(x), m), "mean"] for x in xs], color=c, marker=mk or None, ms=3.8, lw=1.4 if mk else 1.1,
                    ls=ls, label=lbl, zorder=3)
        ax.set_xticks(xs); ax.set_xlabel("mitosis prevalence on the new scanner (resampled)"); ax.set_title(title)
        ax.set_ylim(0.38, 0.90); tidy(ax)
        ax.text(0.385, 0.885, "source range", ha="center", va="top", fontsize=6.3, color=MUTED)
    axes[1, 0].set_ylabel("new-scanner F1")
    axes[1, 1].legend(fontsize=6.4, loc="lower right", handlelength=1.8)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    save(fig, "fig3_threshold_layer")

# ---------------------------------------------------------------- Fig 4
def fig4():
    n1 = pd.read_csv(D / "noninferiority_hopt0.csv"); n2 = pd.read_csv(D / "noninferiority_virchow2.csv")
    def test(df, t):
        r = df[df.test == t].iloc[0]; return r["diff"], r["lo"], r["hi"]
    def desc(df, key):
        r = df[df.quantity.fillna("").str.startswith(key)].iloc[0]; return r["value"], r["lo"], r["hi"]
    R = [  # label, (est, lo, hi), colour, filled, group
        ("KD + TGA (retrained)", test(n1, "N1a"), C_RET, True, "H-optimus-0 teacher"),
        ("KD + TGR (no retraining)", test(n1, "N1b"), C_THR, True, "H-optimus-0 teacher"),
        ("KD + TGA, both labelled cutoff", desc(n1, "same treatment, labelled cutoff"), C_RET, False, "H-optimus-0 teacher"),
        ("KD + TGA, both 8-view TTA", desc(n1, "same treatment, 8-view TTA"), C_RET, False, "H-optimus-0 teacher"),
        ("KD + TGA vs mean of 3 teacher seeds", desc(n1, "cnx_kd_tga source minus mean of teacher"), C_RET, False, "H-optimus-0 teacher"),
        ("ResNet-18 student, KD + TGA", desc(n1, "r18_kd_tga source minus teacher"), "#8A8A8A", False, "H-optimus-0 teacher"),
        ("KD + TGA (retrained)", test(n2, "N2d"), C_RET, True, "Virchow2 teacher"),
        ("KD + TGR (no retraining)", test(n2, "N2e"), C_THR, True, "Virchow2 teacher"),
        ("KD + TGA, both labelled cutoff", desc(n2, "same treatment, labelled cutoff"), C_RET, False, "Virchow2 teacher"),
        ("KD + TGA, both 8-view TTA", desc(n2, "same treatment, 8-view TTA"), C_RET, False, "Virchow2 teacher"),
        ("KD + TGA vs mean of 3 teacher seeds", desc(n2, "cnx_kd_tga_virchow2 source minus mean of teacher"), C_RET, False, "Virchow2 teacher"),
    ]
    fF = D / "teacher_seed_robustness.csv"
    if fF.exists():                                   # Protocol F: one student per teacher LoRA seed, each against its own teacher
        F = pd.read_csv(fF)
        def ftest(t):
            r = F[F.test == t].iloc[0]; return r["diff"], r["lo"], r["hi"]
        i = [k for k, r in enumerate(R) if r[0].startswith("KD + TGA vs mean of 3") and r[4] == "H-optimus-0 teacher"][0]
        del R[i]                                      # replaced by students of every teacher seed, each against its own teacher
        R[0:1] = [("KD + TGA, teacher LoRA seed 0", R[0][1], R[0][2], R[0][3], R[0][4]),
                  ("KD + TGA, teacher LoRA seed 1", ftest("F1"), C_RET, True, "H-optimus-0 teacher"),
                  ("KD + TGA, teacher LoRA seed 2", ftest("F2"), C_RET, True, "H-optimus-0 teacher"),
                  ("KD + TGA, mean over seeds 0–2", ftest("key secondary"), "#7A4A00", True, "H-optimus-0 teacher")]
    fig = plt.figure(figsize=(6.0, 2.75 + (0.2 if fF.exists() else 0)))
    ax = fig.add_axes([0.30, 0.13, 0.40, 0.76]); bx = fig.add_axes([0.80, 0.13, 0.19, 0.76])
    y, ys, heads = 0, [], []
    for i, (lab, (e, lo, hi), col, filled, grp) in enumerate(R):
        if i == 0 or grp != R[i - 1][4]:
            y -= 0.6; heads.append((y, grp)); y -= 1
        ys.append(y)
        ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="o", ms=4.2, color=col, mfc=col if filled else "white", mew=1.0,
                    lw=1.0, capsize=1.8, zorder=3)
        ax.text(-0.0445, y, lab, ha="right", va="center", fontsize=6.8, color=INK if filled else "#444444",
                weight="bold" if filled else "normal", transform=ax.transData, clip_on=False)
        y -= 1
    for yh, grp in heads:
        ax.text(-0.0445, yh, f"ConvNeXt-T student vs {grp}" if "ResNet" not in grp else grp, ha="right", va="center",
                fontsize=7.2, weight="bold", color=MUTED, clip_on=False)
    ax.axvspan(-0.045, -0.02, color="#EFEFEF", zorder=0, lw=0)
    ax.axvline(-0.02, color="#8A8A8A", lw=0.8, ls=(0, (3, 2))); ax.axvline(0, color="#555555", lw=0.6)
    ax.text(-0.0205, 0.0, "margin", ha="right", va="top", fontsize=6.3, color=MUTED)
    ax.set_xlim(-0.045, 0.032); ax.set_ylim(y + 0.3, 0.2); ax.set_yticks([])
    ax.set_xlabel("student minus teacher, new-scanner F1 (95% CI)"); tidy(ax, grid="x"); ax.spines["left"].set_visible(False)
    hd = [Line2D([], [], marker="o", ls="", color="#555555", ms=4), Line2D([], [], marker="o", ls="", color="#555555", mfc="white", ms=4)]
    ax.legend(hd, ["pre-specified test", "descriptive"], fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, borderaxespad=0.2)
    E = pd.read_csv(D / "efficiency.csv")
    order = ["resnet18", "convnext_tiny", "uni", "virchow2", "uni2h", "hopt0"]
    nm = {"resnet18": "ResNet-18", "convnext_tiny": "ConvNeXt-T", **NAME}
    E = E.set_index("model").loc[order]
    yy = np.arange(len(order))[::-1]
    bx.barh(yy, E.cells_per_s, color=["#4D4D4D" if r == "student" else "#BDBDBD" for r in E.role], height=0.62, zorder=3)
    for yv, (m, r) in zip(yy, E.iterrows()):
        bx.text(r.cells_per_s * 1.15, yv, f"{r.cells_per_s:,.0f}", va="center", fontsize=6.3, color=INK)
        bx.text(28, yv, f"{nm[m]}\n{r.params_M:,.0f} M", va="center", ha="right", fontsize=6.3, color=INK, linespacing=0.95)
    bx.set_xscale("log"); bx.set_xlim(30, 1e5); bx.set_xticks([1e2, 1e3, 1e4, 1e5]); bx.set_yticks([]); bx.set_ylim(-0.6, len(order) - 0.4)
    bx.set_xlabel("cells / s (RTX 3090 Ti)"); bx.set_title("throughput", fontsize=7.5); tidy(bx, grid="x"); bx.spines["left"].set_visible(False)
    save(fig, "fig4_close_and_cheap")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()

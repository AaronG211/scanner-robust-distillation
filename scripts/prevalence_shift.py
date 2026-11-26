#!/usr/bin/env python3
"""Protocol P (PREVALENCE_SHIFT_PROTOCOL.md): TGR vs source-rate and vs none when the new scanner's prevalence shifts.

Simulated prevalence shift by resampling the new scanner's cells; everything else as scripts/recal_compare.py.
Inputs (data/pc_runs_2026-11-17/): e0_cells_r18.csv.gz, e0_cells_cnx.csv.gz, teacher_newscanner_cells.csv.gz,
teacher_newscanner_cells_multi.csv.gz. Outputs: prevalence_shift_table.csv, prevalence_shift_contrasts.csv,
figures/prevalence_shift.png.
Usage: python3 scripts/prevalence_shift.py [--draw-offset K]   (K = 50 is the pre-specified sensitivity run)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "pc_runs_2026-11-17"
B, R = 20000, 20
OFFSET = int(sys.argv[sys.argv.index("--draw-offset") + 1]) if "--draw-offset" in sys.argv else 0
FOLDS = ["CS2", "S360", "XR"]
PIS = ["natural", 0.20, 0.30, 0.40, 0.50, 0.60]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))

# ---- method definitions copied unchanged from recal_compare.py ----
def fit_bias(z, pt):
    def loss(b):
        q = np.clip(sg(z + b), 1e-7, 1 - 1e-7); return -(pt * np.log(q) + (1 - pt) * np.log(1 - q)).mean()
    return minimize_scalar(loss, bounds=(-10, 10), method="bounded").x

def best_cut(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); k = np.arange(1, len(p) + 1); return p[o][np.argmax(2 * tp / (k + y.sum()))]

def rate_cut(p, rate):
    return np.inf if rate <= 0 else (-np.inf if rate >= 1 else np.quantile(p, 1 - rate))

def em_prior(p, pi_s, iters=200):
    pi_t = pi_s
    for _ in range(iters):
        a = (pi_t / pi_s) * p; b = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
        new = (a / (a + b)).mean()
        if abs(new - pi_t) < 1e-7:
            break
        pi_t = new
    return pi_t

def methods(p, y, half, thr, vp, T):
    """p, y, half: retained new-scanner cells of one run; thr: source cutoff; vp: source-val probabilities;
    T: {fm: (teacher probabilities on the same cells, teacher cutoff)}. Optimus ('hopt0') drives the main methods."""
    z = lg(p); pt, tt = T["hopt0"]
    src_rate = (vp >= thr).mean(); pi_s = vp.mean()
    out = {k: np.zeros(len(p), bool) for k in ("tgr", "teacher_rate", "teacher_pl", "source_rate", "em_prior", "oracle")}
    out.update({f"tgr_{fm}": np.zeros(len(p), bool) for fm in T if fm != "hopt0"})
    out["none"] = p >= thr
    for fit, to in (("A", "B"), ("B", "A")):
        fi = half == fit; pi = half == to
        out["tgr"][pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= lg(np.array([tt]))[0]
        out["teacher_rate"][pi] = p[pi] >= rate_cut(p[fi], (pt[fi] >= tt).mean())
        out["teacher_pl"][pi] = p[pi] >= best_cut(p[fi], pt[fi] >= tt)
        out["source_rate"][pi] = p[pi] >= rate_cut(p[fi], src_rate)
        pi_t = em_prior(p[fi], pi_s)
        out["em_prior"][pi] = z[pi] + lg(np.array([pi_t]))[0] - lg(np.array([pi_s]))[0] >= lg(np.array([thr]))[0]
        out["oracle"][pi] = p[pi] >= best_cut(p[fi], y[fi])
        for fm, (q, tq) in T.items():
            if fm != "hopt0":
                out[f"tgr_{fm}"][pi] = z[pi] + fit_bias(z[fi], q[fi]) >= lg(np.array([tq]))[0]
    return out

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

def stat(d):
    boot = d[1:]; p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (len(boot) + 1))
    return d[0], np.percentile(boot, 2.5), np.percentile(boot, 97.5), p

# ---- data ----
cells = {"r18": pd.read_csv(D / "e0_cells_r18.csv.gz"), "cnx": pd.read_csv(D / "e0_cells_cnx.csv.gz")}
t1 = pd.read_csv(D / "teacher_newscanner_cells.csv.gz").assign(fm="hopt0")
tm = pd.read_csv(D / "teacher_newscanner_cells_multi.csv.gz")
tm = tm[tm.fm != "hopt0"]
TE = pd.concat([t1[["fm", "holdout", "row", "p_teacher", "thr_teacher"]], tm[["fm", "holdout", "row", "p_teacher", "thr_teacher"]]])
STUDENTS = [("r18_kd", "r18"), ("cnx_kd", "cnx"), ("r18_control", "r18"), ("cnx_control", "cnx")]
BOOT_KEEP = {"tgr", "source_rate", "none"}      # full bootstrap only where the contrasts and the figure need it

point, boot = [], {}
for fi_, h in enumerate(FOLDS):
    ref = cells["r18"]; ref = ref[(ref.holdout == h) & (ref.arm == "r18_kd") & (ref.seed == 0) & (ref.split == "unseen")].sort_values("row")
    rows = ref.row.values; y = (ref.y == 1).values; half = ref.half.values
    imgs = np.sort(ref.image_id.unique()); n = len(imgs); ii = np.searchsorted(imgs, ref.image_id.values)
    th = {fm: g.set_index("row").loc[rows] for fm, g in TE[TE.holdout == h].groupby("fm")}
    T_all = {fm: (g.p_teacher.values, g.thr_teacher.values[0]) for fm, g in th.items()}
    # resampling masks, shared by everything in this fold
    masks = {"natural": [np.ones(len(rows), bool)]}
    npos, nneg = y.sum(), (~y).sum()
    for k, pi in enumerate(PIS[1:]):
        g = np.random.default_rng(20261130 + 100 * fi_ + k + OFFSET)
        ms = []
        for _ in range(R):
            m = np.ones(len(rows), bool)
            if pi < y.mean():
                keep = int(round(pi / (1 - pi) * nneg)); drop = np.flatnonzero(y)[g.permutation(npos)[keep:]]
            else:
                keep = int(round((1 - pi) / pi * npos)); drop = np.flatnonzero(~y)[g.permutation(nneg)[keep:]]
            m[drop] = False; ms.append(m)
        masks[pi] = ms
    gb = np.random.default_rng([20261131, fi_])
    W = np.vstack([np.ones(n), gb.multinomial(n, np.full(n, 1 / n), size=B)]).astype(np.float32)
    nS = 10
    sd = np.vstack([np.arange(nS), gb.integers(0, nS, (B, nS))])
    print(f"{h}: n_cells {len(rows)}, natural pos rate {y.mean():.3f}; realised rates",
          {pi: round(float(np.mean([y[m].mean() for m in ms])), 3) for pi, ms in masks.items()}, flush=True)
    for student, src in STUDENTS:
        c = cells[src]; c = c[(c.holdout == h) & (c.arm == student)]
        acc = {}                                   # (pi, method) -> list over seeds of (B+1,) F1
        for s, g in c.groupby("seed"):
            u = g[g.split == "unseen"].sort_values("row"); assert np.array_equal(u.row.values, rows)
            assert np.array_equal((u.y == 1).values, y) and np.array_equal(u.half.values, half)
            vp = g[g.split == "val"].p_single.values; p = u.p_single.values; thr = u.thr_single.values[0]
            for pi, ms in masks.items():
                cnt = {}
                for m in ms:
                    T = {fm: (q[m], tq) for fm, (q, tq) in T_all.items()}
                    preds = methods(p[m], y[m], half[m], thr, vp, T)
                    preds["teacher_alone"] = T["hopt0"][0] >= T["hopt0"][1]
                    ym, im = y[m], ii[m]
                    for meth, pr in preds.items():
                        a = cnt.setdefault(meth, np.zeros((n, 3)))
                        np.add.at(a, im, np.c_[pr & ym, pr & ~ym, ~pr & ym])
                for meth, a in cnt.items():
                    if meth in BOOT_KEEP and student in ("r18_kd", "cnx_kd"):
                        tp, fp, fn = (W @ a.astype(np.float32)).T.astype(np.float64)
                    else:
                        tp, fp, fn = a.sum(0)[:, None]
                    acc.setdefault((pi, meth), []).append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
        for (pi, meth), lst in acc.items():
            F = np.stack(lst, 1)                   # (B+1 or 1, seeds)
            point.append(dict(student=student, pi=str(pi), method=meth, fold=h, f1=F[0].mean()))
            if F.shape[0] > 1:
                boot.setdefault((student, str(pi), meth), []).append(np.take_along_axis(F, sd, 1).mean(1))
        print(f"  {h} {student} done", flush=True)

tab = pd.DataFrame(point).pivot_table(index=["student", "pi", "method"], columns="fold", values="f1")
tab["mean"] = tab[FOLDS].mean(1)
series = {k: np.mean(v, 0) for k, v in boot.items()}
for k in series:
    assert np.isclose(series[k][0], tab.loc[k, "mean"]), k

# consistency check against recal_compare.py family 1 (natural prevalence)
expect = {("r18_kd", "none"): 0.725, ("r18_kd", "source_rate"): 0.760, ("r18_kd", "tgr"): 0.762,
          ("cnx_kd", "none"): 0.765, ("cnx_kd", "source_rate"): 0.782, ("cnx_kd", "tgr"): 0.801}
for (st, m), v in expect.items():
    got = tab.loc[(st, "natural", m), "mean"]
    assert round(got, 3) == v, f"consistency check FAILED: {st} {m} {got:.4f} vs {v}"
print("consistency check passed (natural prevalence reproduces recal_compare family 1)\n")

rows_c = []
for st in ("r18_kd", "cnx_kd"):
    for comp in ("source_rate", "none"):
        for pi in ("0.2", "0.6"):
            est, lo, hi, p = stat(series[(st, pi, "tgr")] - series[(st, pi, comp)])
            rows_c.append(dict(student=st, pi=float(pi), contrast=f"TGR vs {comp}", diff=est, lo=lo, hi=hi, p=p))
con = pd.DataFrame(rows_c); con["holm"] = holm(con.p.values)
desc = []
for st in ("r18_kd", "cnx_kd"):
    for comp in ("source_rate", "none"):
        for pi in PIS:
            est, lo, hi, p = stat(series[(st, str(pi), "tgr")] - series[(st, str(pi), comp)])
            desc.append(dict(student=st, pi=str(pi), contrast=f"TGR vs {comp}", diff=est, lo=lo, hi=hi))
desc = pd.DataFrame(desc)

pd.set_option("display.width", 250)
order = ["none", "source_rate", "em_prior", "teacher_rate", "teacher_pl", "tgr", "tgr_virchow2", "tgr_uni2h", "tgr_uni", "oracle", "teacher_alone"]
wide = tab["mean"].unstack("pi")[[str(x) for x in PIS]]
wide = wide.reindex([(s, m) for s, _ in STUDENTS for m in order if (s, m) in wide.index])
print("New-scanner F1 (mean over seeds, then folds) by target prevalence:")
print(wide.round(3).to_string()); print()
if OFFSET == 0:
    print("PRE-SPECIFIED family (8 contrasts, Holm):")
    print(con.round(4).to_string(index=False)); print()
    print("Descriptive: TGR minus comparator at every prevalence:")
    print(desc.round(4).to_string(index=False))
    tab.reset_index().to_csv(D / "prevalence_shift_table.csv", index=False)
    con.to_csv(D / "prevalence_shift_contrasts.csv", index=False)
    desc.to_csv(D / "prevalence_shift_descriptive.csv", index=False)
    # figure: absolute F1 against target prevalence, per student
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sty = {"none": ("#999999", "o", "fixed source cutoff (none)"), "source_rate": ("#D55E00", "s", "source-rate"),
           "tgr": ("#0072B2", "^", "TGR (Optimus)"), "oracle": ("#000000", "", "labelled cutoff (upper bound)")}
    xs = [0.20, 0.30, 0.40, 0.50, 0.60]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (st, name) in zip(axes, (("r18_kd", "ResNet-18 KD student"), ("cnx_kd", "ConvNeXt-T KD student"))):
        ax.axvspan(0.32, 0.45, color="#eeeeee", zorder=0)
        for meth, (col, mk, lab) in sty.items():
            ysr = [tab.loc[(st, str(x), meth), "mean"] for x in xs]
            if meth == "oracle":
                ax.plot(xs, ysr, ls=":", color=col, lw=1.5, label=lab)
                continue
            lo = [np.percentile(series[(st, str(x), meth)][1:], 2.5) for x in xs]
            hi = [np.percentile(series[(st, str(x), meth)][1:], 97.5) for x in xs]
            ax.errorbar(xs, ysr, yerr=[np.subtract(ysr, lo), np.subtract(hi, ysr)], color=col, marker=mk, ms=7, lw=2,
                        capsize=3, label=lab)
        ax.set_title(name, fontsize=11); ax.set_xlabel("mitosis prevalence on the new scanner (resampled)")
        ax.set_xticks(xs); ax.grid(axis="y", color="#dddddd", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("new-scanner F1")
    axes[0].text(0.385, axes[0].get_ylim()[0] + 0.005, "source\nprevalence", ha="center", va="bottom", fontsize=8, color="#666666")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Cutoff recalibration under simulated prevalence shift (Optimus teacher, 3 scanners, 10 seeds)", fontsize=11)
    fig.tight_layout()
    out = ROOT / "figures" / "prevalence_shift.png"; fig.savefig(out, dpi=200); print("wrote", out)
else:
    print(f"SENSITIVITY (draw offset {OFFSET}): the 8 pre-specified contrasts, point estimates")
    print(con[["student", "pi", "contrast", "diff"]].round(4).to_string(index=False))

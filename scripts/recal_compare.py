#!/usr/bin/env python3
"""Protocol R (RECAL_PROTOCOL.md): TGR against label-free recalibration baselines, and TGR vs AdaBN with very few images.

Inputs (data/pc_runs_2026-11-17/):
  e0_cells_r18.csv.gz, e0_cells_cnx.csv.gz   per-cell student probabilities on source-val and all new-scanner cells
  teacher_newscanner_cells.csv.gz            Optimus teacher, view 0, with its source-val cutoff
  r18_crossfit_cells.csv.gz                  AdaBN predictions of the ResNet students (fit-half statistics)
  e02_tgr_budget_preds.npz, e02_adabn_budget_preds.npz   per-cell predictions for the few-image family
Missing inputs are skipped with a message.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
B = 20000
rng = np.random.default_rng(20261126)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))

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

def methods(u, v, teacher, adabn=None):
    """u: new-scanner cells of one run (sorted by row); v: its source-val cells. Returns {method: bool predictions}."""
    p = u.p_single.values; z = lg(p); thr = u.thr_single.values[0]; y = (u.y == 1).values
    pt = teacher.p_teacher.values; tt = teacher.thr_teacher.values[0]
    src_rate = (v.p_single.values >= thr).mean(); pi_s = v.p_single.values.mean()
    out = {k: np.zeros(len(u), bool) for k in ("tgr", "teacher_rate", "teacher_pl", "source_rate", "em_prior", "oracle")}
    out["none"] = p >= thr
    for fit, to in (("A", "B"), ("B", "A")):
        fi = (u.half == fit).values; pi = (u.half == to).values
        out["tgr"][pi] = z[pi] + fit_bias(z[fi], pt[fi]) >= lg(np.array([tt]))[0]
        out["teacher_rate"][pi] = p[pi] >= rate_cut(p[fi], (pt[fi] >= tt).mean())
        out["teacher_pl"][pi] = p[pi] >= best_cut(p[fi], pt[fi] >= tt)
        out["source_rate"][pi] = p[pi] >= rate_cut(p[fi], src_rate)
        pi_t = em_prior(p[fi], pi_s)
        out["em_prior"][pi] = z[pi] + lg(np.array([pi_t]))[0] - lg(np.array([pi_s]))[0] >= lg(np.array([thr]))[0]
        out["oracle"][pi] = p[pi] >= best_cut(p[fi], y[fi])
    if adabn is not None:
        out["adabn"] = adabn
    return out

def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i])); adj[i] = run
    return adj

def stat(d):
    boot = d[1:]; p = min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (len(boot) + 1))
    return d[0], np.percentile(boot, 2.5), np.percentile(boot, 97.5), p

teacher = pd.read_csv(D / "teacher_newscanner_cells.csv.gz")
xf = pd.read_csv(D / "r18_crossfit_cells.csv.gz", usecols=["holdout", "arm", "seed", "row", "p_adabn_single", "thr_single"])
series, tables = {}, []
for student, fname in (("r18_kd", "e0_cells_r18.csv.gz"), ("cnx_kd", "e0_cells_cnx.csv.gz"),
                       ("r18_control", "e0_cells_r18.csv.gz"), ("cnx_control", "e0_cells_cnx.csv.gz")):
    if not (D / fname).exists():
        print("missing", fname, "- skipping", student); continue
    c = pd.read_csv(D / fname); c = c[c.arm == student]
    per_fold = {}
    for h, gh in c.groupby("holdout"):
        imgs = np.sort(gh.image_id[gh.split == "unseen"].unique()); n = len(imgs)
        W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
        th = teacher[teacher.holdout == h]
        F = {}
        for s, g in gh.groupby("seed"):
            u = g[g.split == "unseen"].sort_values("row"); v = g[g.split == "val"]
            t = th.set_index("row").loc[u.row.values].reset_index()
            ad = None
            if student.startswith("r18_"):
                a = xf[(xf.holdout == h) & (xf.arm == student) & (xf.seed == s)].set_index("row").loc[u.row.values]
                ad = a.p_adabn_single.values >= a.thr_single.values
            y = (u.y == 1).values; ii = np.searchsorted(imgs, u.image_id.values)
            for m, pred in methods(u, v, t, ad).items():
                cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
                tp, fp, fn = (W @ cnt).T
                F.setdefault(m, []).append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
        per_fold[h] = {m: np.stack(v_, 1) for m, v_ in F.items()}
    nS = next(iter(per_fold.values()))["none"].shape[1]
    sd = {h: np.vstack([np.arange(nS), rng.integers(0, nS, (B, nS))]) for h in per_fold}
    series[student] = {m: np.mean([np.take_along_axis(per_fold[h][m], sd[h], 1).mean(1) for h in per_fold], 0)
                       for m in per_fold[next(iter(per_fold))]}
    tables.append(pd.DataFrame({m: {**{h: per_fold[h][m][0].mean() for h in per_fold}, "mean": series[student][m][0]}
                                for m in series[student]}).T.assign(student=student))

pd.set_option("display.width", 220)
if tables:
    print("New-scanner F1 by method (rows) and fold:")
    print(pd.concat(tables).reset_index().rename(columns={"index": "method"}).set_index(["student", "method"]).round(3).to_string()); print()

rows = []
for student, comps in (("r18_kd", ["none", "source_rate", "em_prior", "teacher_pl", "adabn"]),
                       ("cnx_kd", ["none", "source_rate", "em_prior", "teacher_pl"])):
    if student not in series:
        continue
    for m in comps:
        est, lo, hi, p = stat(series[student]["tgr"] - series[student][m])
        rows.append(dict(family=1, student=student, contrast=f"TGR vs {m}", diff=round(est, 4), lo=round(lo, 4), hi=round(hi, 4), p=p))

# family 2: very few images, TGR vs AdaBN (ResNet KD), same draws
ft, fa = D / "e02_tgr_budget_preds.npz", D / "e02_adabn_budget_preds.npz"
if ft.exists() and fa.exists():
    T, A_ = np.load(ft), np.load(fa); cfg = [tuple(x) for x in T["configs"]]
    assert cfg == [tuple(x) for x in A_["configs"]]
    xc = xf.merge(pd.read_csv(D / "r18_crossfit_cells.csv.gz", usecols=["holdout", "arm", "seed", "row", "image_id", "y"]),
                  on=["holdout", "arm", "seed", "row"])
    for n_img in (1, 2, 5):
        idx = [i for i, (n, k) in enumerate(cfg) if n == n_img]
        diffs = []
        for h in ("CS2", "S360", "XR"):
            g0 = xc[(xc.holdout == h) & (xc.arm == "r18_kd") & (xc.seed == 0)].sort_values("row")
            imgs = np.sort(g0.image_id.unique()); n = len(imgs); y = (g0.y == 1).values; ii = np.searchsorted(imgs, g0.image_id.values)
            W = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=B)])
            per_seed = []
            for s in range(10):
                f = {}
                for name, P in (("tgr", T[f"{h}|r18_kd|{s}"]), ("adabn", A_[f"{h}|r18_kd|{s}"])):
                    vals = []
                    for i in idx:
                        pred = P[i]; cnt = np.zeros((n, 3)); np.add.at(cnt, ii, np.c_[pred & y, pred & ~y, ~pred & y])
                        tp, fp, fn = (W @ cnt).T; vals.append(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9))
                    f[name] = np.mean(vals, 0)
                per_seed.append(f["tgr"] - f["adabn"])
            x = np.stack(per_seed, 1); sdr = np.vstack([np.arange(10), rng.integers(0, 10, (B, 10))])
            diffs.append(np.take_along_axis(x, sdr, 1).mean(1))
        est, lo, hi, p = stat(np.mean(diffs, 0))
        rows.append(dict(family=2, student="r18_kd", contrast=f"TGR vs AdaBN, {n_img} image(s)", diff=round(est, 4), lo=round(lo, 4), hi=round(hi, 4), p=p))
else:
    print("missing budget prediction files - family 2 skipped")

if rows:
    out = pd.DataFrame(rows)
    out["holm"] = np.nan
    for fam, g in out.groupby("family"):
        out.loc[g.index, "holm"] = holm(g.p.values)
    out.to_csv(D / "recal_compare.csv", index=False)
    show = out.copy(); show["p"] = show.p.map(lambda v: f"{v:.4f}"); show["holm"] = show.holm.map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))

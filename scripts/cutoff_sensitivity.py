"""Amendment 4 sensitivity analysis: family A (H1-H6) with a full-pipeline bootstrap.

As cutoff_confirm.py (target cases resampled within fold, seeds resampled), plus the SOURCE-validation cases are
resampled: on every resample t_src is re-chosen by best_threshold on the weighted source cells, and the source
positive rate (PRM) and, for SLD, the Platt map and source prevalence are re-estimated. Runs that share the same set
of validation cases share one draw. Still fixed: trained weights and the cross-fitted AdaBN statistics.

Writes results/cutoff_sensitivity.csv (conditional vs full-pipeline CIs and Holm p) and prints a per-fold table.
Usage: .venv/bin/python scripts/cutoff_sensitivity.py [--B 2000]
"""
import argparse, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from cutoff_transfer import FOLDS, ROOT, RES, meta, logit, sigmoid, fit_platt
from cutoff_confirm import load_all, w_cutoff, w_f1, aggregate


def w_best_threshold(p, y, w):
    """Weighted best_threshold: identical rule (smallest cutoff with the maximal F1; 0.5 if every F1 is 0)."""
    m = w > 0
    p, y, w = p[m], y[m], w[m]
    u, inv = np.unique(p, return_inverse=True)
    tp_at = np.bincount(inv, weights=w * (y == 1), minlength=len(u))
    fp_at = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    TP, FP = tp_at[::-1].cumsum()[::-1], fp_at[::-1].cumsum()[::-1]
    P = tp_at.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        pr = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        rc = TP / P if P > 0 else np.zeros_like(TP)
        f1 = np.where(pr + rc > 0, 2 * pr * rc / (pr + rc), 0.0)
    k = int(np.argmax(f1))
    return float(u[k]) if f1[k] > 0 else 0.5


def w_best_threshold_grid(p, y, w):
    """Weighted version of train_patch.best_threshold (our runs): candidates np.unique(np.round(p, 3)) in float32,
    smallest candidate with the maximal F1 (initial best -1, so the first candidate wins if every F1 is 0)."""
    m = w > 0
    p, y, w = p[m], y[m], w[m]
    cand = np.unique(np.round(p.astype(np.float32), 3)).astype(np.float64)
    o = np.argsort(p, kind="stable"); ps = p[o]
    wy, wn = np.concatenate([(w * (y == 1))[o], [0.0]]), np.concatenate([(w * (y == 0))[o], [0.0]])
    sTP, sFP = wy[::-1].cumsum()[::-1], wn[::-1].cumsum()[::-1]      # suffix sums: cells with index >= k
    k = np.searchsorted(ps, cand, side="left")                        # p >= t  <=>  index >= k
    TP, FP = sTP[k], sFP[k]; FN = sTP[0] - TP
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), 0.0)
    return float(cand[int(np.argmax(f1))])


def w_platt(p, y, w, x0):
    z = logit(p); W = w.sum()

    def nll(v):
        q = np.clip(sigmoid(v[0] * z + v[1]), 1e-12, 1 - 1e-12)
        return -np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))) / W
    return minimize(nll, x0, method="Nelder-Mead").x


def val_image_ids(R):
    if R.source == "ours":
        d = pd.read_csv(RES / "preds" / f"{R.fold}_{R.name}_seed{R.seed}.csv")
        return d[d.set == "val"].image_id.values
    d = pd.read_csv(ROOT / "data" / "axel_predictions" / f"{R.name}.csv")
    v = d[(d.holdout == R.fold) & (d.seed == R.seed) & (d.set == "val")]
    return meta().set_index("ann_id").loc[v.ann_id.values, "image_id"].values


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(); B = a.B
    runs = [R for R in load_all() if R.group in ("nonstain", "stain", "axel")]
    print(f"{len(runs)} runs")

    # unit-weight check: the weighted rule must reproduce every stored t_src
    bt = lambda R: w_best_threshold_grid if R.source == "ours" else w_best_threshold
    mism = [(R.fold, R.name, R.seed, R.ts, bt(R)(R.sp, R.sy, np.ones(len(R.sp)))) for R in runs]
    print("max |t_src(stored) - t_src(weighted rule, unit weights)| =", max(abs(m[3] - m[4]) for m in mism))
    mism = [m for m in mism if abs(m[3] - m[4]) > 0]
    print(f"t_src reproduced by weighted rule at unit weights: {len(runs) - len(mism)}/{len(runs)}")
    for m in mism[:10]:
        print("   mismatch", m)
    assert not mism, "weighted best_threshold does not reproduce stored t_src"

    rng = np.random.default_rng(2029)
    # target cases per fold (as in cutoff_confirm) and seeds
    tcases = {}
    for f in FOLDS:
        ids = np.unique(np.concatenate([R.t.image_id.values for R in runs if R.fold == f]))
        tcases[f] = (ids, rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B).astype(float))
    seedw = {"ours": rng.multinomial(5, np.full(5, 0.2), size=B) / 5.0,
             "axel": rng.multinomial(3, np.full(3, 1 / 3), size=B) / 3.0}
    # source-validation cases: one draw per distinct validation case set
    vids = [val_image_ids(R) for R in runs]
    groups = {}
    for i, R in enumerate(runs):
        key = (R.fold, tuple(np.unique(vids[i])))
        groups.setdefault(key, []).append(i)
    vdraw = {}
    for key in sorted(groups, key=str):
        ids = np.array(key[1])
        vdraw[key] = (ids, rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B).astype(float))
    print(f"{len(groups)} distinct source-validation case sets; cases per set:",
          sorted({len(k[1]) for k in groups}))

    jobs = {}
    for i, R in enumerate(runs):
        if R.source == "ours":
            jobs[i] = [("plain", "crossfit", m) for m in ("SRC", "PRM", "SLD")] + \
                      [("adabncf", "crossfit", m) for m in ("SRC", "PRM")]
        else:
            jobs[i] = [("plain", "official", m) for m in ("SRC", "PRM")]
    vals = {(i, *j): np.empty(B) for i in jobs for j in jobs[i]}
    t0 = time.time()
    for n, (i, js) in enumerate(jobs.items()):
        R = runs[i]
        tid, Wt = tcases[R.fold]; tidx = np.searchsorted(tid, R.t.image_id.values)
        key = (R.fold, tuple(np.unique(vids[i]))); vid, Wv = vdraw[key]; vidx = np.searchsorted(vid, vids[i])
        A = R.A; choose = bt(R)
        ts0 = choose(R.sp, R.sy, np.ones(len(R.sp)))
        assert ts0 == R.ts
        need_sld = any(j[2] == "SLD" for j in js)
        for b in range(B):
            wc, wv = Wt[b, tidx], Wv[b, vidx]
            ts = choose(R.sp, R.sy, wv)
            m = wv > 0
            r_s = float(np.sum(wv[m] * (R.sp[m] >= ts)) / wv[m].sum())
            if need_sld:
                platt = w_platt(R.sp[m], R.sy[m], wv[m], R.platt)
                pi_s = float(np.sum(wv[m] * R.sy[m]) / wv[m].sum())
            else:
                platt, pi_s = R.platt, R.pi_s
            args = (r_s, R.zs_mu, R.zs_sd, ts, platt, pi_s)
            for (mode, split, meth) in js:
                p, y = R.p[mode], R.y
                if split == "crossfit":
                    cut = np.empty(len(p))
                    for est, ev in ((A, ~A), (~A, A)):
                        cut[ev] = w_cutoff(meth, *args, p[est], wc[est])
                    vals[(i, mode, split, meth)][b] = w_f1(p, y, cut, wc)
                else:
                    c = w_cutoff(meth, *args, p[R.tr], wc[R.tr])
                    vals[(i, mode, split, meth)][b] = w_f1(p[R.te], y[R.te], c, wc[R.te])
        if n % 20 == 0:
            print(f"  {n + 1}/{len(jobs)} runs, {time.time() - t0:.0f}s", flush=True)

    nonstain = lambda R: R.group == "nonstain"
    stain = lambda R: R.group == "stain"
    grid = lambda R: R.group in ("nonstain", "stain")
    axel = lambda R: R.group == "axel"
    tests = [
        ("H1", nonstain, ("plain", "crossfit", "PRM"), ("plain", "crossfit", "SRC")),
        ("H2", stain, ("plain", "crossfit", "PRM"), ("plain", "crossfit", "SRC")),
        ("H3", grid, ("adabncf", "crossfit", "PRM"), ("adabncf", "crossfit", "SRC")),
        ("H4", axel, ("plain", "official", "PRM"), ("plain", "official", "SRC")),
        ("H5", grid, ("plain", "crossfit", "SLD"), ("plain", "crossfit", "SRC")),
        ("H6", nonstain, ("plain", "crossfit", "PRM"), ("adabncf", "crossfit", "SRC")),
    ]
    conf = pd.read_csv(RES / "cutoff_confirmatory.csv").set_index("id")
    rows = []
    for hid, sel, x, z in tests:
        d = aggregate(runs, vals, seedw, sel, *x) - aggregate(runs, vals, seedw, sel, *z)
        P = (d <= 0).mean(); p_raw = max(2 * min(P, 1 - P), 2 / B)
        rows.append({"id": hid, "dF1": conf.loc[hid, "dF1"], "cond_ci_lo": conf.loc[hid, "ci_lo"],
                     "cond_ci_hi": conf.loc[hid, "ci_hi"], "cond_p_holm": conf.loc[hid, "p_holm"],
                     "full_ci_lo": np.percentile(d, 2.5), "full_ci_hi": np.percentile(d, 97.5),
                     "full_p_raw": p_raw, "full_P_le0": P, "boot_mean": d.mean()})
    c = pd.DataFrame(rows)
    order = np.argsort(c.full_p_raw.values); m = len(c); adj = np.empty(m); run_max = 0.0
    for rank, k in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * c.full_p_raw.values[k])); adj[k] = run_max
    c["full_p_holm"] = adj
    c["robust"] = (c.cond_p_holm < 0.05) & (c.full_p_holm < 0.05)
    c.round(4).to_csv(RES / "cutoff_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nFAMILY A: conditional (original) vs full-pipeline bootstrap")
    print(c.round(4).to_string(index=False))

    # per-fold point estimates for H1 / H2 / H6 (descriptive): seeds averaged, then tags
    pt = pd.read_csv(RES / "cutoff_point_runs.csv")
    o = pt[(pt.source == "ours") & (pt.split == "crossfit")]
    g = o.groupby(["group", "fold", "name", "mode", "method"]).f1.mean().groupby(["group", "fold", "mode", "method"]).mean()
    print("\nPer held-out scanner, dF1 (point):")
    for grp in ("nonstain", "stain"):
        for f in FOLDS:
            s = g.loc[grp, f]
            print(f"  {grp:9s} {f:5s} PRM-SRC plain {s[('plain', 'PRM')] - s[('plain', 'SRC')]:+.3f}   "
                  f"PRM-SRC adabncf {s[('adabncf', 'PRM')] - s[('adabncf', 'SRC')]:+.3f}   "
                  f"plainPRM-adabncfSRC {s[('plain', 'PRM')] - s[('adabncf', 'SRC')]:+.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

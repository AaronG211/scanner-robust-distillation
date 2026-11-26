"""Phase-0 GPU diagnostics (roadmap E0.2 and E0.3). Exploratory; runs on the PC from C:\\midog\\repo\\axel_model.

  budget   AdaBN with only n unlabelled images of the fit half (n = 1, 2, 5, 10, 25; 10 image draws, the same draws as
           scripts/e02_tgr_budget.py via pick()), r18_control and r18_kd students, cross-fitted, source-val cutoff.
           -> results/e02_adabn_budget.csv (run level) and results/e02_adabn_budget_preds.npz (per cell)
  bnswap   E0.3: is TGA partly AdaBN in disguise? For each TGA-type student, replace its BatchNorm statistics with
           statistics from the source training cells only (what they would be without new-scanner cells in the batches)
           and score the other half again, with the original cutoff and with a re-chosen source-val cutoff.
           -> results/e03_bnswap.csv

  cells    per-cell probabilities of finished students on the source validation cells and on ALL new-scanner cells
           (single view and 8-view TTA) with their source-val cutoffs, for CPU-side recalibration baselines
           (UPA-style rate matching, EM prior adjustment, teacher-pseudo-label cutoff, TGR).
           Students that never saw the new scanner only (KD / control).   -> results/e0_cells_<out>.csv.gz
  probe    E0.4: scanner identifiability of teacher features (cached, training cells, view 0; raw and PCA-512) and of
           student features on the same cells (balanced accuracy of a linear probe, image-grouped CV, probe.py).
           -> results/e04_probe.csv

Usage: python e0_gpu_diag.py budget | bnswap [--arms ...] | cells --arms ... --out NAME [--seeds N] | probe
"""
import sys, argparse
sys.path.insert(0, r"C:\midog\repo\axel_model")
from pathlib import Path
import numpy as np, pandas as pd
from train import make_splits, best_threshold, prf, SCANNERS
from eval_extras import probs, adapt_bn
from probe_check import load
from crossfit import target_halves

A = Path(r"C:\midog\repo\axel_model"); DIST = Path("C:/data/midog/distill")
FOLD_IDX = {"XR": 0, "S360": 1, "CS2": 2}
BUDGETS, DRAWS = [1, 2, 5, 10, 25], 10

def pick(imgs, fold, n, k, half):
    """Deterministic image draw shared with scripts/e02_tgr_budget.py (independent of the student's seed)."""
    imgs = np.sort(imgs)
    if n >= len(imgs):
        return imgs
    return np.random.default_rng([2026, FOLD_IDX[fold], n, k, 0 if half == "A" else 1]).choice(imgs, n, replace=False)

def configs():
    return [(n, k) for n in BUDGETS for k in (range(DRAWS) if n < 25 else [0])]

d = dict(np.load(A / "cache" / "patches.npz")); X, y = d["X"], d["y"].astype(int)
ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["budget", "bnswap", "cells", "probe", "tent"])
ap.add_argument("--arms", nargs="+", default=None); ap.add_argument("--out", default=None)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--smoke", action="store_true", help="tent only: one fold, one seed, 200 cells per half (code check)")
a = ap.parse_args()

def student_path(h, arm, seed):
    """ResNet-18 students carry the backbone in the file name; ConvNeXt students keep the original naming."""
    for f in (DIST / f"student_{h}_resnet18_{arm}_s{seed}.pt", DIST / f"student_{h}_{arm}_s{seed}.pt"):
        if f.exists():
            return f
    raise FileNotFoundError(f"student {h} {arm} {seed}")

if a.what == "budget":
    arms = a.arms or ["r18_kd", "r18_control"]
    rows, preds = [], {}
    for h in SCANNERS:
        S, _ = make_splits(d, h); unseen = S["test_unseen"]; H = target_halves(d, h); yv = y[S["val"]]
        pos = {k: np.searchsorted(unseen, v) for k, v in H.items()}
        for arm in arms:
            for seed in range(10):
                m = load(DIST / f"student_{h}_resnet18_{arm}_s{seed}.pt")
                thr = best_threshold(probs(m, X, S["val"], False), yv)
                P = np.zeros((len(configs()), len(unseen)), bool)
                for ci, (n, k) in enumerate(configs()):
                    for fit, to in (("A", "B"), ("B", "A")):
                        fit_cells = H[fit]; chosen = pick(np.unique(d["image_id"][fit_cells]), h, n, k, fit)
                        ma = adapt_bn(m, X, fit_cells[np.isin(d["image_id"][fit_cells], chosen)])
                        P[ci, pos[to]] = probs(ma, X, H[to], False) >= thr
                    rows.append(dict(holdout=h, arm=arm, seed=seed, n_images=n, draw=k,
                                     f1_adabn=prf(P[ci].astype(float), y[unseen], 0.5)[2]))
                preds[f"{h}|{arm}|{seed}"] = P
                print(h, arm, seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(A / "results" / "e02_adabn_budget.csv", index=False)
    np.savez_compressed(A / "results" / "e02_adabn_budget_preds.npz", configs=np.array(configs()), **preds)
    r = pd.DataFrame(rows)
    print(r.groupby(["arm", "n_images", "holdout"]).f1_adabn.mean().groupby(["arm", "n_images"]).mean().round(3).to_string())

if a.what == "bnswap":
    arms = a.arms or ["r18_kd_tga", "r18_free_tga"]
    rows = []
    for h in SCANNERS:
        S, _ = make_splits(d, h); unseen = S["test_unseen"]; H = target_halves(d, h); yv = y[S["val"]]
        for arm in arms:
            for seed in range(10):
                for fit, to in (("A", "B"), ("B", "A")):
                    m = load(DIST / f"student_{h}_resnet18_{arm}{fit}_s{seed}.pt")
                    thr = best_threshold(probs(m, X, S["val"], False), yv)
                    ms = adapt_bn(m, X, S["train"])                     # source-only BatchNorm statistics
                    thr_s = best_threshold(probs(ms, X, S["val"], False), yv)
                    p, ps = probs(m, X, H[to], False), probs(ms, X, H[to], False)
                    yy = y[H[to]]
                    rows.append(dict(holdout=h, arm=arm, seed=seed, fit=fit, n=len(yy),
                                     tp_orig=int(((p >= thr) & (yy == 1)).sum()), fp_orig=int(((p >= thr) & (yy == 0)).sum()), fn_orig=int(((p < thr) & (yy == 1)).sum()),
                                     tp_src=int(((ps >= thr) & (yy == 1)).sum()), fp_src=int(((ps >= thr) & (yy == 0)).sum()), fn_src=int(((ps < thr) & (yy == 1)).sum()),
                                     tp_src_rethr=int(((ps >= thr_s) & (yy == 1)).sum()), fp_src_rethr=int(((ps >= thr_s) & (yy == 0)).sum()), fn_src_rethr=int(((ps < thr_s) & (yy == 1)).sum())))
                print(h, arm, seed, "done", flush=True)
    r = pd.DataFrame(rows); r.to_csv(A / "results" / "e03_bnswap.csv", index=False)
    g = r.groupby(["arm", "holdout", "seed"]).sum(numeric_only=True)
    for v in ("orig", "src", "src_rethr"):
        g[f"f1_{v}"] = 2 * g[f"tp_{v}"] / (2 * g[f"tp_{v}"] + g[f"fp_{v}"] + g[f"fn_{v}"])
    print(g.groupby(["arm", "holdout"])[["f1_orig", "f1_src", "f1_src_rethr"]].mean().groupby("arm").mean().round(3).to_string())

if a.what == "cells":
    out = []
    for h in SCANNERS:
        S, _ = make_splits(d, h); unseen = S["test_unseen"]; H = target_halves(d, h); yv = y[S["val"]]
        for arm in a.arms:
            for seed in range(a.seeds):
                m = load(student_path(h, arm, seed))
                pv = {tta: probs(m, X, S["val"], tta) for tta in (False, True)}
                thr = {tta: best_threshold(pv[tta], yv) for tta in (False, True)}
                pu = {tta: probs(m, X, unseen, tta) for tta in (False, True)}
                for split, idx, P in (("val", S["val"], pv), ("unseen", unseen, pu)):
                    out.append(pd.DataFrame({"holdout": h, "arm": arm, "seed": seed, "split": split, "row": idx,
                                             "image_id": d["image_id"][idx], "y": y[idx],
                                             "half": np.where(np.isin(idx, H["A"]), "A", np.where(np.isin(idx, H["B"]), "B", "")),
                                             "p_single": P[False], "p_tta": P[True], "thr_single": thr[False], "thr_tta": thr[True]}))
                print(h, arm, seed, "done", flush=True)
    c = pd.concat(out); c.to_csv(A / "results" / f"e0_cells_{a.out}.csv.gz", index=False)
    u = c[c.split == "unseen"]
    print(u.groupby(["arm", "holdout", "seed"]).apply(lambda g: prf((g.p_single >= g.thr_single).astype(float).values, g.y.values, 0.5)[2])
          .groupby(["arm", "holdout"]).mean().groupby("arm").mean().round(3).to_string())

if a.what == "tent":
    # TENT_PROTOCOL.md: AdaBN on the fit half -> entropy minimisation of the BatchNorm affine parameters (batch statistics)
    # -> running statistics re-estimated on the fit half -> predict the other half. Per-cell output for CPU-side stats.
    import copy, torch, torch.nn as nn
    from train import prepare
    dev = next(load(student_path("XR", "r18_kd", 0)).parameters()).device
    TENT_CFG = [(1e-3, 1), (1e-4, 1), (1e-3, 3), (1e-2, 1)]          # first = primary configuration

    def tent(model, idx, lr, epochs, seed, bs=64):
        m = copy.deepcopy(model); m.eval()
        params = []
        for p in m.parameters():
            p.requires_grad_(False)
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.train(); mod.weight.requires_grad_(True); mod.bias.requires_grad_(True); params += [mod.weight, mod.bias]
        opt = torch.optim.Adam(params, lr=lr); g = np.random.default_rng(seed)
        for _ in range(epochs):
            order = g.permutation(idx)
            for s in range(0, len(order), bs):
                b = order[s:s + bs]
                if len(b) < 2:
                    continue
                q = torch.sigmoid(m(prepare(torch.from_numpy(X[b]).to(dev), False, False))[0].float()).clamp(1e-6, 1 - 1e-6)
                loss = -(q * q.log() + (1 - q) * (1 - q).log()).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        for p in m.parameters():
            p.requires_grad_(False)
        return m.eval()

    arms = a.arms or ["r18_kd", "r18_control"]; out = []
    folds = ["XR"] if a.smoke else SCANNERS; seeds = range(1) if a.smoke else range(a.seeds)
    cfgs = TENT_CFG[:1] if a.smoke else TENT_CFG
    for h in folds:
        S, _ = make_splits(d, h); H = target_halves(d, h); yv = y[S["val"]]
        if a.smoke:
            H = {k: v[:200] for k, v in H.items()}
        for arm in (arms[:1] if a.smoke else arms):
            for seed in seeds:
                m0 = load(student_path(h, arm, seed)); thr = best_threshold(probs(m0, X, S["val"], False), yv)
                for lr, ep in cfgs:
                    for fi, (fit, to) in enumerate((("A", "B"), ("B", "A"))):
                        ma = adapt_bn(m0, X, H[fit])
                        mt = adapt_bn(tent(ma, H[fit], lr, ep, [FOLD_IDX[h], seed, fi]), X, H[fit])
                        for role, idx in (("fit", H[fit]), ("apply", H[to])):
                            out.append(pd.DataFrame({"holdout": h, "arm": arm, "seed": seed, "lr": lr, "epochs": ep, "fit": fit,
                                                     "role": role, "row": idx, "image_id": d["image_id"][idx], "y": y[idx],
                                                     "p_adabn": probs(ma, X, idx, False), "p_tent": probs(mt, X, idx, False),
                                                     "thr_single": thr}))
                print(h, arm, seed, "done", flush=True)
    c = pd.concat(out); c.to_csv(A / "results" / ("e1_tent_cells_smoke.csv.gz" if a.smoke else "e1_tent_cells.csv.gz"), index=False)
    u = c[c.role == "apply"]
    f = lambda g, col: prf((g[col] >= g.thr_single).astype(float).values, g.y.values, 0.5)[2]
    s = u.groupby(["arm", "lr", "epochs", "holdout", "seed"]).apply(lambda g: pd.Series({"adabn": f(g, "p_adabn"), "tent": f(g, "p_tent"),
                                                                                       "pos_adabn": (g.p_adabn >= g.thr_single).mean(),
                                                                                       "pos_tent": (g.p_tent >= g.thr_single).mean()}))
    print(s.groupby(["arm", "lr", "epochs", "holdout"]).mean().groupby(["arm", "lr", "epochs"]).mean().round(3).to_string())

if a.what == "probe":
    import torch
    from probe import probe_accuracy
    from distill import features as student_features
    from sklearn.decomposition import PCA
    rows = []
    for h in SCANNERS:
        S, train_scn = make_splits(d, h); tr = S["train"]
        lab = np.array([train_scn.index(sc) for sc in d["scanner"][tr]]); grp = d["image_id"][tr]
        for fm in ("hopt0", "virchow2", "uni", "uni2h"):
            f = DIST / f"teacher_{'' if fm == 'hopt0' else fm + '_'}{h}_s0.npz"
            if not f.exists():
                continue
            t = np.load(f); assert np.array_equal(t["idx"], tr)
            F = t["feat"][:, 0].astype(np.float32)
            rows.append(dict(holdout=h, model=f"teacher_{fm}", dims=F.shape[1], probe=probe_accuracy(F, lab, grp)))
            F512 = PCA(512, random_state=0).fit_transform(F)
            rows.append(dict(holdout=h, model=f"teacher_{fm}_pca512", dims=512, probe=probe_accuracy(F512, lab, grp)))
            print(h, fm, rows[-2]["probe"], rows[-1]["probe"], flush=True)
        for arm in (a.arms or ["r18_control", "r18_kd", "r18_kd_tgaA"]):
            for seed in range(3):
                try:
                    m = load(student_path(h, arm, seed))
                except FileNotFoundError:
                    continue
                F = student_features(m, X, tr, torch.device("cuda"))
                rows.append(dict(holdout=h, model=arm, seed=seed, dims=F.shape[1], probe=probe_accuracy(F, lab, grp)))
                print(h, arm, seed, rows[-1]["probe"], flush=True)
    r = pd.DataFrame(rows); r.to_csv(A / "results" / "e04_probe.csv", index=False)
    print(r.groupby(["model", "holdout"]).probe.mean().groupby("model").mean().round(3).to_string())

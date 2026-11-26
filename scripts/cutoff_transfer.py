"""Label-free decision-cutoff transfer under scanner shift: data loading and methods.

A trained classifier gives a mitosis probability p. Its F1 cutoff t_src is chosen on labelled source-scanner
validation cells. On a new scanner the score distribution moves, so t_src lands in the wrong place (high AUROC,
low F1). Every method below picks a new cutoff for the new scanner from UNLABELLED new-scanner cells
(plus the labelled source validation cells). No target label is used, except by ORACLE (reference only).

Methods, all without tuning parameters:
  SRC     t_src unchanged (status quo)
  PRM     positive-rate matching: the cutoff at which the new-scanner share called positive equals the share
          called positive on source validation at t_src (quantile matching at the cutoff)
  LMM     logit moment matching: standardise new-scanner logits to the source-validation logit mean / SD,
          then apply t_src (parametric PRM)
  SLD     label-shift EM (Saerens, Latinne & Decaestecker, 2002): Platt-calibrate on source validation,
          estimate the new-scanner prevalence by EM, keep the source cutoff on the prior-adjusted posterior
  EF1     plug-in expected F1: source-Platt posteriors of the new-scanner cells used as soft labels; the cutoff
          that maximises the expected F1
  ORACLE  the F1-optimal cutoff on the labelled evaluation cells (upper bound, not a method)

Estimation / evaluation splits on the new scanner
  crossfit  case halves A/B (addon_common.target_halves, the same halves as cross-fitted AdaBN): the cutoff
            estimated on half A's unlabelled cells is applied to half B and vice versa; F1 over all cells
  official  estimated on the scanner's official 'train' images, evaluated on its 'test' images (Axel's split)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RES, PT = ROOT / "results", ROOT / "results" / "preds_tta"
FOLDS = ["CS2", "S360", "XR"]
METHODS = ["SRC", "PRM", "LMM", "SLD", "EF1"]
EPS = 1e-6

_META = None


def meta():
    global _META
    if _META is None:
        _META = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    return _META


def target_halves(img):
    """True for half A: sorted unique case ids, every other one from the first (addon_common.target_halves)."""
    cases = np.unique(img)
    return np.isin(img, cases[::2])


# ------------------------------------------------------------------ data
def load_ours(fold, tag, seed):
    """Source validation (p, y), t_src and the new-scanner table for one of our runs, or None if incomplete."""
    f = RES / "preds" / f"{fold}_{tag}_seed{seed}.csv"
    j = RES / f"{fold}_{tag}_seed{seed}.json"
    if not f.exists() or not j.exists():
        return None
    d = pd.read_csv(f)
    v = d[d.set == "val"]
    t = d[d.set == "test_out_all"][["row", "ann_id", "image_id", "y", "p"]].rename(columns={"p": "p_plain"})
    cf = PT / f"{fold}_{tag}_seed{seed}_cf.csv"
    if cf.exists():
        e = pd.read_csv(cf)
        t = t.merge(e[["row", "p_adabncf"]], on="row", how="left", validate="one_to_one")
    m = meta().set_index("ann_id")
    t["split"] = m.loc[t.ann_id.values, "split"].values
    assert (m.loc[t.ann_id.values, "scanner"].values == fold).all()
    assert (m.loc[t.ann_id.values, "y"].values == t.y.values).all()
    t["halfA"] = target_halves(t.image_id.values)
    return {"src_p": v.p.values.astype(float), "src_y": v.y.values.astype(int),
            "t_src": float(json.loads(j.read_text())["threshold"]), "tgt": t.reset_index(drop=True)}


def load_axel(model, fold, seed, folder=ROOT / "data" / "axel_predictions"):
    """Same structure for one of Axel's models (val = source validation; t_src = his best_threshold on val)."""
    d = pd.read_csv(folder / f"{model}.csv")
    d = d[(d.holdout == fold) & (d.seed == seed)]
    if d.empty:
        return None
    v = d[d.set == "val"]
    t = d[d.set.isin(["new_scanner_train", "new_scanner_test"])].rename(columns={"prob": "p_plain", "label": "y"})
    m = meta().set_index("ann_id")
    t = t.assign(image_id=m.loc[t.ann_id.values, "image_id"].values,
                 split=np.where(t.set == "new_scanner_test", "test", "train"))
    assert (m.loc[t.ann_id.values, "y"].values == t.y.values).all()
    t["halfA"] = target_halves(t.image_id.values)
    return {"src_p": v.prob.values.astype(float), "src_y": v.label.values.astype(int),
            "t_src": best_threshold(v.prob.values.astype(float), v.label.values.astype(int)),
            "tgt": t[["ann_id", "image_id", "y", "p_plain", "split", "halfA"]].reset_index(drop=True)}


# ------------------------------------------------------------------ metrics
def prf(p, y, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def best_threshold(p, y):
    """Max-F1 cutoff over the unique probabilities (Axel's train.best_threshold; ours uses the same rule)."""
    best = (0.0, 0.5)
    for t in np.unique(p):
        f1 = prf(p, y, t)[2]
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(p, y):
    z = logit(p)

    def nll(w):
        q = np.clip(sigmoid(w[0] * z + w[1]), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return minimize(nll, [1.0, 0.0], method="Nelder-Mead").x


# ------------------------------------------------------------------ methods
def cutoff(method, src_p, src_y, t_src, est_p, platt=None):
    """Cutoff on the probability scale for the new scanner, from unlabelled est_p."""
    if method == "SRC":
        return t_src
    if method == "PRM":
        r = np.mean(src_p >= t_src)                     # share called positive on source validation
        if r <= 0:
            return np.inf
        # smallest cutoff with share(est_p >= c) <= r  ->  the k-th largest value, k = round(r * n)
        k = max(1, int(round(r * len(est_p))))
        return float(np.sort(est_p)[::-1][k - 1])
    if method == "LMM":
        zs, zt = logit(src_p), logit(est_p)
        z = zt.mean() + zt.std() * (logit(t_src) - zs.mean()) / zs.std()
        return float(sigmoid(z))
    a, b = platt if platt is not None else fit_platt(src_p, src_y)
    q_est = sigmoid(a * logit(est_p) + b)
    if method == "SLD":
        pi_s = src_y.mean(); pi = pi_s
        for _ in range(200):
            w1, w0 = pi / pi_s * q_est, (1 - pi) / (1 - pi_s) * (1 - q_est)
            new = np.mean(w1 / (w1 + w0))
            if abs(new - pi) < 1e-8:
                break
            pi = new
        R = (pi * (1 - pi_s)) / (pi_s * (1 - pi))          # odds multiplier of the prior shift
        q_star = sigmoid(a * logit(np.array([t_src]))[0] + b)
        zq = np.log(q_star / (1 - q_star)) - np.log(R)       # calibrated logit whose adjusted posterior is q_star
        return float(sigmoid((zq - b) / a))
    if method == "EF1":
        order = np.argsort(-est_p); ps, qs = est_p[order], q_est[order]
        tp = np.cumsum(qs); k = np.arange(1, len(qs) + 1)
        ef1 = 2 * tp / (k + qs.sum())                        # E[TP] / ((#called + E[#pos]) / 2), plug-in
        return float(ps[int(np.argmax(ef1))])
    raise ValueError(method)


def evaluate(run, mode, method, split="crossfit"):
    """(precision, recall, F1, cutoff(s)) on the new scanner for one run, mode ('plain' / 'adabncf') and method.

    Source quantities always come from source validation under source BatchNorm (= plain); for 'adabncf' the
    new-scanner probabilities are the cross-fitted AdaBN ones."""
    t = run["tgt"]; col = f"p_{mode}"
    if col not in t.columns or t[col].isna().any():
        return None
    p, y = t[col].values.astype(float), t.y.values.astype(int)
    sp, sy, ts = run["src_p"], run["src_y"], run["t_src"]
    platt = fit_platt(sp, sy) if method in ("SLD", "EF1") else None
    if split == "crossfit":
        A = t.halfA.values
        cut = np.empty(len(p))
        if method == "ORACLE":                               # oracle per evaluation half, same structure
            for h in (A, ~A):
                cut[h] = best_threshold(p[h], y[h])
        else:
            for est, ev in ((A, ~A), (~A, A)):
                cut[ev] = cutoff(method, sp, sy, ts, p[est], platt)
        pred = p >= cut
    elif split == "official":
        est, ev = t.split.values == "train", t.split.values == "test"
        c = best_threshold(p[ev], y[ev]) if method == "ORACLE" else cutoff(method, sp, sy, ts, p[est], platt)
        p, y, pred = p[ev], y[ev], p[ev] >= c
        cut = np.array([c])
    else:
        raise ValueError(split)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), float(np.mean(cut))

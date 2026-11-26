"""EXPLORATORY (CUTOFF_PROTOCOL.md): does a referral guarantee certified on familiar scanners hold on a new scanner?

Referral layer, as in Axel's calibrate.py:
  - Platt scaling fitted on source validation.
  - Learn-then-Test fixed-sequence binomial test on the FAMILIAR-scanner test cells, risk 10 %, delta 0.10:
    the model decides alone on cells with confidence max(q, 1-q) >= c; the decided label is q > 0.5.
  - The certified confidence cutoff c is applied unchanged to the new scanner.
Reported per run: coverage (share decided alone) and realised error among decided cells, on familiar test cells and
on the new scanner (our runs: all 50 cases, plain and cross-fitted AdaBN; Axel's: official test images).
A run "violates" when its realised new-scanner error exceeds 10 %.

Writes results/cutoff_referral_runs.csv and prints the summary.
"""
import numpy as np, pandas as pd
from scipy.stats import binom
from cutoff_transfer import FOLDS, ROOT, RES, load_ours, fit_platt, sigmoid, logit

RISK, DELTA = 0.10, 0.10
OURS = ["baseline", "adv", "stain", "stain_adv", "all", "all_no_grl", "all_no_soft", "all_no_stain", "all_stain_default"]
AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]


def certify(conf, err, risk=RISK, delta=DELTA, start=0.25):
    srt = np.sort(conf)[::-1]; cut = np.inf
    for cov in np.arange(start, 1.0 + 1e-9, 0.01):
        thr = srt[max(1, int(round(cov * len(srt)))) - 1]
        dec = conf >= thr
        if binom.cdf(err[dec].sum(), dec.sum(), risk) <= delta:
            cut = thr
        else:
            break
    return cut


def layer(p_val, y_val, p_fam, y_fam):
    a, b = fit_platt(p_val, y_val)
    cal = lambda p: sigmoid(a * logit(p) + b)
    q = cal(p_fam); conf = np.maximum(q, 1 - q); err = (q > 0.5).astype(int) != y_fam
    return cal, certify(conf, err)


def score(cal, c, p, y):
    q = cal(p); conf = np.maximum(q, 1 - q); err = (q > 0.5).astype(int) != y
    dec = conf >= c
    return dec.mean(), (err[dec].mean() if dec.any() else np.nan)


def main():
    rows = []
    for f in FOLDS:
        for tag in OURS:
            for s in range(5):
                d = pd.read_csv(RES / "preds" / f"{f}_{tag}_seed{s}.csv")
                v, fam = d[d.set == "val"], d[d.set == "test_in"]
                run = load_ours(f, tag, s); t = run["tgt"]
                cal, c = layer(v.p.values, v.y.values, fam.p.values, fam.y.values)
                cf, ef = score(cal, c, fam.p.values, fam.y.values)
                for mode in ("plain", "adabncf"):
                    if f"p_{mode}" in t and not t[f"p_{mode}"].isna().any():
                        cn, en = score(cal, c, t[f"p_{mode}"].values, t.y.values)
                        rows.append({"source": "ours", "model": tag, "fold": f, "seed": s, "mode": mode,
                                     "cov_familiar": cf, "err_familiar": ef, "cov_new": cn, "err_new": en})
        for m in AXEL:
            d = pd.read_csv(ROOT / "data" / "axel_predictions" / f"{m}.csv")
            for s, g in d[d.holdout == f].groupby("seed"):
                v, fam, new = (g[g.set == k] for k in ("val", "familiar_test", "new_scanner_test"))
                cal, c = layer(v.prob.values, v.label.values, fam.prob.values, fam.label.values)
                cf, ef = score(cal, c, fam.prob.values, fam.label.values)
                cn, en = score(cal, c, new.prob.values, new.label.values)
                rows.append({"source": "axel", "model": m, "fold": f, "seed": int(s), "mode": "plain",
                             "cov_familiar": cf, "err_familiar": ef, "cov_new": cn, "err_new": en})
    r = pd.DataFrame(rows)
    r.round(5).to_csv(ROOT / "results" / "cutoff_referral_runs.csv", index=False)
    g = r.groupby(["source", "model", "mode"]).agg(
        runs=("err_new", "size"), cov_familiar=("cov_familiar", "mean"), err_familiar=("err_familiar", "mean"),
        cov_new=("cov_new", "mean"), err_new=("err_new", "mean"),
        violations=("err_new", lambda x: (x > RISK).mean()), worst=("err_new", "max"))
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())
    print("\nby held-out scanner (all models, plain):")
    print(r[r["mode"] == "plain"].groupby(["source", "fold"]).agg(cov_new=("cov_new", "mean"), err_new=("err_new", "mean"),
          violations=("err_new", lambda x: (x > RISK).mean())).round(3).to_string())


if __name__ == "__main__":
    main()

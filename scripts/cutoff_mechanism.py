"""POST-HOC EXPLORATORY (after the confirmatory results; suggested by the literature check): why does PRM fail
where it fails, and why does SLD fail?

1. Prevalence diagnostics per run: source-validation prevalence, true new-scanner prevalence, positive rate at
   t_src on source validation and on the new scanner, and SLD's estimated new-scanner prevalence.
2. Oracle-prior variants that are given the TRUE prevalence of the estimation cells (uses target labels, so a
   diagnostic only):
     PRM-op  positive-rate matching to r_s * pi_T / pi_S (the source positive rate rescaled by the prevalence ratio)
     SLD-op  label-shift adjustment with the true pi_T instead of the EM estimate
   If PRM-op repairs CS2, PRM's CS2 failure is its equal-prevalence assumption.
   If SLD-op still fails, SLD's failure is the fixed-p(x|y) assumption, not a bad prior estimate.
3. How much unlabelled new-scanner data PRM needs: the estimation half subsampled to 10 / 25 / 50 / 100 % of its
   cases (50 random draws per fraction).

Writes results/cutoff_mechanism_runs.csv and results/cutoff_dataamount.csv.
"""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, load_ours, load_axel, fit_platt, logit, sigmoid, cutoff, prf

GRID = ["baseline", "adv", "stain", "stain_adv", "all"]
AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]


def sld_prior(q, pi_s):
    pi = pi_s
    for _ in range(500):
        w1, w0 = pi / pi_s * q, (1 - pi) / (1 - pi_s) * (1 - q)
        new = np.mean(w1 / (w1 + w0))
        if abs(new - pi) < 1e-10:
            break
        pi = new
    return pi


def sld_cut(a, b, t_src, pi_s, pi):
    R = (pi * (1 - pi_s)) / (pi_s * (1 - pi))
    qs = sigmoid(a * logit(np.array([t_src]))[0] + b)
    return float(sigmoid((np.log(qs / (1 - qs)) - np.log(R) - b) / a))


def prm_cut(p, r):
    k = max(1, int(round(r * len(p))))
    return float(np.sort(p)[::-1][k - 1])


def one(source, name, f, s, run, mode, pairs, evalmask):
    t = run["tgt"]; p = t[f"p_{mode}"].values.astype(float); y = t.y.values.astype(int)
    sp, sy, ts = run["src_p"], run["src_y"], run["t_src"]
    a, b = fit_platt(sp, sy); pi_s = sy.mean(); r_s = np.mean(sp >= ts)
    cuts = {k: np.full(len(p), np.nan) for k in ("SRC", "PRM", "PRM-op", "SLD", "SLD-op")}
    est_prev, true_prev = [], []
    for est, ev in pairs:
        q = sigmoid(a * logit(p[est]) + b); pi_hat = sld_prior(q, pi_s); pi_true = y[est].mean()
        est_prev.append(pi_hat); true_prev.append(pi_true)
        cuts["SRC"][ev] = ts
        cuts["PRM"][ev] = prm_cut(p[est], r_s)
        cuts["PRM-op"][ev] = prm_cut(p[est], min(1.0, r_s * pi_true / pi_s))
        cuts["SLD"][ev] = sld_cut(a, b, ts, pi_s, pi_hat)
        cuts["SLD-op"][ev] = sld_cut(a, b, ts, pi_s, pi_true)
    row = {"source": source, "name": name, "fold": f, "seed": s, "mode": mode,
           "prev_source_val": pi_s, "prev_new_true": float(np.mean(true_prev)), "prev_new_SLD": float(np.mean(est_prev)),
           "posrate_source_at_tsrc": r_s, "posrate_new_at_tsrc": float(np.mean(p[evalmask] >= ts))}
    for k, c in cuts.items():
        pred = p >= c
        m = evalmask
        tp = int((pred & (y == 1) & m).sum()); fp = int((pred & (y == 0) & m).sum()); fn = int((~pred & (y == 1) & m).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        row[f"F1_{k}"] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return row


def main():
    rows = []
    for f in FOLDS:
        for tag in GRID:
            for s in range(5):
                run = load_ours(f, tag, s); t = run["tgt"]; A = t.halfA.values
                for mode in ("plain", "adabncf"):
                    rows.append(one("ours", tag, f, s, run, mode, [(A, ~A), (~A, A)], np.ones(len(t), bool)))
        for m in AXEL:
            for s in range(3):
                run = load_axel(m, f, s)
                if run is None:
                    continue
                t = run["tgt"]; tr, te = t.split.values == "train", t.split.values == "test"
                rows.append(one("axel", m, f, s, run, "plain", [(tr, te)], te))
    d = pd.DataFrame(rows)
    d.round(5).to_csv(ROOT / "results" / "cutoff_mechanism_runs.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["prev_source_val", "prev_new_true", "prev_new_SLD", "posrate_source_at_tsrc", "posrate_new_at_tsrc"]
    print(d.groupby(["source", "mode", "fold"])[cols].mean().round(3).to_string())
    f1 = [c for c in d.columns if c.startswith("F1_")]
    g = d.groupby(["source", "mode", "fold"])[f1].mean()
    print((g.sub(g.F1_SRC, axis=0)).round(3).to_string())
    g2 = d.groupby(["source", "mode"])[f1].mean()
    print((g2.sub(g2.F1_SRC, axis=0)).round(3).to_string())

    # data amount (our grid, plain, cross-fitted)
    rng = np.random.default_rng(7); out = []
    for f in FOLDS:
        for tag in GRID:
            for s in range(5):
                run = load_ours(f, tag, s); t = run["tgt"]; p = t.p_plain.values; y = t.y.values
                sp, ts = run["src_p"], run["t_src"]; r_s = np.mean(sp >= ts); img = t.image_id.values
                base = prf(p, y, ts)[2]
                for frac in (0.1, 0.25, 0.5, 1.0):
                    gains = []
                    for rep in range(1 if frac == 1.0 else 50):
                        cut = np.empty(len(p))
                        for est, ev in ((t.halfA.values, ~t.halfA.values), (~t.halfA.values, t.halfA.values)):
                            cases = np.unique(img[est]); k = max(1, int(round(frac * len(cases))))
                            pick = np.isin(img, rng.choice(cases, k, replace=False)) & est
                            cut[ev] = prm_cut(p[pick], r_s)
                        pred = p >= cut
                        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
                        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
                        gains.append((2 * pr * rc / (pr + rc) if pr + rc else 0) - base)
                    out.append({"fold": f, "tag": tag, "seed": s, "frac": frac, "cases_per_half": None,
                                "dF1_mean": float(np.mean(gains)), "dF1_p10": float(np.percentile(gains, 10))})
    q = pd.DataFrame(out)
    n_cases = {f: len(np.unique(load_ours(f, "baseline", 0)["tgt"].image_id)) // 2 for f in FOLDS}
    q["cases_per_half"] = [max(1, round(r.frac * n_cases[r.fold])) for r in q.itertuples()]
    q.round(5).to_csv(ROOT / "results" / "cutoff_dataamount.csv", index=False)
    print(q.groupby("frac")[["cases_per_half", "dF1_mean", "dF1_p10"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()

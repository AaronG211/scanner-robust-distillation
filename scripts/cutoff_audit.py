"""POST-HOC EXPLORATORY: a small labelled audit on the new scanner. Is it better spent on a prevalence estimate for
PRM, or on picking the F1 cutoff directly?

For k labelled new-scanner cells drawn at random from the estimation part (50 draws per k):
  DIRECT-k  F1-optimal cutoff on the k labelled cells (best_threshold)
  PRM-k     positive-rate matching to r_s * pi_k / pi_S, where pi_k = share of mitoses among the k cells; the rate
            is applied to ALL unlabelled estimation cells
compared with SRC and label-free PRM. Our grid tags (plain and AdaBN-cf; cross-fitted halves) and Axel's models
(official split). Writes results/cutoff_audit.csv.
"""
import numpy as np, pandas as pd
from cutoff_transfer import FOLDS, ROOT, load_ours, load_axel, best_threshold, prf
from cutoff_mechanism import prm_cut

GRID = ["baseline", "adv", "stain", "stain_adv", "all"]
AXEL = ["optimus_teacher", "student_final", "student_final_conf", "student_final_grl", "student_final_target"]
KS = (20, 50, 100, 200)


def f1_mask(p, y, cut, m):
    pred = p >= cut
    tp = int((pred & (y == 1) & m).sum()); fp = int((pred & (y == 0) & m).sum()); fn = int((~pred & (y == 1) & m).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def run_one(rng, source, name, f, s, run, mode, pairs, evalmask, reps=50):
    t = run["tgt"]; p = t[f"p_{mode}"].values.astype(float); y = t.y.values.astype(int)
    sp, sy, ts = run["src_p"], run["src_y"], run["t_src"]; r_s = np.mean(sp >= ts); pi_s = sy.mean()
    out = []
    src = f1_mask(p, y, np.full(len(p), ts), evalmask)
    cut = np.empty(len(p))
    for est, ev in pairs:
        cut[ev] = prm_cut(p[est], r_s)
    prm = f1_mask(p, y, cut, evalmask)
    for k in KS:
        d_direct, d_prmk = [], []
        for _ in range(reps):
            cd, cp = np.empty(len(p)), np.empty(len(p))
            for est, ev in pairs:
                idx = np.flatnonzero(est)
                pick = rng.choice(idx, min(k, len(idx)), replace=False)
                cd[ev] = best_threshold(p[pick], y[pick])
                cp[ev] = prm_cut(p[est], min(1.0, r_s * y[pick].mean() / pi_s))
            d_direct.append(f1_mask(p, y, cd, evalmask) - src); d_prmk.append(f1_mask(p, y, cp, evalmask) - src)
        out.append({"source": source, "name": name, "fold": f, "seed": s, "mode": mode, "k": k,
                    "dF1_PRM_labelfree": prm - src, "dF1_DIRECT_k": np.mean(d_direct), "dF1_PRM_k": np.mean(d_prmk),
                    "p10_DIRECT_k": np.percentile(d_direct, 10), "p10_PRM_k": np.percentile(d_prmk, 10)})
    return out


def main():
    rng = np.random.default_rng(11); rows = []
    for f in FOLDS:
        for tag in GRID:
            for s in range(5):
                run = load_ours(f, tag, s); t = run["tgt"]; A = t.halfA.values
                for mode in ("plain", "adabncf"):
                    rows += run_one(rng, "ours", tag, f, s, run, mode, [(A, ~A), (~A, A)], np.ones(len(t), bool))
        for m in AXEL:
            for s in range(3):
                run = load_axel(m, f, s)
                if run is None:
                    continue
                t = run["tgt"]; tr, te = t.split.values == "train", t.split.values == "test"
                rows += run_one(rng, "axel", m, f, s, run, "plain", [(tr, te)], te)
    d = pd.DataFrame(rows)
    d.round(5).to_csv(ROOT / "results" / "cutoff_audit.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.groupby(["source", "mode", "k"])[["dF1_PRM_labelfree", "dF1_DIRECT_k", "dF1_PRM_k", "p10_DIRECT_k", "p10_PRM_k"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()

"""Compare Axel's Optimus teacher / ConvNeXt-T students with our ResNet-18 runs on IDENTICAL cells.

Axel's predictions (Drive folder axel_models/predictions, copied to data/axel_predictions/) hold, per model,
holdout and seed, the probability of every cell in four sets: val, familiar_test, new_scanner_train,
new_scanner_test. The shared evaluation set is the held-out scanner's official TEST images
(Axel's `new_scanner_test` = our `test_out`), because his adapted students saw that scanner's train images.

Thresholds
  Axel: recomputed on his own `val` predictions with his train.best_threshold (max F1 over unique probabilities);
        reproduced numbers are checked against his reported means.
  Ours: the stored source-val thresholds (thr_plain / thr_tta), exactly as in summarize_addons.py.

Statistics: case-level bootstrap within each fold (cases = images of the official test split), the same
resample for every model, metrics averaged over seeds and then over the 3 folds. Seeds are not resampled.

Writes results/axel_compare_summary.csv and results/axel_compare_pairs.csv.
Usage: .venv/bin/python scripts/compare_axel.py [--B 2000]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_addons as sa  # noqa: E402  (loader for our stored predictions)

AX = ROOT / "data" / "axel_predictions"
FOLDS = ["CS2", "S360", "XR"]
AXEL = {"optimus_teacher": "Axel teacher: Optimus + LoRA (1.1B)",
        "student_final": "Axel student: ConvNeXt-T distilled",
        "student_final_conf": "Axel student: + confusion loss",
        "student_final_grl": "Axel student: + GRL",
        "student_final_target": "Axel student: + teacher-guided adaptation"}
OURS = [("baseline", "plain", "Aaron ResNet-18: baseline, as trained"),
        ("all", "plain", "Aaron ResNet all, as trained"),
        ("all", "tta", "Aaron ResNet all + TTA"),
        ("stain_adv", "adabncf_tta", "Aaron Stain+GRL + AdaBN + TTA"),
        ("all", "adabncf_tta", "Aaron ResNet all + AdaBN + TTA")]
SETTING = {"optimus_teacher": "no target data", "student_final": "no target data",
           "student_final_conf": "no target data", "student_final_grl": "no target data",
           "student_final_target": "unlabelled target", "plain": "no target data", "tta": "no target data",
           "adabncf_tta": "unlabelled target"}


def prf(p, y, t):                       # Axel's train.prf
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def best_threshold(p, y):               # Axel's train.best_threshold
    best = (0.0, 0.5)
    for t in np.unique(p):
        f1 = prf(p, y, t)[2]
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def auroc(p, y):
    r = pd.Series(p).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def weighted_metrics(p, y, thr, w):
    """F1 / precision / recall with per-cell weights w (bootstrap counts)."""
    pred = p >= thr
    tp = (w * (pred & (y == 1))).sum(); fp = (w * (pred & (y == 0))).sum(); fn = (w * (~pred & (y == 1))).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
    row_of = pd.Series(meta.index.values, index=meta.ann_id.values)
    runs = {}          # name -> fold -> list of (p, thr) on the fold's evaluation cells
    evalset = {}       # fold -> (ann_ids, y, image_id)

    for m in AXEL:
        d = pd.read_csv(AX / f"{m}.csv")
        for f in FOLDS:
            for s, g in d[d.holdout == f].groupby("seed"):
                v, t = g[g.set == "val"], g[g.set == "new_scanner_test"].sort_values("ann_id")
                thr = best_threshold(v.prob.values, v.label.values)
                ids = t.ann_id.values
                if f not in evalset:
                    mm = meta.loc[row_of[ids].values]
                    assert (mm.scanner.values == f).all(), "official test cells are not on the held-out scanner"
                    assert (mm.y.values == t.label.values).all(), "Axel's labels differ from data/patches_meta.csv"
                    evalset[f] = (ids, t.label.values.astype(int), mm.image_id.values)
                assert np.array_equal(ids, evalset[f][0]), f"{m} {f} s{s}: different evaluation cells"
                runs.setdefault(m, {}).setdefault(f, []).append((t.prob.values, thr))

    for tag, mode, name in OURS:
        for f in FOLDS:
            tgt = meta.index[meta.scanner == f].values
            y_all, img_all = meta.y.values[tgt].astype(bool), meta.image_id.values[tgt]
            pos = pd.Series(np.arange(len(tgt)), index=meta.ann_id.values[tgt])[evalset[f][0]].values
            for s in range(5):
                dd, _ = sa.load_target(f, tag, s, tgt, y_all, img_all)
                arr = sa.run_arrays(dd, mode)
                if arr is None:
                    raise SystemExit(f"missing {f} {tag} s{s} {mode}")
                p, thr = arr
                runs.setdefault(name, {}).setdefault(f, []).append((p[pos], float(thr[0])))

    # point estimates
    rows = []
    for name, by_fold in runs.items():
        per = {}
        for f in FOLDS:
            y = evalset[f][1]
            ms = np.array([[*prf(p, y, t), auroc(p, y)] for p, t in by_fold[f]])
            per[f] = ms.mean(0)
        mean = np.mean([per[f] for f in FOLDS], 0)
        key = name if name in AXEL else next(md for tg, md, nm in OURS if nm == name)
        rows.append({"model": AXEL.get(name, name), "setting": SETTING[key], "seeds": len(by_fold["CS2"]),
                     "F1": mean[2], "precision": mean[0], "recall": mean[1], "AUROC": mean[3],
                     **{f"F1_{f}": per[f][2] for f in FOLDS}})
    summ = pd.DataFrame(rows)

    # paired case-level bootstrap of F1 (same resample for every model)
    rng = np.random.default_rng(2027)
    boot = {name: np.zeros(a.B) for name in runs}
    for f in FOLDS:
        ids, y, img = evalset[f]
        cases, ci = np.unique(img, return_inverse=True)
        Wc = rng.multinomial(len(cases), np.full(len(cases), 1 / len(cases)), size=a.B).astype(float)
        W = Wc[:, ci]                                                           # B x cells
        for name, by_fold in runs.items():
            fb = np.zeros(a.B)
            for p, t in by_fold[f]:
                fb += np.array([weighted_metrics(p, y, t, W[b]) for b in range(a.B)])
            boot[name] += fb / len(by_fold[f]) / len(FOLDS)
    summ["F1_lo"] = [np.percentile(boot[n], 2.5) for n in runs]
    summ["F1_hi"] = [np.percentile(boot[n], 97.5) for n in runs]

    pairs = [("Aaron ResNet all, as trained", "student_final", "small models, no target data"),
             ("Aaron ResNet all + TTA", "student_final", "small models, no target data (ours with TTA)"),
             ("Aaron ResNet all + AdaBN + TTA", "student_final_target", "small models, unlabelled target data"),
             ("Aaron ResNet all + AdaBN + TTA", "optimus_teacher", "our best small model vs the teacher"),
             ("student_final_target", "optimus_teacher", "Axel's best student vs the teacher"),
             ("student_final", "optimus_teacher", "plain distilled student vs the teacher")]
    pr = []
    for x, z, what in pairs:
        dlt = boot[x] - boot[z]
        point = summ.loc[summ.model == AXEL.get(x, x), "F1"].item() - summ.loc[summ.model == AXEL.get(z, z), "F1"].item()
        pr.append({"comparison": f"{AXEL.get(x, x)}  minus  {AXEL.get(z, z)}", "what": what, "dF1": point,
                   "ci_lo": np.percentile(dlt, 2.5), "ci_hi": np.percentile(dlt, 97.5),
                   "P(first better)": (dlt > 0).mean()})
    pairs_df = pd.DataFrame(pr)

    summ.round(4).to_csv(ROOT / "results" / "axel_compare_summary.csv", index=False)
    pairs_df.round(4).to_csv(ROOT / "results" / "axel_compare_pairs.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print("evaluation cells (official test images of the held-out scanner):",
          {f: (len(evalset[f][0]), len(np.unique(evalset[f][2]))) for f in FOLDS}, "(cells, images)")
    print(summ.round(3).to_string(index=False))
    print()
    print(pairs_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

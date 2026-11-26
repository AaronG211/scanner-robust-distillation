"""Source-validation probabilities under TTA for existing checkpoints (needed to run the cutoff methods in TTA modes).

eval_tta_adabn.py computed these to pick thr_tta but did not store them. This script recomputes them with the same
functions and checks that best_threshold on them reproduces the stored thr_tta exactly.
Writes results/preds_valtta/{fold}_{tag}_seed{s}.csv (row, y, p_val_plain, p_val_tta). GPU (MPS); run alone.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from eval_tta_adabn import load_run, run_patches, val_test, hard_labels, predict, best_threshold, FOLDS
from addon_common import get_device, load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "preds_valtta"


def main(tags, seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(); cache = {}
    for fold in FOLDS:
        for tag in tags:
            for s in seeds:
                name = f"{fold}_{tag}_seed{s}"
                if (OUT / f"{name}.csv").exists():
                    continue
                run = load_run(fold, tag, s)
                if run is None:
                    print("missing", name); continue
                cfg, stored = run; t0 = time.time()
                key = cfg.get("patches", "default")
                if key not in cache:
                    cache = {key: run_patches(cfg)}
                X = cache[key]
                val, _ = val_test(stored); vi, yv = val.row.values, hard_labels(val)
                model = load_model(fold, tag, s, device, cfg)
                pv, pt = predict(model, X, vi, device), predict(model, X, vi, device, tta=True)
                ref = json.loads((ROOT / "results" / "preds_tta" / f"{name}.json").read_text())
                thr_p, thr_t = best_threshold(yv, pv)[0], best_threshold(yv, pt)[0]
                ok = abs(thr_p - ref["thr_plain"]) < 1e-6 and abs(thr_t - ref["thr_tta"]) < 1e-6
                pd.DataFrame({"row": vi, "y": yv, "p_val_plain": pv, "p_val_tta": pt}).to_csv(OUT / f"{name}.csv", index=False)
                print(f"{name}: thr_plain {thr_p:.4f}/{ref['thr_plain']:.4f} thr_tta {thr_t:.4f}/{ref['thr_tta']:.4f} "
                      f"{'OK' if ok else 'MISMATCH'} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["baseline", "adv", "stain", "stain_adv", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args(); main(a.tags, a.seeds)

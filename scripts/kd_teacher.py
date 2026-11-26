"""Teacher for self-contained distillation (no teammate files): the 5-seed ensemble of our "ResNet all" runs.

- Source-scanner cells (all source rows): mean over the 5 seeds of the 8-view TTA probability, source BatchNorm.
- New-scanner cells: mean over the 5 seeds of p_adabncf_tta (cross-fitted AdaBN + TTA, results/preds_tta/*_cf.csv).
  No new-scanner label is read. For a cell in half A these come from BatchNorm statistics of half B (and vice versa).

Inference cost of the teacher per cell: 5 models x 8 views = 40 ResNet-18 forward passes (plus the AdaBN pass).
Writes data/kd/{fold}_teacher_all.csv with columns row, p_teacher, domain ('source' | 'target').
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tta_adabn import predict, run_patches  # noqa: E402
from addon_common import FOLDS, get_device, load_model, run_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kd"; OUT.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(ROOT / "data" / "patches_meta.csv")
device = get_device()
for fold in FOLDS:
    src = np.where(meta.scanner.values != fold)[0]
    tgt = np.where(meta.scanner.values == fold)[0]
    ps, pt = [], []
    for s in range(5):
        cfg = run_config(fold, "all", s)
        model = load_model(fold, "all", s, device, cfg)
        X = run_patches(cfg)
        ps.append(predict(model, X, src, device, tta=True))
        cf = pd.read_csv(ROOT / "results" / "preds_tta" / f"{fold}_all_seed{s}_cf.csv").set_index("row")
        pt.append(cf.loc[tgt, "p_adabncf_tta"].values)
        print(f"{fold} seed {s} done", flush=True)
    df = pd.concat([pd.DataFrame({"row": src, "p_teacher": np.mean(ps, 0), "domain": "source"}),
                    pd.DataFrame({"row": tgt, "p_teacher": np.mean(pt, 0), "domain": "target"})])
    df.to_csv(OUT / f"{fold}_teacher_all.csv", index=False)
    y = meta.y.values
    print(f"{fold}: teacher mean |p - y| source {np.abs(df[df.domain == 'source'].p_teacher.values - y[src]).mean():.3f} "
          f"(diagnostic), share of source cells with 0.1 < p < 0.9: "
          f"{((np.mean(ps, 0) > 0.1) & (np.mean(ps, 0) < 0.9)).mean():.1%}", flush=True)

"""Teacher with the student's 8-view TTA (descriptive; NONINFERIORITY_PROTOCOL.md). Runs on the PC from C:\\midog.

For each foundation model (LoRA seed 0) and fold: single view (= view 0 of distill_teacher.py) and 8-view TTA (the
views of eval_extras.probs: identity and horizontal flip, each rotated by 0/90/180/270 degrees; mean of the sigmoid
outputs) on the training scanners' validation cells and on ALL new-scanner cells. Cutoffs: best F1 cutoff on the
validation cells, separately for single view and TTA. The single view is checked against the existing targetall cache.
Writes results/teacher_tta_cells.csv.gz and results/teacher_tta_summary.csv.
"""
import sys; sys.path.insert(0, r"C:\midog\repo\axel_model")
from pathlib import Path
import numpy as np, pandas as pd, torch
from train import make_splits, prf, auroc, best_threshold
from fm_lora import FMNet, FM, DEV, CROP

A = Path(r"C:\midog\repo\axel_model"); DIST = Path(r"C:\data\midog\distill")
d = dict(np.load(FM / "patches256.npz")); X, y = d["X"], d["y"].astype(int)
o = (X.shape[1] - CROP) // 2
fms = sys.argv[1:] or ["hopt0", "virchow2"]

@torch.no_grad()
def probs(model, idx, tta, bs=32):
    mean, std = model.mean.to(DEV), model.std.to(DEV); out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(X[idx[s:s + bs], o:o + CROP, o:o + CROP]).to(DEV).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        views = [torch.rot90(v, k, (2, 3)) for v in ([x, x.flip(3)] if tta else [x]) for k in (range(4) if tta else [0])]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.stack([torch.sigmoid(model.mitosis(model.features(v.to(torch.bfloat16))).squeeze(1).float()) for v in views]).mean(0)
        out.append(p.cpu().numpy())
    return np.concatenate(out)

cells, rows = [], []
for fm in fms:
    model = FMNet(adversarial=False, checkpointing=False, fm=fm).to(DEV).eval()
    for h in ("CS2", "S360", "XR"):
        S, _ = make_splits(d, h); u = S["test_unseen"]
        sd = torch.load(FM / "models" / f"{model.prefix}_lora_{h}_plain_e5_s0.pt", map_location=DEV, weights_only=True)
        missing = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("scanner.")}, strict=False).missing_keys
        assert not [k for k in missing if ".A" in k or ".B" in k or k.startswith("mitosis")], "LoRA / head weights missing"
        out = {"fm": fm, "holdout": h, "row": u}
        for name, tta in (("single", False), ("tta", True)):
            pv = probs(model, S["val"], tta); thr = best_threshold(pv, y[S["val"]]); pu = probs(model, u, tta)
            out[f"p_{name}"] = pu; out[f"thr_{name}"] = thr
            pr, rc, f1 = prf(pu, y[u], thr)
            rows.append(dict(fm=fm, holdout=h, view=name, f1=f1, precision=pr, recall=rc, auroc=auroc(pu, y[u]), cutoff=thr))
        c = DIST / f"teacher_{'' if fm == 'hopt0' else fm + '_'}{h}_targetall_s0.npz"
        t = np.load(c); assert np.array_equal(t["idx"], u)
        check = float(np.abs(1 / (1 + np.exp(-t["logit"][:, 0])) - out["p_single"]).max())
        cells.append(pd.DataFrame(out))
        print(fm, h, f"single {rows[-2]['f1']:.3f}  tta {rows[-1]['f1']:.3f}  max |single - cache| {check:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
pd.concat(cells).to_csv(A / "results" / "teacher_tta_cells.csv.gz", index=False)
r = pd.DataFrame(rows); r.to_csv(A / "results" / "teacher_tta_summary.csv", index=False)
print(r.pivot_table(index=["fm", "view"], columns="holdout", values="f1").assign(mean=lambda x: x.mean(1)).round(3).to_string())

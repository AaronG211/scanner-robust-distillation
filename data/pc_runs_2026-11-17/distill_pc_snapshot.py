"""Distillation: a small ConvNeXt-Tiny student (28M parameters) learns from the Optimus + LoRA teacher (1.1B).

Everything matches the Demo I ConvNeXt-Tiny runs (train.py: 144 px patches, random 128 crop, 20 epochs, batch 64,
lr 1e-4, checkpoint + cutoff chosen on validation, same splits), except what the student learns from:
  hard   (alpha)  the expert labels, as in Demo I
  soft   (1-alpha) the teacher's answer on the SAME view (same rotation / flip / stain jitter), softened by temperature T
  feat   (beta)   copy the teacher's 3,072-number description of the cell (cosine distance through a linear projection)
The control run (alpha=1, beta=0) sees exactly the same views but only the expert labels, so any difference to it is
the effect of the teacher. Views and teacher answers come from distill_teacher.py.

Reports F1 / AUROC on familiar and new scanners, the familiar-vs-new gap, and the independent scanner test on the
student's own features (does the student inherit the teacher's robustness?).

Options for the scanner shift:
  --grl G    scanner head behind a gradient reversal layer (max strength G, DANN warm-up; as in train.py's grl variant)
  --conf C   domain confusion instead (Tzeng et al., 2015; train.py's conf variant, Demo I's chosen method): the scanner
             head learns from detached features; the encoder is pushed to make the head's guess 50/50 (weight up to C)
  --target   teacher-guided adaptation: the student also copies the teacher on UNLABELED cells of the new scanner's
             'train' images (distill_teacher.py --target). No new-scanner label is used; cutoff and checkpoint are
             still chosen on the training scanners' validation set. Evaluate on test_unseen_official (the new
             scanner's 'test' images, never seen in any form).

Usage: python distill.py --holdout XR --alpha 0 --T 2 --beta 1 --seeds 0 1 2 --tag kd_feat
       ->  results/distill_results.csv (one row per seed)
"""
import argparse, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from train import Net, GradReverse, make_splits, prf, best_threshold, auroc, predict, MEAN, STD, CROP, BACKBONES, DEFAULT_LR
from probe import probe_accuracy
from distill_teacher import hed_apply, K
from crossfit import target_halves

HERE = Path(__file__).resolve().parent
DIST = Path("C:/data/midog/distill")

def student_batch(Xg, b, v, sa, sb):
    """Xg (N,144,144,3) uint8 on GPU; b batch rows; v (B,) view ids; sa/sb (B,3) stain params -> normalised (B,3,128,128)."""
    x = Xg[b].permute(0, 3, 1, 2).float() / 255
    S = x.shape[-1]; out = []
    for i in range(len(b)):
        ox, oy = np.random.randint(0, S - CROP + 1, 2)
        p = torch.rot90(x[i, :, oy:oy + CROP, ox:ox + CROP], int(v[i]) % 4, (1, 2))
        out.append(p.flip(2) if int(v[i]) % 8 >= 4 else p)
    x = torch.stack(out)
    stained = torch.as_tensor(v >= 8, device=x.device)
    if stained.any():
        x[stained] = hed_apply(x[stained], sa[stained], sb[stained])
    return (x - MEAN.to(x.device)) / STD.to(x.device)

@torch.no_grad()
def features(model, X, idx, device, bs=256):
    from train import prepare
    model.eval(); out = []
    for s in range(0, len(idx), bs):
        xb = torch.from_numpy(X[idx[s:s + bs]]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model.features(prepare(xb, False, False)).flatten(1).float().cpu().numpy())
    return np.concatenate(out)

def run(a, seed, d, T_cache, T_target, device):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = d["X"], d["y"].astype(np.int64)
    S, train_scn = make_splits(d, a.holdout)
    tr = S["train"]
    assert np.array_equal(T_cache["idx"], tr), "teacher cache does not match this fold's training cells"
    caches = [T_cache] + ([T_target] if T_target is not None else [])
    if T_target is not None:
        u = S["test_unseen"]
        want = target_halves(d, a.holdout)[a.target_half] if a.target_half else u[d["split"][u] == "train"]
        assert np.array_equal(T_target["idx"], want), "target cache mismatch"
    rows = np.concatenate([c["idx"] for c in caches]); n_src = len(tr)
    src = torch.zeros(len(rows), dtype=torch.bool, device=device); src[:n_src] = True
    Xg = torch.from_numpy(X[rows]).to(device)
    yg = torch.from_numpy(np.where(np.arange(len(rows)) < n_src, y[rows], 0)).float().to(device)   # target labels never used
    cat = lambda k: torch.from_numpy(np.concatenate([c[k] for c in caches])).to(device)
    zt = cat("logit")                                                                  # (N, K)
    ft = cat("feat") if a.beta > 0 else None                                           # (N, K, 3072) float16
    sa, sb = cat("stain_a"), cat("stain_b")
    pos_weight = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), device=device)
    dg = torch.from_numpy(np.array([train_scn.index(s) if s in train_scn else 0 for s in d["scanner"][rows]])).to(device)

    assert not (a.grl > 0 and a.conf > 0), "use --grl or --conf, not both"
    model = Net(a.grl > 0 or a.conf > 0, a.backbone).to(device)
    proj = nn.Linear(BACKBONES[a.backbone][1], ft.shape[-1]).to(device) if a.beta > 0 else None
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    total = math.ceil(len(rows) / a.bs) * a.epochs; step = 0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    yv = y[S["val"]]; best = {"score": -1.0}; t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(rows), device=device)
        for s in range(0, len(rows), a.bs):
            b = perm[s:s + a.bs]
            v = torch.randint(0, K, (len(b),), device=device) if a.views else torch.zeros(len(b), dtype=torch.long, device=device)
            xb = student_batch(Xg, b, v.cpu().numpy(), sa[b, v], sb[b, v])
            with torch.autocast("cuda", dtype=torch.float16):
                f = model.features(xb).flatten(1)
                z = model.mitosis(f).squeeze(1).float()
                loss = torch.zeros((), device=device)
                m = src[b]
                if a.source_labels:                                    # no Optimus anywhere: one loss term per cell
                    t_ = ~m                                            #   training-scanner cells: expert labels, exactly as the control
                    loss = loss + F.binary_cross_entropy_with_logits(z[m], yg[b][m], pos_weight=pos_weight, reduction="sum") / len(b)
                    if t_.any():                                       #   new-scanner cells: the teacher's answer, weighted as with --target
                        loss = loss + F.binary_cross_entropy_with_logits(z[t_] / a.T, torch.sigmoid(zt[b, v][t_] / a.T),
                                                                         reduction="sum") * a.T ** 2 / len(b)
                elif a.alpha > 0 and m.any():
                    loss = loss + a.alpha * F.binary_cross_entropy_with_logits(z[m], yg[b][m], pos_weight=pos_weight)
                if a.alpha < 1:
                    target = torch.sigmoid(zt[b, v] / a.T)
                    loss = loss + (1 - a.alpha) * F.binary_cross_entropy_with_logits(z / a.T, target) * a.T ** 2
                if a.beta > 0:
                    fm_ = m if a.target_soft_only else torch.ones_like(m)       # which cells copy the description
                    if fm_.any():
                        loss = loss + a.beta * (1 - F.cosine_similarity(proj(f[fm_].float()), ft[b, v][fm_].float(), dim=1)).mean()
                if a.grl > 0 and m.any():                                                  # scanner head on training-scanner cells
                    lam = a.grl * (2 / (1 + math.exp(-10 * step / total)) - 1)
                    loss = loss + F.cross_entropy(model.scanner(GradReverse.apply(f[m], lam)).float(), dg[b][m])
                if a.conf > 0 and m.any():                                                 # domain confusion
                    lam = a.conf * (2 / (1 + math.exp(-10 * step / total)) - 1)
                    fm = f[m]
                    dom_head = model.scanner(fm.detach())
                    frozen = {k: w.detach() for k, w in model.scanner.named_parameters()}
                    dom_enc = torch.func.functional_call(model.scanner, frozen, (fm,))
                    confusion = -F.log_softmax(dom_enc.float(), dim=1).mean()
                    loss = loss + F.cross_entropy(dom_head.float(), dg[b][m]) + lam * confusion
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step += 1
        pv = predict(model, X, S["val"], device); t = best_threshold(pv, yv); score = prf(pv, yv, t)[2]
        if score > best["score"]:
            best = {"score": score, "thr": t, "epoch": ep + 1, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    row = {"holdout": a.holdout, "backbone": a.backbone, "lr": a.lr, "tag": a.tag, "seed": seed, "alpha": a.alpha, "T": a.T, "beta": a.beta, "views": a.views,
           "grl": a.grl, "conf": a.conf, "target": a.target, "target_half": a.target_half, "target_soft_only": a.target_soft_only, "target_teacher": a.target_teacher, "source_labels": a.source_labels,
           "epochs": a.epochs, "best_epoch": best["epoch"], "threshold": round(best["thr"], 4), "val_f1": round(best["score"], 4)}
    for name in ["test_seen", "test_unseen", "test_unseen_official"]:
        idx = S[name]; p = predict(model, X, idx, device); pr, rc, f1 = prf(p, y[idx], best["thr"])
        row.update({f"{name}_f1": round(f1, 4), f"{name}_precision": round(pr, 4), f"{name}_recall": round(rc, 4),
                    f"{name}_auroc": round(auroc(p, y[idx]), 4)})
    row["f1_drop"] = round(row["test_seen_f1"] - row["test_unseen_f1"], 4)
    held = np.concatenate([S["val"], S["test_seen"]])
    lab = np.array([train_scn.index(s) for s in d["scanner"][held]])
    row["scanner_test"] = round(probe_accuracy(features(model, X, held, device), lab, d["image_id"][held]), 4)
    row["minutes"] = round((time.time() - t0) / 60, 2)
    if a.save:
        torch.save(best["state"], DIST / f"student_{a.holdout}_{a.backbone}_{a.tag}_s{seed}.pt")
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="XR", choices=["XR", "S360", "CS2"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alpha", type=float, default=0.0, help="weight of the expert labels (1 = control, no teacher answers)")
    ap.add_argument("--T", type=float, default=2.0, help="temperature for the teacher's answers")
    ap.add_argument("--beta", type=float, default=0.0, help="weight of copying the teacher's description")
    ap.add_argument("--no_views", dest="views", action="store_false", help="only view 0 (no rotation/stain views)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--backbone", choices=list(BACKBONES), default="convnext_tiny")
    ap.add_argument("--lr", type=float, default=None, help="default: train.py DEFAULT_LR for the backbone")
    ap.add_argument("--out", default="distill_results.csv", help="results/<out>")
    ap.add_argument("--teacher_seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--grl", type=float, default=0.0, help="max gradient-reversal strength for a scanner head (0 = off)")
    ap.add_argument("--conf", type=float, default=0.0, help="max domain-confusion weight (0 = off)")
    ap.add_argument("--target", action="store_true", help="teacher-guided adaptation on unlabeled new-scanner cells")
    ap.add_argument("--target_half", choices=["A", "B"], default=None,
                    help="cross-fitted adaptation: adapt on this half of the new scanner's images (crossfit.py); score the other half")
    ap.add_argument("--target_soft_only", action="store_true",
                    help="on new-scanner cells copy only the teacher's answer, not its description (feature loss on training-scanner cells only)")
    ap.add_argument("--target_teacher", default="optimus", choices=["optimus", "r18ens"],
                    help="who answers on the new-scanner cells with --target_half (r18ens: r18_teacher_cache.py)")
    ap.add_argument("--source_labels", action="store_true",
                    help="with --target_half: training-scanner cells learn from the expert labels only (as the control, --alpha 1 --beta 0); "
                         "only the new-scanner cells learn from the teacher's answers")
    a = ap.parse_args()
    assert not a.source_labels or (a.target_half and a.alpha == 1 and a.beta == 0), "--source_labels needs --target_half --alpha 1 --beta 0"
    if a.lr is None: a.lr = DEFAULT_LR[a.backbone]
    device = torch.device("cuda")
    d = dict(np.load(HERE / "cache" / "patches.npz"))
    T_cache = dict(np.load(DIST / f"teacher_{a.holdout}_s{a.teacher_seed}.npz"))
    T_target = dict(np.load(DIST / f"teacher_{a.holdout}_target_s{a.teacher_seed}.npz")) if a.target else None
    if a.target_half:
        assert not a.target, "use --target or --target_half, not both"
        full = dict(np.load(DIST / f"teacher_{a.holdout}_targetall_s{a.teacher_seed}.npz"))
        keep = np.isin(full["idx"], target_halves(d, a.holdout)[a.target_half])
        T_target = {k: (v[keep] if k in ("idx", "logit", "feat", "stain_a", "stain_b") else v) for k, v in full.items()}
        a.target = True
        if a.target_teacher != "optimus":                                     # control: someone else answers on the new scanner
            assert a.target_soft_only, "a teacher without Optimus descriptions needs --target_soft_only"
            alt = np.load(DIST / f"teacher_{a.holdout}_{a.target_teacher}{a.target_half}_s{a.teacher_seed}.npz")
            assert np.array_equal(alt["idx"], T_target["idx"]) and np.array_equal(alt["stain_a"], T_target["stain_a"]) \
                and np.array_equal(alt["stain_b"], T_target["stain_b"]), "control teacher cache does not match"
            T_target["logit"] = alt["logit"]; T_target["feat"] = np.zeros_like(T_target["feat"])   # never read (soft only)
    out = HERE / "results" / a.out
    for seed in a.seeds:
        row = run(a, seed, d, T_cache, T_target, device)
        print({k: row[k] for k in ["tag", "seed", "best_epoch", "val_f1", "test_seen_f1", "test_unseen_f1", "test_unseen_auroc",
                                   "test_unseen_official_f1", "f1_drop", "scanner_test", "minutes"]}, flush=True)
        df = pd.concat([pd.read_csv(out), pd.DataFrame([row])]) if out.exists() else pd.DataFrame([row])
        df.to_csv(out, index=False)

if __name__ == "__main__":
    main()

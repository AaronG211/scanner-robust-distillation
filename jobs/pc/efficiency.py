"""Efficiency (descriptive; NONINFERIORITY_PROTOCOL.md). Runs on the PC from C:\\midog, GPU otherwise idle.

Parameters and inference throughput (cells per second) of the students (128 px crop, fp16 autocast, batch 256: as in
train.predict / eval_extras.probs) and the four LoRA teachers (224 px crop, bf16, batch 64: as in distill_teacher.py).
Random inputs; 5 warm-up batches, then 30 timed batches, repeated 3 times (median). Writes results/efficiency.csv.
"""
import sys; sys.path.insert(0, r"C:\midog\repo\axel_model")
import time
from pathlib import Path
import numpy as np, pandas as pd, torch
from train import Net, CROP as SCROP
from fm_lora import FMNet, DEV, CROP as TCROP

A = Path(r"C:\midog\repo\axel_model")

def bench(fwd, x, n=30, warm=5):
    vals = []
    for _ in range(3):
        for _ in range(warm): fwd(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fwd(x)
        torch.cuda.synchronize(); vals.append(n * x.shape[0] / (time.perf_counter() - t))
    return float(np.median(vals))

rows = []
with torch.no_grad():
    for bb in ("resnet18", "convnext_tiny"):
        m = Net(False, bb).to(DEV).eval(); x = torch.randn(256, 3, SCROP, SCROP, device=DEV)
        def fwd(x, m=m):
            with torch.autocast("cuda", dtype=torch.float16):
                return m(x)[0]
        torch.cuda.reset_peak_memory_stats()
        rows.append(dict(model=bb, role="student", params_M=sum(p.numel() for p in m.parameters()) / 1e6, input_px=SCROP, batch=256,
                         cells_per_s=bench(fwd, x), peak_mem_GB=torch.cuda.max_memory_allocated() / 2**30))
        print(rows[-1], flush=True); del m, x; torch.cuda.empty_cache()
    for fm in ("hopt0", "virchow2", "uni", "uni2h"):
        m = FMNet(adversarial=False, checkpointing=False, fm=fm).to(DEV).eval(); x = torch.randn(64, 3, TCROP, TCROP, device=DEV)
        def fwd(x, m=m):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m.mitosis(m.features(x.to(torch.bfloat16)))
        torch.cuda.reset_peak_memory_stats()
        rows.append(dict(model=fm, role="teacher", params_M=sum(p.numel() for p in m.parameters()) / 1e6, input_px=TCROP, batch=64,
                         cells_per_s=bench(fwd, x), peak_mem_GB=torch.cuda.max_memory_allocated() / 2**30))
        print(rows[-1], flush=True); del m, x; torch.cuda.empty_cache()
r = pd.DataFrame(rows); r.to_csv(A / "results" / "efficiency.csv", index=False)
print(r.round(2).to_string(index=False))

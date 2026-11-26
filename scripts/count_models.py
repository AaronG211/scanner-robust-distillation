#!/usr/bin/env python3
"""How many networks the study trained, and the GPU training time (for the Study Design section).

Reads the run tables in data/pc_runs_2026-11-17/ (one row per trained network; `minutes` = training-loop time on the
RTX 3090 Ti). The three H-optimus-0 seed-0 LoRA teachers were fine-tuned by a teammate; their row lives in his
fm_lora_results.csv on the PC, which is not copied here (team-private file), so their count and minutes are entered by
hand below (variant plain, 5 epochs, seed 0: XR 26.3, S360 24.0, CS2 23.9 min).
Output: data/pc_runs_2026-11-17/model_counts.csv
"""
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "pc_runs_2026-11-17"
rows = []
def add(group, what, n, minutes, in_paper, note=""):
    rows.append(dict(group=group, what=what, n=int(n), gpu_minutes=round(float(minutes), 1), in_paper=in_paper, note=note))

# teachers
add("teacher", "H-optimus-0 LoRA, seed 0", 3, 26.3 + 24.0 + 23.9, True, "fine-tuned by a teammate")
m = pd.read_csv(D / "fm_lora_results_multi.csv")
for model, g in m.groupby("model"):
    add("teacher", f"{model}, seed 0", len(g), g.minutes.sum(), True)
s = pd.read_csv(D / "fm_lora_results_seeds.csv")
for model, g in s.groupby("model"):
    add("teacher", f"{model}, seeds 1-2", len(g), g.minutes.sum(), True)

# students
def arm(df, prefix_ok, label, student, in_paper=True, note=""):
    g = df[df.tag.isin(prefix_ok)]
    assert len(g), label
    add(f"student {student}", label, len(g), g.minutes.sum(), in_paper, note)
r0 = pd.read_csv(D / "resnet18_distill.csv"); rcf = pd.read_csv(D / "resnet18_distill_cf.csv")
r17 = pd.read_csv(D / "resnet18_distill_e17.csv"); rx2 = pd.read_csv(D / "resnet18_distill_exp2.csv")
rx2b = pd.read_csv(D / "resnet18_distill_exp2b.csv"); cx = pd.read_csv(D / "convnext_distill.csv")
arm(r0, ["r18_control"], "no-teacher student", "ResNet-18")
arm(r0, ["r18_kd"], "KD, H-optimus-0", "ResNet-18")
arm(r17, ["r18_kd_virchow2", "r18_kd_uni", "r18_kd_uni2h"], "KD, Virchow2 / UNI / UNI2-h", "ResNet-18")
arm(rcf, ["r18_kd_tgaA", "r18_kd_tgaB"], "KD + TGA, H-optimus-0 (halves A/B)", "ResNet-18")
arm(r17, [f"r18_kd_tga_{f}{h}" for f in ("virchow2", "uni", "uni2h") for h in "AB"], "KD + TGA, Virchow2 / UNI / UNI2-h (halves A/B)", "ResNet-18")
arm(rx2b, ["r18_free_tgaA", "r18_free_tgaB"], "no-teacher adaptation (halves A/B)", "ResNet-18")
arm(cx, ["cnx_control"], "no-teacher student", "ConvNeXt-T")
arm(cx, ["cnx_kd", "cnx_kd_virchow2"], "KD, H-optimus-0 / Virchow2", "ConvNeXt-T")
arm(cx, ["cnx_kd_tgaA", "cnx_kd_tgaB", "cnx_kd_tga_virchow2A", "cnx_kd_tga_virchow2B"], "KD + TGA, H-optimus-0 / Virchow2 (halves A/B)", "ConvNeXt-T")
for tags, label in ((["cnx_free_tgaA", "cnx_free_tgaB"], "no-teacher adaptation (halves A/B; Protocol C)"),
                    (["cnx_kd_t1", "cnx_kd_t2"], "KD, H-optimus-0 LoRA seeds 1-2 (Protocol F)"),
                    ([f"cnx_kd_tga_t{k}{h}" for k in (1, 2) for h in "AB"], "KD + TGA, H-optimus-0 LoRA seeds 1-2 (halves A/B; Protocol F)")):
    if cx.tag.isin(tags).any():                      # phase 6 (2026-11-26); present once convnext_distill.csv is refreshed
        arm(cx, tags, label, "ConvNeXt-T")
arm(r0, ["r18_kd_target"], "KD + target adaptation without cross-fitting (superseded)", "ResNet-18", False, "exploratory")
arm(rx2, ["r18_kd_tgar18A", "r18_kd_tgar18B", "r18_kd_tgasoftA", "r18_kd_tgasoftB"], "TGA variants (ResNet pseudo-labeller, soft-only)", "ResNet-18", False, "exploratory")

t = pd.DataFrame(rows); t.to_csv(D / "model_counts.csv", index=False)
pd.set_option("display.width", 200)
print(t.to_string(index=False)); print()
p = t[t.in_paper]
for grp in ("teacher", "student"):
    g = p[p.group.str.startswith(grp)]
    print(f"in the paper, {grp}s: {g.n.sum()} networks, {g.gpu_minutes.sum() / 60:.1f} GPU hours of training")
print(f"in the paper, total: {p.n.sum()} networks, {p.gpu_minutes.sum() / 60:.1f} GPU hours")
print(f"also trained but not in the paper: {t[~t.in_paper].n.sum()} networks, {t[~t.in_paper].gpu_minutes.sum() / 60:.1f} GPU hours")
print("test-time adaptation (TENT, not counted above): 2 students x 4 settings x 10 seeds x 3 scanners x 2 halves = 480 runs (120 in the primary setting)")

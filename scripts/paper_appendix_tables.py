#!/usr/bin/env python3
"""Appendix tables for the MIDL manuscript, generated from the result files (paper/manuscript/tables/app_*.tex).

  app_tests       every pre-specified contrast, its expectation and outcome
  app_perscanner  F1 per held-out scanner for the main arms
  app_orders      the two post-hoc layers in both orders and their Shapley credit (ResNet-18)
  app_bnswap      TGA students with their BatchNorm statistics swapped back to source statistics (E0.3)
  app_recaldiff   TGR minus each recalibration method, with intervals
  app_priors      label-free prevalence estimates against the true prevalence
  app_budget      the few-image curves
  app_teachers    teacher LoRA seeds, teacher 8-view TTA
  app_efficiency  parameters and throughput
  app_tent        TENT settings
  app_probe       scanner probe
  app_compute     trained networks and GPU time
Optional (only once the files exist): app_fc (Protocols F and C).
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "pc_runs_2026-11-17"
OUT = ROOT / "paper" / "manuscript" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
NAME = {"hopt0": "H-optimus-0", "virchow2": "Virchow2", "uni2h": "UNI2-h", "uni": "UNI"}
FOLDS = ["CS2", "S360", "XR"]
f3 = lambda x: f"{x:.3f}"
s3 = lambda x: (f"{x:+.3f}")
def ci(v, lo, hi): return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"
def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def table(label, caption, colspec, header, rows, size=r"\scriptsize", sep="3pt", extra_top="", stretch=None):
    body = "\n".join(rows)
    st = f"\\renewcommand{{\\arraystretch}}{{{stretch}}}\n" if stretch else ""
    return (f"\\begin{{table}}[htbp]\n\\floatconts{{{label}}}\n{{\\caption{{{caption}}}}}\n{{%\n{size}\n{st}\\setlength{{\\tabcolsep}}{{{sep}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{extra_top}{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n}}\n\\end{{table}}\n")
def write(name, s):
    (OUT / f"{name}.tex").write_text(s); print("wrote", name)

# ------------------------------------------------------------------ app_tests
rows = []
def add(fam, contrast, diff, lo, hi, holm, expect, outcome, interval=None):
    rows.append(f"{fam} & {contrast} & {interval or ci(diff, lo, hi)} & {pfmt(holm)} & {expect} & {outcome} \\\\")
KNOWN = "$^\\ast$"                                    # point estimate known when the protocol was written
def judge(kind, d, lo, hi, holm):
    if kind == "pos_sig":   return "met" if lo > 0 and holm < 0.05 else "not met"
    if kind == "neg_sig":   return "met" if hi < 0 and holm < 0.05 else ("not met (negative, n.s.)" if d < 0 else "not met")
    if kind == "ns_small":  return "met" if holm >= 0.05 and abs(d) <= 0.02 else "not met"
    if kind == "range":     return "met" if 0 <= d <= 0.03 else "not met"
    if kind == "small_ns":  return "met" if abs(d) <= 0.01 and holm >= 0.05 else ("partly (within $\\pm$0.01, but significant)" if abs(d) <= 0.01 else "not met")
    if kind == "dir_pos":   return "met" if lo > 0 and holm < 0.05 else ("not met (positive, n.s.)" if d > 0 else "not met")
    if kind == "dir_neg":   return "met" if hi < 0 and holm < 0.05 else ("not met (negative, n.s.)" if d < 0 else "not met")
    if kind == "geq":       return "met" if d >= 0 else "not met"
    return "--"
E = pd.read_csv(D / "r18_e17_boot.csv")
for _, r in E.iterrows():
    fm, what = r.contrast.split(": ")
    kind, exp = {"KD vs no teacher": ("ns_small", "small, n.s."), "KD+TGA vs KD": ("pos_sig", "$>0$, significant"),
                 "KD+TGA vs Optimus-free adaptation": ("range", "0 to +0.03")}[what]
    lab = {"KD vs no teacher": "KD $-$ no-teacher", "KD+TGA vs KD": "KD+TGA $-$ KD",
           "KD+TGA vs Optimus-free adaptation": "KD+TGA $-$ no-teacher adaptation"}[what]
    add("E1.7", f"R18, {NAME[fm]}: {lab}", r["diff"], r.lo, r.hi, r.holm, exp, judge(kind, r["diff"], r.lo, r.hi, r.holm))
A1 = pd.read_csv(D / "cnx_e17a_contrast.csv").iloc[0]      # E1.7 amendment 1: single two-sided contrast (scripts/cnx_boot.py)
add("E1.7a", "CNX, H-optimus-0: KD+TGA $-$ KD", A1["diff"], A1.lo, A1.hi, A1.p, "$>0$, significant", judge("pos_sig", A1["diff"], A1.lo, A1.hi, A1.p))
R = pd.read_csv(D / "recal_compare.csv")
EXP_R = {"TGR vs none": ("dir_pos", "$>0$"), "TGR vs source_rate": ("dir_pos", "$>0$"), "TGR vs em_prior": ("dir_pos", "$>0$"),
         "TGR vs teacher_pl": (None, "none"), "TGR vs adabn": ("dir_neg", "$<0$")}
LAB_R = {"none": "source cutoff", "source_rate": "source positive rate", "em_prior": "EM prior", "teacher_pl": "teacher pseudo-label cutoff", "adabn": "AdaBN"}
for _, r in R.iterrows():
    st = "R18" if r.student.startswith("r18") else "CNX"
    if r.family == 1:
        kind, exp = EXP_R[r.contrast]
        other = r.contrast.split(' vs ')[1]
        known = KNOWN if st == "R18" and other in ("none", "adabn") else ""   # RECAL_PROTOCOL.md: 0.725 / 0.762 / 0.772 known
        add("R1", f"{st} KD: TGR $-$ {LAB_R[other]}{known}", r["diff"], r.lo, r.hi, r.holm, exp, judge(kind, r["diff"], r.lo, r.hi, r.holm))
    else:
        n = int(r.contrast.split(", ")[1].split()[0])
        add("R2", f"R18 KD, {n} image{'s' if n > 1 else ''}: TGR $-$ AdaBN", r["diff"], r.lo, r.hi, r.holm,
            "$\\ge 0$" if n <= 2 else "none", judge("geq", r["diff"], r.lo, r.hi, r.holm) if n <= 2 else "--")
T = pd.read_csv(D / "tent_compare.csv")
for _, r in T.iterrows():
    st = "R18 KD" if r.student == "r18_kd" else "R18 no-teacher"
    other = r.contrast.split(" vs ")[1]
    if other == "AdaBN":
        add("T", f"{st}: TENT $-$ AdaBN", r["diff"], r.lo, r.hi, r.holm, "within $\\pm$0.01, n.s.", judge("small_ns", r["diff"], r.lo, r.hi, r.holm))
    else:
        lab = "KD+TGA" if other == "r18_kd_tga" else "no-teacher adaptation"
        add("T", f"{st}: TENT $-$ {lab}", r["diff"], r.lo, r.hi, r.holm, "$<0$, significant", judge("neg_sig", r["diff"], r.lo, r.hi, r.holm))
P = pd.read_csv(D / "prevalence_shift_contrasts.csv")
for _, r in P.iterrows():
    st = "R18" if r.student.startswith("r18") else "CNX"
    comp = "source positive rate" if "source_rate" in r.contrast else "source cutoff"
    add("P", f"{st} KD, $\\pi={r.pi:.1f}$: TGR $-$ {comp}", r["diff"], r.lo, r.hi, r.holm, "$>0$, significant", judge("pos_sig", r["diff"], r.lo, r.hi, r.holm))
for fn, fam in (("noninferiority_hopt0.csv", "N1"), ("noninferiority_virchow2.csv", "N2")):
    N = pd.read_csv(D / fn); N = N[N.kind == "pre-specified"]
    tea = "H-optimus-0" if fam == "N1" else "Virchow2"
    for _, r in N.iterrows():
        if r.test == "N2c":
            add("N2c", "CNX, Virchow2: KD+TGA $-$ KD (superiority)", r["diff"], r.lo, r.hi, r.holm, "$>0$, significant", judge("pos_sig", r["diff"], r.lo, r.hi, r.holm))
            continue
        arm = "KD+TGA" if "tga" in r.contrast else "KD+TGR"
        expect = {"N1a": "borderline", "N1b": "not non-inferior", "N2d": "same as N1a (borderline)", "N2e": "not non-inferior"}[r.test]
        ni = bool(r.non_inferior)
        outcome = ("non-inferior" if ni else "NI not established")
        outcome += {"N1a": "", "N1b": " (met)", "N2d": " (better than expected)", "N2e": " (met)"}[r.test] if r.test != "N1a" else ""
        known = KNOWN if fam == "N1" else ""          # NONINFERIORITY_PROTOCOL.md lists the N1 point estimates
        add(r.test, f"CNX vs {tea} teacher: {arm} $-$ teacher{known}", r["diff"], r.lo, r.hi, r.holm, expect, outcome)
fc = D / "teacher_seed_robustness.csv"
if fc.exists():
    F = pd.read_csv(fc); F = F[F.kind == "pre-specified"]
    for _, r in F.iterrows():
        ni = bool(r.non_inferior); verdict = "non-inferior" if ni else "NI not established"
        if r.test in ("F1", "F2"):                    # protocol expectation: F1 (teacher 0.821) fails, F2 (teacher 0.801) passes
            k = r.test[1]; exp_ni = k == "2"
            add(r.test, f"CNX vs H-optimus-0 LoRA seed {k}: KD+TGA $-$ teacher", r["diff"], r.lo, r.hi, r.holm,
                "non-inferior" if exp_ni else "NI not established", verdict + (" (met)" if ni == exp_ni else " (not met)"))
        else:                                         # key secondary: not in the Holm family; unadjusted one-sided p
            add("F key", "CNX vs H-optimus-0, mean over LoRA seeds 0--2: KD+TGA $-$ teacher", r["diff"], r.lo, r.hi,
                r.p_one_sided, "borderline", verdict + "; partly seen data",
                interval=f"{r['diff']:+.3f} [{r.lo:+.4f}, {r.hi:+.3f}]")   # lower limit to 4 decimals: it sits at the margin
cc = D / "cnx_free_control.csv"
if cc.exists():
    C = pd.read_csv(cc); C = C[C.kind == "pre-specified"]
    for _, r in C.iterrows():
        tea = "H-optimus-0" if r.test == "C1" else "Virchow2"
        add(r.test, f"CNX, {tea}: KD+TGA $-$ no-teacher adaptation", r["diff"], r.lo, r.hi, r.holm, "0 to +0.04", "met" if 0 <= r["diff"] <= 0.04 else "not met")
write("app_tests", table("tab:app-tests",
    "Every pre-specified contrast, in the order the protocols were written. R18 / CNX: ResNet-18 / ConvNeXt-T student. "
    "Differences are new-scanner F1 with 95\\% paired-bootstrap intervals; $p$ and ``met'' as defined in "
    "Appendix~\\ref{app:prereg}. $^\\ast$Point estimate known when the protocol was written. NI: non-inferiority at the margin 0.02.",
    "@{}l>{\\raggedright\\arraybackslash}p{5.0cm}ccp{1.8cm}>{\\raggedright\\arraybackslash}p{2.4cm}@{}", "Family & Contrast & Difference [95\\% CI] & Holm $p$ & Expected & Outcome", rows, size=r"\tiny", sep="2.5pt", stretch="0.92"))

# ------------------------------------------------------------------ app_perscanner
L = pd.read_csv(D / "ladder_recovered.csv")
def fold_val(st, fm, arm, h):
    r = L[(L.student == st) & (L.teacher == fm) & (L.kind == "f1_fold") & (L.quantity == f"{arm} | {h}")]
    return np.nan if r.empty else r.value.iloc[0]
def mean_val(st, fm, arm):
    r = L[(L.student == st) & (L.teacher == fm) & (L.kind == "f1") & (L.quantity == arm)]
    return np.nan if r.empty else r.value.iloc[0]
ARMLAB = [("ctrl", "No-teacher student"), ("kd", "KD"), ("kd_tgr", "KD + TGR"), ("kd_adabn_tgr", "KD + AdaBN + TGR"),
          ("kd_tga", "KD + TGA"), ("free_tga", "No-teacher adaptation"), ("teacher", "Teacher (seed 0)")]
rows = []
for st, fms in (("resnet18", ("hopt0", "virchow2", "uni2h", "uni")), ("convnext_t", ("hopt0", "virchow2"))):
    sname = "ResNet-18" if st == "resnet18" else "ConvNeXt-T"
    for j, fm in enumerate(fms):
        rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{sname}, {NAME[fm]} teacher}}}} \\")
        for arm, lab in ARMLAB:
            if arm in ("ctrl", "free_tga") and j > 0 and st == "resnet18":
                continue
            m = mean_val(st, fm, arm)
            if np.isnan(m):
                continue
            rows.append(f"\\quad {lab} & " + " & ".join(f3(fold_val(st, fm, arm, h)) for h in FOLDS) + f" & {f3(m)} \\\\")
write("app_perscanner", table("tab:app-perscanner", "New-scanner F1 per held-out scanner (mean over 10 student seeds) and mean over scanners. "
    "The no-teacher rows of the ResNet-18 student do not depend on the teacher and are shown once.",
    "@{}lcccc@{}", "Arm & CS2 & S360 & XR & Mean", rows))

# ------------------------------------------------------------------ app_orders
rows = []
for fm in ("hopt0", "virchow2", "uni2h", "uni"):
    g = lambda a: mean_val("resnet18", fm, a)
    comp = lambda q: L[(L.student == "resnet18") & (L.teacher == fm) & (L.kind == "component_f1") & (L.quantity == q)].iloc[0]
    th, bn = comp("threshold, Shapley"), comp("BN statistics, Shapley")
    rows.append(f"{NAME[fm]} & {s3(g('kd_tgr') - g('kd'))} & {s3(g('kd_adabn_tgr') - g('kd_tgr'))} & {s3(g('kd_adabn') - g('kd'))} & "
                f"{s3(g('kd_adabn_tgr') - g('kd_adabn'))} & {ci(th.value, th.lo, th.hi)} & {ci(bn.value, bn.lo, bn.hi)} \\\\")
write("app_orders", table("tab:app-orders",
    "The two post-hoc layers of the ResNet-18 KD student in both orders (F1 gain of each step) and their order-free "
    "Shapley credit with 95\\% intervals. AdaBN also moves the operating point, so the threshold gain shrinks when AdaBN comes first.",
    "@{}lcccccc@{}", "Teacher & \\makecell{TGR\\\\first} & \\makecell{then\\\\AdaBN} & \\makecell{AdaBN\\\\first} & \\makecell{then\\\\TGR} & "
    "Shapley: threshold & Shapley: statistics", rows))

# ------------------------------------------------------------------ app_bnswap
S = pd.read_csv(D / "e03_bnswap.csv"); rng = np.random.default_rng(20261205); rows = []
f1c = lambda tp, fp, fn: 2 * tp / np.maximum(2 * tp + fp + fn, 1)
for arm in sorted(S.arm.unique()):
    for h in FOLDS:
        g = S[(S.arm == arm) & (S.holdout == h)].groupby("seed")[[c for c in S.columns if c.startswith(("tp_", "fp_", "fn_"))]].sum()
        o = f1c(g.tp_orig, g.fp_orig, g.fn_orig).values; w = f1c(g.tp_src_rethr, g.fp_src_rethr, g.fn_src_rethr).values
        d = o - w; bs = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
        rows.append(f"{'KD + TGA' if 'kd' in arm else 'No-teacher adaptation'} & {h} & {f3(o.mean())} & {f3(w.mean())} & "
                    f"{d.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] \\\\")
write("app_bnswap", table("tab:app-bnswap",
    "ResNet-18 students retrained on the new scanner's fit half, evaluated with their own BatchNorm statistics and with "
    "statistics re-estimated on source cells only (cutoff re-chosen on source validation). Drop: mean over 10 seeds, "
    "interval from a seed bootstrap. On S360 and XR, where retraining helps, the statistics explain little of its gain; on "
    "CS2 weights and statistics have co-adapted, so the swap is not interpretable as the statistics' contribution.",
    "@{}llccc@{}", "Student & Scanner & Own statistics & Source statistics & Drop [95\\% CI]", rows))

# ------------------------------------------------------------------ app_recaldiff and app_priors
Dd = pd.read_csv(D / "recal_baselines_diffs.csv")
MLAB = [("none", "Source cutoff"), ("upa", "UPA"), ("source_rate", "Source positive rate"), ("source_bias", "One bias, source mean"),
        ("em_prior", "EM prior"), ("em_bcts", "EM + BCTS"), ("bbse", "BBSE"), ("teacher_bt", "Bias + temp. (teacher)"),
        ("teacher_qmap", "Quantile map (teacher)"), ("teacher_rate", "Teacher positive rate"), ("teacher_pl", "Teacher pseudo-label cut."),
        ("oracle", "Labelled cutoff"), ("teacher_alone", "Teacher alone")]
rows = []
for m, lab in MLAB:
    cells = []
    for st in ("r18_kd", "cnx_kd"):
        for pi in ("natural", "0.2", "0.6"):
            r = Dd[(Dd.student == st) & (Dd.pi == pi) & (Dd.contrast == f"TGR minus {m}")].iloc[0]
            cells.append(rf"\makecell{{{r['diff']:+.3f}\\[-1pt]{{\tiny[{r.lo:+.3f}, {r.hi:+.3f}]}}}}")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
write("app_recaldiff", table("tab:recal-diff", "TGR minus each recalibration method (F1, 95\\% paired-bootstrap interval; descriptive). A separate bootstrap run from \\tableref{tab:app-tests}, so shared contrasts can differ in the third decimal of an interval limit.",
    "@{}lcccccc@{}", "Method & \\multicolumn{3}{c}{ResNet-18 KD: natural, 0.2, 0.6} & \\multicolumn{3}{c}{ConvNeXt-T KD: natural, 0.2, 0.6}", rows, size=r"\tiny", sep="1.5pt"))
Pr = pd.read_csv(D / "prior_estimates_check.csv").groupby(["student", "fold"]).mean(numeric_only=True)
rows = []
for (st, h), r in Pr.iterrows():
    rows.append(f"{'ResNet-18' if st == 'r18_kd' else 'ConvNeXt-T'} & {h} & {r.true:.2f} & {r.student_pos_rate:.2f} & {r.em:.2f} & {r.em_bcts:.2f} & {r.bbse:.2f} & {r.teacher_pos_rate:.2f} \\\\")
write("app_priors", table("tab:app-priors",
    "Label-free estimates of the new scanner's mitosis prevalence (mean over 10 student seeds) against the truth. The "
    "prior-shift estimators follow the student's own shifted positive rate and amplify it (S360, XR).",
    "@{}llcccccc@{}", "KD student & Scanner & True & \\makecell{Student positive\\\\rate} & EM & EM + BCTS & BBSE & \\makecell{Teacher positive\\\\rate}", rows))

# ------------------------------------------------------------------ app_budget
lab_ = pd.read_csv(D / "e02_labelled_budget.csv"); ada = pd.read_csv(D / "e02_adabn_budget.csv")
def bmean(df, arm, col):
    return df[df.arm == arm].groupby(["n_images", "holdout"])[col].mean().groupby("n_images").mean()
rr, ra, rl = bmean(lab_, "r18_kd", "f1_tgr"), bmean(ada, "r18_kd", "f1_adabn"), bmean(lab_, "r18_kd", "f1_labelled")
cr, cl = bmean(lab_, "cnx_kd", "f1_tgr"), bmean(lab_, "cnx_kd", "f1_labelled")
rows = [f"{n} & {f3(rr[n])} & {f3(ra[n])} & {f3(rl[n])} & {f3(cr[n])} & {f3(cl[n])} \\\\" for n in (1, 2, 5, 10, 25)]
base_r = lab_[lab_.arm == "r18_kd"].groupby("holdout").f1_none.mean().mean(); base_c = lab_[lab_.arm == "cnx_kd"].groupby("holdout").f1_none.mean().mean()
write("app_budget", table("tab:app-budget",
    f"New-scanner F1 with $n$ images per half from the new scanner (mean over folds, seeds and 10 image draws; the same draws "
    f"for every method). TGR and AdaBN use the images unlabelled; the labelled cutoff uses their labels. Source cutoff: "
    f"{base_r:.3f} (ResNet-18), {base_c:.3f} (ConvNeXt-T).",
    "@{}cccccc@{}", "$n$ & R18: TGR & R18: AdaBN & R18: labelled cutoff & CNX: TGR & CNX: labelled cutoff", rows, size=r"\footnotesize", sep="5pt"))

# ------------------------------------------------------------------ app_teachers, app_efficiency
TS = pd.read_csv(D / "teacher_seeds_summary.csv"); TS = TS[TS.set == "all50"]
rows = []
for fm in ("hopt0", "virchow2", "uni2h", "uni"):
    g = TS[TS.fm == fm]
    f = g.groupby("seed").f1.mean(); a = g.groupby("seed").auroc.mean()
    rows.append(f"{NAME[fm]} & " + " & ".join(f3(f[s]) for s in (0, 1, 2)) + f" & {f3(f.mean())} & {f3(a.min())}--{f3(a.max())} \\\\")
TT = pd.read_csv(D / "teacher_tta_summary.csv").groupby(["fm", "view"]).f1.mean()
rows.append(r"\midrule")
for fm in ("hopt0", "virchow2"):
    rows.append(f"{NAME[fm]}, 8-view TTA (seed 0) & {f3(TT[(fm, 'tta')])} & & & & \\\\")
write("app_teachers", table("tab:app-teachers",
    "Teachers on the unseen scanner: F1 for three LoRA seeds (mean over the three held-out scanners), their mean, and the "
    "range of AUROC over seeds; below, seed 0 with eight-view test-time augmentation and a TTA cutoff from source validation. Seed-0 values are "
    "re-evaluated from the saved LoRA weights and can differ in the third decimal from the cached predictions used elsewhere "
    "(UNI: 0.766 here, 0.767 in \\tableref{tab:main}).",
    "@{}lccccc@{}", "Teacher & Seed 0 & Seed 1 & Seed 2 & Mean & AUROC range", rows, size=r"\footnotesize", sep="5pt"))
Ef = pd.read_csv(D / "efficiency.csv").set_index("model")
rows = []
for m, lab in (("resnet18", "ResNet-18 student"), ("convnext_tiny", "ConvNeXt-T student"), ("hopt0", "H-optimus-0 teacher"),
               ("virchow2", "Virchow2 teacher"), ("uni2h", "UNI2-h teacher"), ("uni", "UNI teacher")):
    r = Ef.loc[m]
    rows.append(f"{lab} & {r.params_M:,.1f} & {int(r.input_px)} & {int(r.batch)} & {r.cells_per_s:,.0f} & {r.peak_mem_GB:.2f} \\\\")
write("app_efficiency", table("tab:app-efficiency",
    "Inference cost on one RTX 3090 Ti: random inputs at the resolution used in training, mixed precision (fp16 students, "
    "bf16 teachers), 5 warm-up batches, then 30 timed batches, median of 3 repetitions. Parameters include LoRA and heads.",
    "@{}lrrrrr@{}", "Model & Params (M) & Input (px) & Batch & Cells/s & Peak memory (GB)", rows, size=r"\footnotesize", sep="5pt"))

# ------------------------------------------------------------------ app_tent
TN = pd.read_csv(D / "tent_table.csv")
rows = []
for (arm, lr, ep), g in TN.groupby(["arm", "lr", "epochs"]):
    v = g.set_index("method")["mean"]
    rows.append(f"{'KD' if arm == 'r18_kd' else 'No-teacher'} & {lr:g} & {ep} & {f3(v['adabn'])} & {f3(v['tent'])} & {f3(v['tent_tgr'])} \\\\")
write("app_tent", table("tab:app-tent", "TENT on the ResNet-18 students (new-scanner F1, mean over seeds and scanners). Primary setting: "
    "learning rate $10^{-3}$, one epoch; the others are a descriptive sensitivity analysis.",
    "@{}lccccc@{}", "Student & lr & Epochs & AdaBN & AdaBN + TENT & + TGR", rows, size=r"\footnotesize", sep="5pt"))

# ------------------------------------------------------------------ app_probe
PB = pd.read_csv(D / "e04_probe.csv").groupby("model").agg(dims=("dims", "first"), acc=("probe", "mean"))
order = ["teacher_hopt0", "teacher_virchow2", "teacher_uni2h", "teacher_uni", "teacher_hopt0_pca512", "teacher_virchow2_pca512",
         "teacher_uni2h_pca512", "teacher_uni_pca512", "r18_control", "r18_kd", "r18_kd_tgaA"]
lab = {"r18_control": "ResNet-18 no-teacher", "r18_kd": "ResNet-18 KD (H-optimus-0)", "r18_kd_tgaA": "ResNet-18 KD + TGA (H-optimus-0, half A)"}
rows = []
for m in [o for o in order if o in PB.index]:
    nm = lab.get(m) or (NAME[m.split("_")[1]] + (" (PCA 512)" if "pca" in m else ""))
    rows.append(f"{nm} & {int(PB.loc[m, 'dims'])} & {PB.loc[m, 'acc']:.3f} \\\\")
write("app_probe", table("tab:app-probe", "Linear probe for the training scanner on source-cell features (image-grouped cross-validation, "
    "two classes, mean over folds and seeds). All models encode the scanner almost perfectly, so robustness does not mean "
    "that scanner information is absent.", "@{}lcc@{}", "Features & Dimension & Probe accuracy", rows, size=r"\footnotesize", sep="5pt"))

# ------------------------------------------------------------------ app_compute
MC = pd.read_csv(D / "model_counts.csv")
rows = [f"{r.group.replace('student ', '')} & {r.what} & {r.n} & {r.gpu_minutes / 60:.2f} & {'yes' if r.in_paper else 'no'} \\\\" for _, r in MC.iterrows()]
write("app_compute", table("tab:app-compute", "Networks trained for the study and their training-loop time on one RTX 3090 Ti "
    "(excluding teacher caching and evaluation). Not in the paper: two superseded exploratory variants.",
    "@{}llrrc@{}", "Group & Networks & $n$ & GPU hours & In paper", rows, size=r"\tiny", sep="3pt"))

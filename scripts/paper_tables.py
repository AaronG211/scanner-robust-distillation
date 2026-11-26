#!/usr/bin/env python3
"""Draft paper tables as LaTeX (booktabs, makecell, amssymb), generated from the result files so the numbers cannot drift.

  Table 1  the repair ladder and what each step needs on the new scanner
  Table 2  new-scanner F1 with 95% intervals for every step x teacher x student          (ladder_recovered.csv)
  Table 3  label-free recalibration baselines at the natural and shifted prevalence      (recal_baselines_table.csv)

Writes paper/tables/table{1,2,3}.tex and paper/tables/preview.tex (stand-alone, for checking the layout).
Captions are drafts to be rewritten by the authors.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "pc_runs_2026-11-17"
OUT = ROOT / "paper" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
NAME = {"hopt0": "H-optimus-0", "virchow2": "Virchow2", "uni2h": "UNI2-h", "uni": "UNI"}
Y, N = r"\checkmark", r"--"

# ------------------------------------------------------------------ Table 1
t1 = r"""\begin{table}[t]
\centering
\caption{The repair ladder. Each step acts on one layer of the student and needs different things on the unseen
scanner. All label-free steps are fitted on the unlabelled cells of one half of the new-scanner images and scored on the
other half, then the halves are swapped (cross-fitting).}
\label{tab:ladder}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llcccl@{}}
\toprule
 & & \multicolumn{3}{c}{Needs on the new scanner} & \\
\cmidrule(lr){3-5}
Step & Layer & Unlabelled images & Teacher & Retraining & Students \\
\midrule
No-teacher student (0\%%) & -- & %(N)s & %(N)s & %(N)s & both \\
KD, source cutoff & -- & %(N)s & %(N)s & %(N)s & both \\
\quad + TGR & operating point & %(Y)s\ (about 5) & probabilities & %(N)s & both \\
\quad + AdaBN & BN statistics & %(Y)s & %(N)s & %(N)s & ResNet-18 \\
\quad + TENT & BN scale and shift & %(Y)s & %(N)s & at test time & ResNet-18 \\
KD + TGA & representation & %(Y)s\ (25 per half) & soft labels & %(Y)s & both \\
No-teacher adaptation & representation & %(Y)s\ (25 per half) & %(N)s & %(Y)s & both \\
Labelled cutoff (upper bound) & operating point & labelled & %(N)s & %(N)s & both \\
Teacher (100\%%) & -- & %(N)s & -- & %(N)s & -- \\
\bottomrule
\end{tabular}
\end{table}
""" % {"Y": Y, "N": N}
(OUT / "table1.tex").write_text(t1)

# ------------------------------------------------------------------ Table 2
L = pd.read_csv(D / "ladder_recovered.csv")
cols = [("resnet18", f) for f in ("hopt0", "virchow2", "uni2h", "uni")] + [("convnext_t", f) for f in ("hopt0", "virchow2")]
def cell(st, fm, q, kind="f1", pct=False, mark=""):
    r = L[(L.student == st) & (L.teacher == fm) & (L.kind == kind) & (L.quantity == q)]
    if r.empty:
        return N
    v, lo, hi = r.iloc[0][["value", "lo", "hi"]]
    if pct:
        return rf"\makecell{{{100 * v:.0f}\%{mark}\\{{\scriptsize[{100 * lo:.0f}, {100 * hi:.0f}]}}}}"
    f = lambda x: f"{x:.3f}"[1:] if x >= 0 else f"{x:.3f}"
    return rf"\makecell{{{v:.3f}\\{{\scriptsize[{f(lo)}, {f(hi)}]}}}}"
TS = pd.read_csv(D / "teacher_seeds_summary.csv"); TS = TS[TS.set == "all50"]
tmean = TS.groupby(["fm", "seed"]).f1.mean().groupby("fm").agg(["mean", "min", "max"])
rows2 = [("No-teacher student", "ctrl"), ("Distillation (KD)", "kd"), (r"\quad + TGR", "kd_tgr"),
         (r"\quad + AdaBN + TGR", "kd_adabn_tgr"),
         ("KD + TGA (retrained)", "kd_tga"), ("No-teacher adaptation", "free_tga"), ("Teacher (seed 0)", "teacher")]
body = []
for lab, q in rows2:
    body.append(lab + " & " + " & ".join(cell(st, fm, q) for st, fm in cols) + r" \\")
    if q == "kd_tga":
        pass
body.append(r"\midrule")
body.append(r"Share recovered by KD + TGA & " + " & ".join(
    cell(st, fm, "kd_tga", kind="share", pct=True, mark=r"$^\dagger$" if fm == "uni" else "") for st, fm in cols) + r" \\")
t2 = r"""\begin{table}[t]
\centering
\caption{F1 on the unseen scanner (mean over 10 student seeds, then over the three held-out scanners; 95\%% interval from
a paired bootstrap over images and seeds). Each student uses its own teacher for TGR. --: not applicable (ConvNeXt-T has
no BatchNorm) or not run. Last row: (student $-$ no-teacher student) / (teacher $-$ no-teacher student).
$^\dagger$UNI's advantage over the no-teacher student is only %(uadv)s F1, so its share is unstable; its seed-0 LoRA
teacher chose an unusually low cutoff on the source scanners.}
\label{tab:main}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccc@{\hspace{8pt}}cc@{}}
\toprule
 & \multicolumn{4}{c}{ResNet-18 student (11 M)} & \multicolumn{2}{c}{ConvNeXt-T student (28 M)} \\
\cmidrule(lr){2-5}\cmidrule(l){6-7}
Teacher & %(heads)s \\
\midrule
%(body)s
\bottomrule
\end{tabular}
\end{table}
""" % {"heads": " & ".join(NAME[fm] for _, fm in cols), "body": "\n".join(body),
       "uadv": f"{L[(L.student == 'resnet18') & (L.teacher == 'uni') & (L.quantity == 'teacher advantage (teacher - ctrl)')].value.iloc[0]:.3f}"}
(OUT / "table2.tex").write_text(t2)

# ------------------------------------------------------------------ Table 3
B = pd.read_csv(D / "recal_baselines_table.csv").set_index(["student", "pi", "method"])["mean"]
groups = [("No recalibration", [("Source cutoff", "none", "")]),
          ("Teacher-free (reference: the student's source-validation outputs)",
           [("UPA, quantile alignment \\citep{roschewitz2023automatic}", "upa", ""),
            ("Source positive rate", "source_rate", ""),
            ("One bias, source mean probability", "source_bias", ""),
            ("EM prior adjustment", "em_prior", ""),
            ("EM + bias-corrected temperature scaling", "em_bcts", "$^\\ddagger$"),
            ("BBSE", "bbse", "$^\\ddagger$")]),
          ("Teacher-referenced (reference: the teacher on the same unlabelled cells)",
           [("TGR: one bias (ours)", "tgr", ""),
            ("Bias + temperature", "teacher_bt", ""),
            ("Quantile map onto the teacher", "teacher_qmap", ""),
            ("Teacher positive rate", "teacher_rate", ""),
            ("Cutoff from teacher pseudo-labels", "teacher_pl", "")]),
          ("Uses new-scanner labels", [("Labelled cutoff (upper bound)", "oracle", "")]),
          ("Reference", [("Teacher alone", "teacher_alone", "")])]
pis = ["natural", "0.2", "0.6"]
label_free = {"none", "upa", "source_rate", "source_bias", "em_prior", "em_bcts", "bbse", "tgr", "teacher_bt", "teacher_qmap", "teacher_rate", "teacher_pl"}
best = {(st, pi): max(B[(st, pi, m)] for m in label_free) for st in ("r18_kd", "cnx_kd") for pi in pis}
body = []
for gi, (gname, items) in enumerate(groups):
    if gi:
        body.append(r"\addlinespace[2pt]")
    body.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{gname}}}}} \\")
    for lab, m, mark in items:
        vals = []
        for st in ("r18_kd", "cnx_kd"):
            for pi in pis:
                v = B[(st, pi, m)]
                s_ = f"{v:.3f}"
                vals.append(rf"\textbf{{{s_}}}" if m in label_free and s_ == f"{best[(st, pi)]:.3f}" else s_)
        body.append(r"\quad " + lab + mark + " & " + " & ".join(vals) + r" \\")
t3 = r"""\begin{table}[t]
\centering
\caption{Recalibrating the operating point without new-scanner labels (KD students of H-optimus-0). F1 on the unseen
scanner at its natural mitosis prevalence and after resampling its cells to a prevalence of 0.2 or 0.6 (source scanners:
0.32 to 0.45). Bold: best label-free method in the column. $^\ddagger$Also uses the labels of the source-validation
cells. At a single operating point, UPA reduces to keeping the source positive rate. Paired differences to TGR with 95\%%
intervals are in \tableref{tab:recal-diff}.}
\label{tab:recal}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccc@{\hspace{10pt}}ccc@{}}
\toprule
 & \multicolumn{3}{c}{ResNet-18 KD student} & \multicolumn{3}{c}{ConvNeXt-T KD student} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Prevalence on the new scanner & natural & 0.2 & 0.6 & natural & 0.2 & 0.6 \\
\midrule
%(body)s
\bottomrule
\end{tabular}
\end{table}
""" % {"body": "\n".join(body)}
(OUT / "table3.tex").write_text(t3)

preview = r"""\documentclass[11pt]{article}
\usepackage[textwidth=6.25in,textheight=9in]{geometry}
\usepackage{booktabs,makecell,amssymb,natbib}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\renewcommand{\citep}[1]{[UPA]}
\begin{document}
\section*{Draft tables (generated by scripts/paper\_tables.py)}
\input{table1}
\input{table2}
\input{table3}
\end{document}
"""
(OUT / "preview.tex").write_text(preview)
print("wrote", *(OUT / f"table{i}.tex" for i in (1, 2, 3)), OUT / "preview.tex")

# MIDL/jmlr versions (\floatconts) for the manuscript
import re
MS = ROOT / "paper" / "manuscript" / "tables"; MS.mkdir(parents=True, exist_ok=True)
for i, t in ((1, t1), (2, t2), (3, t3)):
    m = re.match(r"\\begin\{table\}\[t\]\n\\centering\n\\caption\{(.*?)\}\n\\label\{(.*?)\}\n(.*)\\end\{table\}\n?$", t, re.S)
    assert m, i
    cap, lab, body = m.groups()
    (MS / f"table{i}.tex").write_text("\\begin{table}[t]\n\\floatconts{%s}\n{\\caption{%s}}\n{%%\n%s}\n\\end{table}\n" % (lab, cap, body))
print("wrote MIDL versions to", MS)

# Protocol N: is the distilled ConvNeXt-T student non-inferior to its foundation-model teacher on a new scanner?

Written 2026-11-25 16:03 CT. Margin agreed with the user before any ConvNeXt student-minus-teacher interval was
computed, and before any ConvNeXt student with the Virchow2 teacher was trained.

## Question

Paper headline candidate: "a student 40 times smaller is close to the teacher on an unseen scanner". This protocol
fixes what "close" means and how it is tested.

## Setting (unchanged from `MULTI_TEACHER_PROTOCOL.md` amendment 1)

New-scanner F1, single view, cross-fitted over all 50 images of each held-out scanner; F1 per run, mean over 10
student seeds, then over the 3 folds. Student cutoff: its source-validation cutoff unless stated. Teacher: view 0,
its own source-validation cutoff, LoRA seed 0 (the teacher the students were distilled from).

## Margin and test

Non-inferiority margin: **delta = 0.02 F1 (absolute)**. Reason: it equals the range of the Optimus teacher's own
new-scanner F1 over three LoRA seeds (0.801 to 0.821, `teacher_seeds_summary.csv`), so "within delta" means "within
what retraining the teacher itself changes".

Difference D = F1(student) - F1(teacher). Paired bootstrap, B = 20000: the same multinomial weights over the 50
images of a fold for student and teacher, student seeds resampled per fold, teacher fixed (seed 0); generator seeded
[20261201, fold index], folds ordered CS2, S360, XR, shared by every arm and method. The student is declared
non-inferior if the lower limit of the two-sided 95% percentile interval of D is above -0.02. One-sided p =
(number of bootstrap D <= -0.02, plus 1) / (B + 1); Holm within each family at one-sided alpha 0.025.

## Family N1: Optimus teacher (2 contrasts; students already trained)

| # | Student | Against | Known point estimate |
|---|---|---|---|
| N1a | `cnx_kd_tga` (KD + TGA, retrained on unlabelled new-scanner images) | Optimus teacher | 0.805 vs 0.814 |
| N1b | `cnx_kd` + TGR (one bias, no retraining; cross-fitted as in `scripts/cnx_boot.py`) | Optimus teacher | 0.801 vs 0.814 |

Expectation, stated honestly: the point estimates are inside the margin, but the paired interval of the ResNet
student against the same teacher (already seen, `layer_ladder.csv`: 0.026, 0.014 to 0.039) has a half-width of about
0.012, so N1a is expected to be borderline and N1b to fail. Whatever comes out is reported.

## Family N2: Virchow2 teacher (students NOT trained yet; replication of the headline on a second teacher)

Students: `cnx_kd_virchow2` and `cnx_kd_tga_virchow2{A,B}`, the recipe of the Optimus ConvNeXt arms (alpha 0, T 2,
beta 1, 60 epochs, 10 seeds, 3 folds, halves A/B), `--teacher_fm virchow2`, existing seed-0 Virchow2 caches;
evaluated with `cf_eval_any.py --out cnx_crossfit_virchow2`.

| # | Contrast | Test | Expectation |
|---|---|---|---|
| N2c | `cnx_kd_tga_virchow2` vs `cnx_kd_virchow2` | two-sided superiority, alpha 0.05 (single contrast) | positive and significant (Optimus: +0.040) |
| N2d | `cnx_kd_tga_virchow2` vs Virchow2 teacher | non-inferiority, delta 0.02 | same outcome as N1a |
| N2e | `cnx_kd_virchow2` + TGR (Virchow2) vs Virchow2 teacher | non-inferiority, delta 0.02 | same outcome as N1b |

N2d and N2e form one Holm family (one-sided alpha 0.025).

## Descriptive only (no test, no correction)

- Share of the teacher's advantage recovered: (student - `cnx_control`) / (teacher - `cnx_control`), with a bootstrap
  interval, for `cnx_kd`, `cnx_kd` + TGR and `cnx_kd_tga` (and the Virchow2 arms). This is the number to quote if a
  non-inferiority test fails.
- ResNet-18 KD + TGA against its teacher (interval already seen, so not a test).
- The same differences against teacher LoRA seeds 1 and 2 (`teacher_seeds_cells.csv.gz`).
- Same treatment for both sides: (i) 8-view TTA for the teacher with a TTA cutoff from its source validation set
  (`jobs/pc/teacher_tta.py`, the views of `eval_extras.probs`) against the student's TTA; (ii) the cross-fitted
  labelled cutoff for both.
- Efficiency (`jobs/pc/efficiency.py`): parameters and cells per second on the RTX 3090 Ti for ResNet-18 and
  ConvNeXt-T (128 px crop, as trained) and the four teachers (224 px crop, as trained), mixed precision as used in
  evaluation.
- `cnx_kd_virchow2` vs `cnx_control` (Optimus: +0.029).

## Context already known when this was written

Point estimates in the table above; ConvNeXt KD + TGA vs KD (+0.040) and vs KD + TGR (+0.004); the ResNet
teacher-minus-student intervals for all four teachers (`layer_ladder.csv`); teacher F1 for LoRA seeds 0 to 2. Not
known: any interval of a ConvNeXt student against a teacher; anything about ConvNeXt students with Virchow2; teacher
TTA; throughput.

## Amendments

(none yet)

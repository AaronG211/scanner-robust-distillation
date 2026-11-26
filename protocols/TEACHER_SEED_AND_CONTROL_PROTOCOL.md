# Protocols F and C: teacher-seed robustness of the headline, and the ConvNeXt-T no-teacher adaptation control

Written 2026-11-26 00:44 CT, before any of the students below was trained and before any teacher cache for H-optimus-0
LoRA seeds 1 and 2 existed. The user asked for both runs on 2026-11-26 ("今晚PC空着 跑F和ConvNeXt对照吧").

## Context already known when this was written

- Protocol N (`NONINFERIORITY_PROTOCOL.md`): ConvNeXt-T KD + TGA minus the H-optimus-0 teacher (LoRA seed 0) -0.0092
  (-0.0216 to +0.0028), non-inferiority at 0.02 not established; minus Virchow2 -0.0006 (-0.0138 to +0.0122), established.
- The ConvNeXt-T KD + TGA student (seed-0 teacher) against teacher LoRA seeds 1 and 2, descriptively: -0.016 and +0.004.
- Teacher F1 for H-optimus-0 LoRA seeds 0 / 1 / 2: 0.814 / 0.821 / 0.801 (`teacher_seeds_summary.csv`).
- ResNet-18: KD + TGA minus the no-teacher adaptation (`r18_free_tga`) +0.019 (Optimus), +0.012, +0.013, +0.002.
- Not known: any student distilled from H-optimus-0 LoRA seed 1 or 2; any ConvNeXt-T student adapted without a teacher.

## Protocol F: does the headline survive other trainings of the teacher?

### Students (new)

ConvNeXt-T, the recipe of `cnx_kd` / `cnx_kd_tga` unchanged (alpha 0, T 2, beta 1, 60 epochs, 10 seeds, 3 folds, halves
A/B), distilled from H-optimus-0 LoRA seed k (phase 4 checkpoints `hopt0_lora_{h}_plain_e5_s{k}.pt`, same LoRA recipe as
seed 0), `distill.py --teacher_seed k`:

- k = 1: `cnx_kd_t1`, `cnx_kd_tga_t1{A,B}`
- k = 2: `cnx_kd_t2`, `cnx_kd_tga_t2{A,B}`

Teacher caches `teacher_{h}_s{k}.npz` (16 views of the source training cells) and `teacher_{h}_targetall_s{k}.npz`
(all new-scanner cells, for the halves) are made by `distill_teacher.py --seed k` exactly as for seed 0. The only code
change: its validation-F1 self-check reads the saved run from `fm_lora_results_seeds.csv` when k > 0 (the seed-0 runs are
in `fm_lora_results.csv`). As in that script, the stain jitter of the cached views is drawn with generator 1000 + k, so it
differs between teacher seeds; this is part of re-running the pipeline with another teacher training.
Evaluation: `cf_eval_any.py --arms cnx_kd_t1 cnx_kd_tga_t1 cnx_kd_t2 cnx_kd_tga_t2 --out cnx_crossfit_tseeds`.

### Comparisons

Same definitions as Protocol N: new-scanner F1, single view, all 50 images per held-out scanner, cross-fitted; student at
its source-validation cutoff; teacher = the SAME LoRA seed k the student was distilled from, view 0, its own
source-validation cutoff (`teacher_seeds_cells.csv.gz`). Paired bootstrap B = 20000, generator [20261201, fold index]
(folds CS2, S360, XR), the same multinomial image weights and student-seed resampling as Protocol N.

**Family F (pre-specified, 2 contrasts, Holm at one-sided alpha 0.025), non-inferiority margin delta = 0.02:**

| # | Contrast | Non-inferior if |
|---|---|---|
| F1 | `cnx_kd_tga_t1` minus teacher LoRA seed 1 | lower 95% limit > -0.02 and Holm-adjusted one-sided p < 0.025 |
| F2 | `cnx_kd_tga_t2` minus teacher LoRA seed 2 | same |

**Key secondary (pre-specified estimate; its verdict is reported but labelled as partly based on data already seen):**
the teacher-seed-averaged gap, mean over k = 0, 1, 2 of (`cnx_kd_tga` with teacher k minus teacher k), with the same
bootstrap (the k = 0 term is Protocol N's N1a). This is the number the paper quotes as "the gap to H-optimus-0 averaged
over teacher trainings".

Expectation, stated before training: in every setting so far the retrained student landed near 0.80 to 0.81 whatever the
teacher (ConvNeXt-T: 0.805 with both teachers). If that holds, F1 (teacher 0.821) is about -0.016 and fails, F2 (teacher
0.801) is about +0.004 and passes, and the averaged gap is about -0.007 with a lower limit near -0.02 (borderline).
Whatever comes out is reported.

Descriptive only: share of the teacher's advantage recovered per teacher seed (reference `cnx_control`); KD + TGR (one
bias fitted to teacher k's probabilities, cross-fitted) minus teacher k; KD + TGA minus KD per teacher seed; the spread of
student F1 across teacher seeds next to the spread of teacher F1.

## Protocol C: how much of the ConvNeXt-T student's retraining gain needs a teacher?

### Students (new)

`cnx_free_tga{A,B}`: the recipe of `r18_free_tga` (Experiment 2b) with the ConvNeXt-T backbone: source cells with expert
labels (`--source_labels --alpha 1 --beta 0`), the fit half's unlabelled new-scanner cells with the soft answers of the
ResNet-18 no-teacher ensemble + AdaBN (`--target_soft_only --target_teacher r18ens`, the same pseudo-label files
`teacher_{h}_r18ens{A,B}_s0.npz` as `r18_free_tga`), 60 epochs, 10 seeds, 3 folds, halves A/B. No foundation model is
used anywhere. Evaluation: `cf_eval_any.py --arms cnx_free_tga --out cnx_crossfit_free`.

### Comparisons

Same F1 definition and bootstrap as above.

**Family C (pre-specified, 2 contrasts, Holm at two-sided alpha 0.05):**

| # | Contrast | Expectation |
|---|---|---|
| C1 | `cnx_kd_tga` (H-optimus-0) minus `cnx_free_tga` | positive, between 0 and +0.04 (ResNet-18 analogue: +0.019) |
| C2 | `cnx_kd_tga_virchow2` minus `cnx_free_tga` | positive, between 0 and +0.04 (ResNet-18 analogue: +0.012) |

**Pre-committed wording rule for the paper (RQ3b).** Descriptive share of the retraining gain that needs no teacher =
(`cnx_free_tga` - `cnx_control`) / (`cnx_kd_tga` - `cnx_control`), per teacher, with a bootstrap interval (ResNet-18 with
H-optimus-0: 0.75). If both point estimates are at least 0.5, the paper says "for both students most of the retraining
gain comes from the unlabelled images rather than from the teacher". If either is below 0.5, the paper says this only for
ResNet-18 and states that for ConvNeXt-T the teacher contributes most of the retraining gain.

Descriptive only: `cnx_free_tga` minus `cnx_control`; `cnx_free_tga` minus `cnx_kd` + TGR.

## Order on the PC (`run_phase6.ps1`, starts after the GPU has been free for 10 minutes)

1. one teacher cache (XR, seed 1) as an early check of the changed self-check;
2. Protocol C students, then their evaluation;
3. the remaining eleven teacher caches;
4. Protocol F students (seed 1, then seed 2), then their evaluation.

Analysis scripts are written before the result files are copied to the Mac: `scripts/teacher_seed_robustness.py`
(family F) and `scripts/cnx_free_control.py` (family C).

## Amendments

(none yet)

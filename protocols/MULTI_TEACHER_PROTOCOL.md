# Protocol E1.7: do the distillation findings hold for other foundation-model teachers?

Written 2026-11-25 (US Central, night), before any Virchow2 / UNI / UNI2-h teacher or student was trained.
Roadmap: `ai_health/Research_Roadmap_2026-11-24.md`, experiment E1.7. Suggested by a teammate.

## Question

With H-optimus-0 as teacher we found, for a ResNet-18 student on the new scanner (cross-fit, 10 seeds):
no teacher 0.711, KD 0.725 (KD vs no teacher +0.013, n.s.), KD + TGA 0.788, Optimus-free adaptation 0.769
(Optimus TGA vs Optimus-free −0.019 in favour of Optimus, Holm p 0.009). Do these hold for other teachers?

## Teachers (all trained the same way as the existing Optimus teacher)

| fm | Model | Backbone | Params | Normalisation | Features |
|---|---|---|---|---|---|
| hopt0 | H-optimus-0 (existing, not retrained) | ViT-g/14 | 1.1B | model card | CLS + mean patch, 3072 |
| virchow2 | Virchow2 | ViT-H/14, 4 registers | 632M | ImageNet (timm config) | CLS + mean patch, 2560 |
| uni | UNI | ViT-L/16 | 303M | ImageNet | CLS + mean patch, 2048 |
| uni2h | UNI2-h | ViT-H/14, 8 registers | 681M | ImageNet | CLS + mean patch, 3072 |

Recipe (unchanged from `fm_lora.py`): frozen bf16 encoder, LoRA rank 16 / alpha 16 / dropout 0.1 on every qkv, proj,
fc1, fc2; linear mitosis head on [CLS, mean patch token]; 5 epochs, lr 1e-4, batch 16, one-cycle, dihedral augmentation,
no stain augmentation, seed 0; checkpoint and cutoff chosen on the training scanners' validation F1. Same 256 px patches
at native 40x, centre 224 crop for all teachers (the same magnification mismatch applies to every teacher).

Teacher caches (`distill_teacher.py --fm`): same 16 views and the same stain-jitter draws as the Optimus caches
(the random generator does not depend on the teacher).

Code change check (done before this protocol): the patched `distill_teacher.py` regenerates the Optimus XR
`targetall` cache bit for bit (all arrays identical).

## Students (ResNet-18, identical to the Optimus arms)

For each new teacher T:

- `r18_kd_T`: alpha 0, T 2, beta 1, 16 views, 60 epochs, lr 3e-4, 3 folds × 10 seeds.
- `r18_kd_tga_T{A,B}`: as `r18_kd_tga`: teacher answers and features on one half of the new scanner's images
  (unlabelled), scored on the other half; 3 folds × 2 halves × 10 seeds.

Reused unchanged: `r18_control`, `r18_kd`, `r18_kd_tga` (Optimus), `r18_free_tga` (Optimus-free adaptation).

## Outcome

New-scanner F1, single view, cutoff from the training scanners' validation set, cross-fitted over all 50 images of
each new scanner; F1 per run, mean over 10 seeds, mean over the 3 folds. Teacher alone: view 0 with its own
validation cutoff (one seed).

## Pre-specified contrasts (family of 9, Holm; paired bootstrap over images and seeds, B = 20000)

For each T in {virchow2, uni, uni2h}:

1. KD_T vs no teacher (`r18_kd_T` vs `r18_control`). Optimus: +0.013, n.s.
2. KD+TGA_T vs KD_T. Optimus: +0.064.
3. KD+TGA_T vs Optimus-free adaptation (`r18_kd_tga_T` vs `r18_free_tga`). Optimus: +0.019.

Expectations written down now: (1) small and not significant for every teacher; (2) positive and significant for
every teacher; (3) between 0 and +0.03 for every teacher. Any teacher that breaks an expectation is reported as such.

## Descriptive only (no test, no correction)

- Each teacher's own new-scanner F1 and AUROC (all 50 images, and official 11 test images).
- Gap decomposition (`scripts/gap_decomposition_pilot.py` logic): source-cutoff F1, oracle-cutoff F1, AUROC per arm.
- Across the 4 teachers: teacher robustness against student KD gain and TGA gain (n = 4, a plot, not a test).
- With TTA and AdaBN columns, as for the Optimus arms.

## Deviations

Any change after results are seen goes into an "Amendments" section below with date and reason.

## Amendments

1. 2026-11-25 04:44 CT, after the Virchow2 results and before any UNI, UNI2-h or ConvNeXt-T result. Addition, not a
   change: a separate one-contrast family for a second student architecture. ConvNeXt-T students (LayerNorm, no
   BatchNorm) with the Optimus teacher, same recipe as the ResNet arms (alpha 0, T 2, beta 1, 60 epochs, 10 seeds):
   `cnx_kd_tga` (cross-fitted over halves A/B) vs `cnx_kd`. Expectation: positive and significant. Evaluated with
   `jobs/pc/cf_eval_any.py` (a copy of `r18_crossfit_eval.py` that also finds ConvNeXt file names). The nine contrasts
   above are unchanged.
2. 2026-11-25 06:04 CT, after the Virchow2 and UNI results (UNI's KD+TGA student, 0.782, is above the UNI teacher,
   0.767), before any UNI2-h result. Descriptive addition, no test: LoRA seeds 1 and 2 for all four teachers (same recipe:
   plain, 5 epochs; written to `results/fm_lora_results_seeds.csv` so the original results files stay untouched),
   evaluated by `jobs/pc/teacher_seeds.py` (view 0, own validation cutoff). Reported: each teacher's new-scanner F1 as
   mean and range over three seeds, and whether "student above teacher" holds against every UNI seed. Students stay
   trained on the seed-0 teachers only. Motivated by the UNI result, so it is labelled post hoc wherever it is used.

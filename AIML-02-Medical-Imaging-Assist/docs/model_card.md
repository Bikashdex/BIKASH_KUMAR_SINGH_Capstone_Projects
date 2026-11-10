# Model card

> Portal step 7: *"Containerize with Docker and document the model's limits and bias. Include a clear 'for education
> only, not clinical use' notice."* This file **is** that deliverable. Every number below is filled from a measured
> hosted-GPU run — nothing here is a placeholder.

## ⚠ Intended use

**Education only. Not for clinical use.**

This is not a medical device. It has no FDA, CE, or CDSCO clearance. It must not be used to triage, diagnose,
screen, or treat any patient, and no clinical decision should rely on its output.

---

## Where these numbers come from

| Section | Source of truth | Host and sample |
|---|---|---|
| Track A metrics, confusion matrix, baseline comparison | `results/densenet_metrics.json`; the matrix itself is plotted in `results/plots/confusion_matrix.png`, recomputed from `results/test_predictions_with_metadata.csv` and cross-checked against the JSON counts — notebook 03, kernel `bikash24795/aiml-02-notebook-03-densenet121-chest-x-ray` | Kaggle T4, 4,003 test images |
| Calibration | `results/calibration.json`, computed from `results/test_predictions_with_metadata.csv` | same 4,003-image test split |
| Subgroups, Grad-CAM | `results/gradcam_fairness_summary.json` and `results/gradcam_focus.csv` — notebook 04, kernel `bikash24795/aiml-02-notebook-04-grad-cam-and-fairness` | Kaggle T4, same split |
| Track B | `results/spleen_metrics.json`, `results/spleen_per_case.csv`, overlays in `results/plots/spleen_overlays.png` — notebook 05, kernel `bikash24795/aiml-02-notebook-05-monai-spleen-segmentation` | Kaggle T4, 6 held-out CT volumes |

The `notebooks/_smoke/executed_*.ipynb` files are **local CPU smoke runs**, kept to prove the code paths execute:
notebook 03 was exercised on 96/60/40 images for 3 epochs (test AUROC 0.9475 on 40 images) and notebook 05 on a
64³ ROI for 1 epoch over 2 test cases. They are not the source of any number in this card — their test sets are two
orders of magnitude smaller than the real one. Where a smoke number and a Kaggle number disagree, the Kaggle number
is the one that counts.

## Track A — Chest X-ray pneumonia classifier

### Identity

| Field | Value |
|---|---|
| Architecture | DenseNet121, ImageNet-pretrained, final layer replaced |
| Task | Binary classification: lung opacity vs not. Opacity is the **sign** the label encodes, used by RSNA as the pneumonia surrogate — it is not a pneumonia diagnosis |
| Input | 224×224 RGB from 8-bit DICOM, ImageNet normalisation |
| Output | Sigmoid probability + Grad-CAM heatmap |
| Training data | RSNA 2018 stage_2, 26,684 images, 22.53 % positive (see `dataset_card.md`) |
| Split policy | **Image-level split on one-row-per-picture data**, seed 7, 70/15/15, folds asserted disjoint |
| Version / date | `densenet121-2026-10-26` (Kaggle T4, torch 2.11.0+cu128, python 3.13.15, seed 7) |

### Performance

Measured in `notebooks/03_train_densenet.ipynb` on the Kaggle T4 run of 2026-10-26; raw numbers in
`results/densenet_metrics.json`. Early stopping picked epoch 4 of 8; threshold 0.5108 is Youden's J read on
validation only and then applied unchanged to the test split.

| Metric | Val (n=4,003) | Test (n=4,003) |
|---|---|---|
| AUROC | 0.8882 | 0.8819 |
| PR-AUC | 0.7096 | 0.7085 |
| Sensitivity (recall) | 0.8104 | 0.7661 |
| Specificity | 0.8217 | 0.8055 |
| Precision | 0.5693 | 0.5340 |
| F1 | 0.6688 | 0.6293 |
| Accuracy | 0.8191 | 0.7967 |
| Threshold used | 0.5108 | 0.5108 |

Confusion matrix on the test split (n = 4,003, of which 902 are positive) at the same 0.5108 threshold:

| | Predicted negative | Predicted positive |
|---|---|---|
| Actually negative (n = 3,101) | TN 2,498 | FP 603 |
| Actually positive (n = 902) | FN 211 | TP 691 |

Accuracy is reported but **is not the headline metric** — the class balance makes it misleading. AUROC, sensitivity
and specificity are the numbers that matter for a screening aid.

### Baseline comparison

Measured in `notebooks/02_baseline_logreg.ipynb`, full detail in `results/baseline_metrics.json`. Subsets are not
identical between rows - level 1 uses 4,000 train images, level 2 uses 1,200 - so compare the test columns, not the
train ones.

| Model | AUROC val | AUROC test | Note |
|---|---|---|---|
| Constant predictor | 0.5000 | - | 77.47 % accuracy for a switch. The reason accuracy never leads a table |
| Logistic regression on 1,024 pixels (32x32 downsample) | 0.7520 | **0.7400** | Course Day 10 baseline. Fits its own train subset at 0.9037 |
| Logistic regression on 512 frozen ResNet-18 ImageNet features | 0.7234 | 0.7250 | Course Day 12 without the fine-tuning. Train AUROC 0.9999 on 1,200 samples |
| DenseNet121, fine-tuned (this model) | 0.8882 | **0.8819** | Beats the 0.740 bar by **+0.142** test AUROC. Early stopping picked epoch 4 of 8 (`results/densenet_metrics.json`) |

The frozen-feature row scoring *below* the pixel row is a result, not a mistake. Pretrained ImageNet features are
tuned for natural photographs; the value of transfer learning here appears only once the weights are fine-tuned —
and it does: fine-tuning moves that backbone from 0.7250 to **0.8819** test AUROC, +0.157, which is the answer to
"did transfer learning add anything" that this table exists to give.

Thresholds are not shared across rows. The pixel baseline's test numbers are taken at its own **0.163**, Youden's J
read on validation only; at that cut-off it reaches sensitivity 0.698 / specificity 0.663. The DenseNet's numbers
use its own 0.5108. AUROC leads the table because it is the one number that does not depend on that choice.

A random forest was not used as a baseline. The three levels above - constant, hand-of-pixels, frozen-CNN - isolate
the two things worth separating (representation and fine-tuning), which a second tree model on the same 1,024
features would not.

### Calibration

A probability of 0.87 should mean roughly 87% of the time. It does not here. Measured by
`tools/calibration_report.py` from `results/test_predictions_with_metadata.csv` (n = 4,003), with the reliability
diagram in `results/plots/calibration.png` and the numbers in `results/calibration.json`.

| Metric | Value |
|---|---|
| Prevalence on the test split | 0.2253 |
| Mean predicted score | 0.3509 |
| Brier score | 0.1323 |
| Log loss | 0.4034 |
| ECE (10 bins) | 0.1256 (95 % bootstrap CI 0.1147–0.1360) |
| MCE (10 bins) | 0.3152 |

The model is **over-confident**. Its average score is 0.351 against a true positive rate of 0.225, and the reliability
curve sits above the diagonal across most bins. The cause is visible in training: BCE used `pos_weight = 3.4387` to
offset the 77.5 / 22.5 class split, which inflates scores for the minority class. The 0.5108 threshold was chosen on
this same uncalibrated scale, so it is a *ranking* cut-off, not a risk estimate. Recalibration would mean Platt
scaling or isotonic regression fit on a held-out fold; that is not done here, and the model card says so instead of
printing a sigmoid as a probability.

### Explainability — and its honest limits

Grad-CAM produces a ReLU'd activation map from the final dense block, upsampled onto the input.

**What it is not:** it is not SHAP, it is not a causal explanation, and it is not a guarantee of where the model
looks. A Grad-CAM map over the heart does not prove the model used the heart. It shows a region whose activations
contributed to the class score. Say this in the viva; do not oversell the heatmap.

The gallery is `results/plots/gradcam_gallery.png` (3 confident hits at p=0.99, 3 confident misses at
p≤0.01, 2 false positives at p=0.97, 1 true negative at p=0.00) and `results/plots/gradcam_uncertain.png`
(the two test cases inside threshold ±0.05: p=0.476 on a true opacity, p=0.523 on a negative). The misses
are the useful panels: their heatmaps sit in the **image corners and collimation edges**, not on lung
parenchyma, so the evidence the model used to call them negative was background, not anatomy.

That reading is supported by a number, not just by eye: of the 300 sampled maps, **130 (43 %)** put less
than half their energy inside the body, and the median in-body fraction is **0.484 for label-0 images
against 0.707 for label-1 images** (`results/gradcam_focus.csv`, `gradcam.by_label` in
`results/gradcam_fairness_summary.json`). The classifier's gradient is partly locked onto frame borders —
consistent with the false-negative row above.

### Bias and subgroup performance

| Slice | n (test split / whole archive) | AUROC on the held-out test split (95 % CI) | Gap vs overall (0.8819) | Sens / spec at 0.5108 |
|---|---|---|---|---|
| Sex = M | 2,299 / 15,166 | 0.8743 (0.8604–0.8902) | −0.0076 | 0.762 / 0.794 |
| Sex = F | 1,704 / 11,518 | 0.8919 (0.8764–0.9083) | +0.0100 | 0.772 / 0.821 |
| Age 0-39 | 1,332 / 9,044 | 0.8945 (0.8744–0.9113) | +0.0126 | 0.795 / 0.810 |
| Age 40-59 | 1,708 / 11,667 | 0.8908 (0.8748–0.9083) | +0.0089 | 0.775 / 0.809 |
| Age 60-79 | 924 / 5,731 | 0.8433 (0.8091–0.8724) | −0.0386 | 0.699 / 0.797 |
| Age 80+ | 39 / 242 | 0.6855 (0.4530–0.9476) | −0.1964 | 0.625 / 0.710 |
| Label source = `Normal` | 1,331 / 8,851 | n/a — no positives in this slice | n/a | spec 0.990 |
| Label source = `No Lung Opacity / Not Normal` | 1,770 / 11,821 | n/a — no positives in this slice | n/a | spec 0.667 |

Reading the table: sex is flat (0.018 AUROC gap, CIs overlap), so there is no measurable sex disparity here.
The 60–79 dip (924 images, 176 positives) is the one age gap with enough support to take seriously.
The 80+ number (39 images, 8 positives) has a CI spanning 0.45–0.95 and must not be quoted as evidence of a
disparity — the widest `sex_x_age` gap (0.3291) is the same artefact, built from 17- and 22-image cells.
The two label-source rows cannot have an AUROC because those slices contain **no positive labels at all**;
what they measure is specificity, and the gap between them (0.990 vs 0.667) is the finding: the model is
right almost always on truly-normal films but rejects a third of the "abnormal for another reason" films
out of hand, i.e. it has partly learned *"not obviously normal"* rather than *"opacity"*. The source of every
number is `fairness.*.table` in `results/gradcam_fairness_summary.json`.

The split is 4,003 test images, so each row's test sample is roughly one sixth of the archive count. The
subgroup tables are emitted once per band label (`sex`, `age_band`, `class_source`, `sex_x_age`) by
`notebooks/04_gradcam_fairness.ipynb`; nothing here is re-binned by hand. The run's missing-data audit is
clean on all six checks (`results/missing_data_audit.csv`): 0 of 4,003 test images lack a cached render, a
`PatientSex`, a `PatientAge`, or a joined radiological class, and no `PatientID` crosses the split.

Slice counts and the sex/age values come from the DICOM tags, recovered in `notebooks/01_eda.ipynb` §3 and written
into `results/split_*.csv`. Observed label prevalence before any model is involved: 21.7 % female vs 23.1 % male,
and 25.5 % under 40, 21.0 % in 40–59, 20.9 % in 60–79, 24.8 % in 80–110. Prevalence barely moves across these slices,
which means **a large AUROC gap between them would point at the model, not at the data** — that is the useful
comparison to make.

The last two rows matter more than the demographic ones. The negative class is not "normal chest": 11,821 of the
20,672 negatives are abnormal for some other reason. Measuring accuracy separately on the 8,851 truly-normal films
versus the 11,821 other-abnormal films tells you whether the model learned "opacity" or merely learned "not
obviously normal". Report both.

**The PA vs AP view confound cannot be examined here, and that is stated rather than skipped.** `PatientPosition`
is empty on all 26,684 files, so view is not recoverable from this release. The mechanism is real and worth naming in
the viva: AP films are usually portable X-rays on patients who cannot stand, so view position correlates with illness
severity, and a model can learn "portable machine → opacity" without learning the pathology. This dataset simply
cannot show it.

What can be measured, and was:

- **Gross exposure as a shortcut.** Mean image brightness correlates with the label at **-0.061** on 4,000 training
  images (`results/plots/baseline_pixels.png`). A plausible confounder, measured, and ruled out as a major driver.
  Ruled out for brightness specifically - not for every low-frequency cue the pixel model uses.
- **Prevalence by slice**, so an AUROC gap can be attributed to the model rather than to unequal base rates
  (table above).
- **Performance by slice** on a 2,000-image test subset for the pixel baseline: female 0.7274, male 0.7500, and
  0.7625 under 40 / 0.7557 in 40-59 / 0.6576 in 60-79. The 80-110 band is too small to score. With 397 images and
  75 positives in the weakest band, that 0.10 spread is suggestive and not proven; it is exactly the kind of
  quantity a confidence interval settles, which notebook 03 computes on the full test set.

Still open: duplicate renders of the same film within the release, and whether `PixelSpacing` clusters by
acquisition. Neither is claimed as done.

### Known limitations

1. Single-dataset training; no external validation on a different hospital's cameras or population.
2. Binary task only — cannot report the other 13 findings a real chest X-ray study needs assessed.
3. **Not paediatric-capable.** 1,276 of 26,684 images (4.8 %) are from patients under 18, which is enough to keep
   them in training but nowhere near enough to evaluate a paediatric subgroup. Paediatric chests also look
   structurally different. Do not present this as valid on children.
4. No 3D information — overlapping structures can hide findings a radiologist would resolve from two views.
5. Label noise: bounding boxes came from 18 radiologists across 16 institutions.
6. Demographic distribution mirrors one US dataset; transferability to the local population is untested.
7. Confidence scores are not calibrated — mean score 0.351 against 22.5 % prevalence, ECE 0.126 (see *Calibration*).
   Read the 0.5108 cut-off as a ranking threshold, not as a risk estimate.

### Failure modes observed

Three concrete cases, each a row of `results/test_predictions_with_metadata.csv` or a panel of the two galleries:

1. **Confident false negative.** `3e9ff8ec-edfa-4435-a3ef-898b09b234bc`, a 25-year-old man labelled
   `Lung Opacity`, scored **p = 0.0034** — the most confident miss in the test split. Its Grad-CAM panel in
   `gradcam_gallery.png` puts the hot spot in a top corner of the frame, not on the lungs. The model was not
   borderline about this film; it read background/clipping structure and concluded "no opacity". This is the
   failure mode a screening deployment would actually suffer, because the miss arrives with high confidence.
2. **Confident false positive on another abnormality.** `595786d9-b679-4722-90a8-bad34a40c350`, a 24-year-old
   man whose radiological source is `No Lung Opacity / Not Normal`, scored **p = 0.968**. The heatmap spreads
   across the chest rather than concentrating. Combined with the 0.667 specificity on that same slice, this is
   the model reacting to "this chest is abnormal" instead of "this chest has an opacity".
3. **The uncertain band.** `gradcam_uncertain.png` shows the two test cases closest to the cut-off: a true
   opacity at **p = 0.476** (0.035 below 0.5108) and a negative at **p = 0.523** (0.012 above). Both heatmaps
   spill off the body onto the frame. These are the cases a human-in-the-loop would wave through, and they are
   why the viewer shows the score and the heatmap instead of a yes/no badge.

Across all three the same underlying weakness shows up: part of the evidence is the frame, not the patient —
43 % of sampled maps put less than half their energy inside the body.

### Intended deployment and guardrails as built

- Human-in-the-loop is mandatory: the viewer's Approve / Reject decision is recorded and the model never finalises a
  result alone.
- Full audit trail of every prediction and every human decision.
- Confidence thresholding plus an explicit "uncertain" band rather than forcing a binary call.
- Input drift monitor (PSI on pixel statistics) to warn when incoming studies stop looking like training data.
- Education-only banner rendered inside the React viewer, not just in this document.

### What would change the assessment

External validation on a second institution, a multi-label target, calibration, and a prospective review of whether
the model changes radiologist behaviour at all.

---

## Track B — Spleen segmentation, 3D U-Net (MONAI `DynUNet`)

| Field | Value |
|---|---|
| Architecture | `DynUNet`, 3D, deep supervision (5 downsamples, filters to 320, 31,409,478 params) |
| Input | CT volume resampled to 1.5 mm iso, windowed to [-150, 200] HU |
| Output | Binary spleen mask |
| Training data | MSD Task09_Spleen, 41 volumes — 28 train / 7 val / 6 test, seed 7, patient-safe by construction |
| Metric | Mean Dice, plus per-case Dice |
| Inference | Sliding window (0.5 overlap, Gaussian blend), ROI 128³ |
| Status | **Run on a Kaggle T4 (2026-10-27, 119.3 GPU-min, 90 epochs) — quotable.** The earlier 40-epoch run is kept below only as the record of a bug that has since been fixed |

| Metric | Value |
|---|---|
| Mean Dice (val) | 0.9357 best at epoch 79 (quick val, 3 cases); trajectory 0.305 → 0.665 → 0.885 → 0.906 → 0.695 → 0.837 → 0.540 → **0.936** → 0.899 at epochs 9/19/29/39/49/59/69/79/89 |
| Mean Dice (test) | **0.8586** — median 0.9124, range 0.5972–0.9480, n = 6, no zero-Dice cases |
| Worst-case Dice | 0.5972 (`spleen_31`) — under-segmentation, 159 ml predicted against a 357 ml reference, centroid 8.1 mm off |
| Best case | 0.9480 (`spleen_10`), 242 ml against 253 ml, centroid 0.7 mm off |
| `n_below_0_5` | 0 of 6 |
| Inference | sliding-window pass over all 6 test volumes took 4 min wall clock |

The mean sits inside the 0.80–0.90 band MSD spleen baselines target, and the per-case table shows *why* the mean
is not the whole story: five of six volumes score ≥ 0.84, and the one outlier misses most of a large spleen rather
than hallucinating (prediction smaller than reference, not a blob elsewhere). Predicted volumes now land within
3–13 % of the reference for those five, against a 55 % shortfall on the outlier and the 4–5× overshoot of the first
run; the two volumes whose centroids move most (`spleen_24` 36.6 mm, `spleen_63` 13.7 mm) still score 0.846 and
0.905, which is the signature of an elongated organ against a round centroid measure, not of a misplaced mask.

**What the first run got wrong, and what changed.** The 2026-10-26 run (`epochs_total = 40`) scored 0.329 mean
test Dice with every prediction 4–5× too large — 917–1,777 ml against references of 55–361 ml, centroids 114–160 mm
away. The sampler built each positive crop from a voxel *inside* the organ and each negative crop from the volume
centre, so the net learned a fixed blob near the middle of the window rather than the organ: the same checkpoint
scored 0.87 Dice on a hand-picked 128³ crop containing the spleen. `sample_patch` now jitters the positive centre
by up to a quarter of the ROI and draws negatives uniformly over the volume; the schedule went to 90 epochs
(128 patches/epoch, batch 2, 128³ ROI, lr 1e-4, seed 7 — all recorded in `results/spleen_metrics.json`). That
single change moved test Dice from 0.329 to 0.8586, which is the honest attribution for this number.

For the record, `notebooks/_smoke/executed_05_segment_spleen.ipynb` is only a code-path check (`run_on: local`,
`device: cpu`, 1 epoch / 2 optimizer steps, Dice 0.0142); by the repo's rule (§5 of the runbook) a local run is
never a result.

**Limits:** trained on 41 volumes from a single collection. Small spleens, post-surgical anatomy, and unusual
contrast timing are underrepresented, and the worst case here is exactly the failure mode to watch — a large spleen
partly left unlabelled. Six test volumes is a small sample: 0.8586 carries a wide confidence interval, and a single
extra miss would move the mean by roughly 0.05. Dilated or displaced organs, CT with strong artefacts, and any
non-contrast protocol are untested.

---

## Reproducibility

| Item | Value |
|---|---|
| Commit hash | `62511731238215809b258a92a9bfe1383943a958` at the time of writing. It moves every time a result is committed, so re-read `git rev-parse HEAD` before the presentation rather than trusting this row |
| DVC data revision | images `3fd56c2857f39d9f9df084ddadd7c8de.dir` (26,684 files, 3,555,311,924 B); same images as one archive `a8307fbc6d9138a7760dafde21383714` (3,534,067,864 B), which is the object `dvc push` transfers; `stage_2_train_labels.csv` `010a01a3f9a7cf76cb14704a95b22263`; `stage_2_detailed_class_info.csv` `c3bb15f6d279cfe39e09da8903902478`; derived header cache `dicom_headers.csv` `28b6ca9e1af3dde6e89e0cc3a755a3c9` |
| DVC remote | `gdrive://1_MZ3txKpgRucQ5Ep8YxZAbKovIzDJFiL` (a private Drive folder named `aiml02-dvc-remote`). Two independent obstacles had to be cleared, both recorded in `README.md`: the plugin's shared OAuth client is refused by Google for the `drive` scope — fixed by registering a personal **Desktop app** client and enabling the Google Drive API on that project (a *Web application* client can never work, because pydrive2 redirects to `http://localhost:8080/`) — and pushing the image folder would mean 26,688 separate Drive uploads, fixed by pushing the archive instead. Only the archive and the three CSVs are pushed; the folder revision above stays as the per-image md5 provenance record. |
| Seed | 7 |
| Runtime | baselines L0-L2: this laptop, CPU only, torch 2.11.0+cpu. L3 DenseNet121: **executed on a Kaggle T4** on 2026-10-26 — torch 2.11.0+cu128, CUDA 12.8, python 3.13.15, recorded in `densenet_metrics.json`'s `runtime` block. Notebooks 04 and 05 ran on the same Kaggle T4. A laptop CPU smoke run is written as `host: local` and is a code check, not a result |
| Package versions | `requirements.txt` |
| Environment that produced the baseline numbers | `results/environment.json` — Python 3.13.1, torch 2.11.0+cpu, numpy 2.4.3, scikit-learn 1.9.1, pydicom 3.0.2 |
| Reproduction status of the baselines | Ran twice end to end, every AUROC and PR-AUC identical to 4 decimal places, threshold reproduced at 0.163. Determinism holds *within* this environment; it is not claimed across environments, because the pinned `torch==2.4.1` / `numpy==1.26.4` cannot even install on 3.13 (no cp313 wheels) |
| Training command / notebook | `notebooks/03_train_densenet.ipynb` |

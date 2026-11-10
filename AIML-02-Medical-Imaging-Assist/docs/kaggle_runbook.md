# Kaggle runbook — notebooks 03, 04 and 05

**Primary runbook while the Colab quota is exhausted.** All three notebooks detect the
host from the filesystem (`RUN_ON = "kaggle"` when `/kaggle` exists), so nothing
in them needs editing — this file is about the Kaggle-specific clicks.

---

## 0. One-time prep

- [ ] **Kaggle account** — sign up at kaggle.com.
- [ ] **Accept the competition rules.** Open the *RSNA Pneumonia Detection
      Challenge* page → Rules → "I Understand and Accept". Without this the
      dataset does not appear in any Add Input list.
- [ ] **Phone verification** — Settings → verify your phone number. Kaggle
      enables GPU T4 only after this (and it is needed for Internet on).
- [ ] Quota: ~30 GPU hours/week on the free tier — notebook 03 needs only a few.

## 1. repo.zip — the code archive (once)

Use `repo.zip` from the workspace folder (~38 MB). It carries `src/`, `results/split_*.csv`,
the notebooks, the vendored monai/nibabel wheels, and `weights/densenet121-a639ec97.pth` —
the ImageNet weights are inside on purpose: `src/models/classifier.py` copies them into the
torch hub cache so **no internet is needed during the run**.

Upload it once as a private dataset. The **canonical dataset that all three kernels attach is
`bikash24795/aiml02-code`** (last updated 2026-10-26 08:25 UTC). The earlier uploads —
`bikash24795/aiml02-repo`, `bikash24795/repo-zip`, and `bikash24795/repo1-zip` — plus the
`bikash24795/aiml02-probe` scratch dataset still exist, but no kernel uses them.

1. kaggle.com → Datasets → **New Dataset** → drag `repo.zip` → Create.
   (Private by default — keep it that way; it is your coursework.)
2. From then on, attaching it to any notebook is one click (Add Input → Your
   Datasets). Alternatively drag the zip into Add Input → Upload per session.

Kaggle **unzips** a zip uploaded to a dataset, so the attached input lands as a plain tree and
there is no `repo.zip` file to find. The tree is **not** at `/kaggle/input/<slug>/`. In the
2026-10-26 runs it landed at **`/kaggle/input/datasets/<owner>/<slug>/`**, i.e.
`/kaggle/input/datasets/bikash24795/aiml02-code/{src,results,vendor,weights,...}` — visible in
the notebook-03, -04 and -05 logs. Kernel outputs use a different shape,
`/kaggle/input/notebooks/<owner>/<kernel-slug>/`. The code-cell of notebooks 03/04/05 handles
both dataset shapes: it uses `repo.zip` when one exists, otherwise it finds the marker file at
any depth under `/kaggle/input` and copies the extracted tree into scratch (`/kaggle/input` is
read-only, and the cells write results next to the code). Do not hard-code
`/kaggle/input/<slug>` — for a dataset upload that path does not exist.

## 1b. The kernels as they stand (2026-10-26)

| Notebook | Kernel slug | Status | Output files |
|---|---|---|---|
| 03 | `bikash24795/aiml-02-notebook-03-densenet121-chest-x-ray` | COMPLETE, 06:52 UTC | `densenet121_lungopacity.pt`, `densenet_last.pt`, `densenet_metrics.json`, `densenet_training.png` |
| 04 | `bikash24795/aiml-02-notebook-04-grad-cam-and-fairness` | COMPLETE, 07:17 UTC | `gradcam_fairness_summary.json`, `gradcam_focus.csv`, `gradcam_gallery.png`, `missing_data_audit.csv`, `test_predictions_with_metadata.csv` |
| 05 | `bikash24795/aiml-02-notebook-05-monai-spleen-segmentation` | version 3 failed (OOM, 192³); version 4 pushed 2026-10-26 — **terminal status unverified**, the local CLI is unavailable (§4b) | `spleen_metrics.json`, `spleen_dynunet_best.pt`, `spleen_dynunet_last.pt`, `spleen_per_case.csv`, `spleen_overlays.png` |

Kernels 03 and 04 are final — **do not re-run them**; their outputs are already pulled into
`results/` and `artifacts/`. Notebook 04 reads notebook 03's checkpoint from
`/kaggle/input/notebooks/bikash24795/aiml-02-notebook-03-densenet121-chest-x-ray/`, so that
kernel has to be attached as a *kernel source*, not just left finished.

## 2. Session A — `03_train_densenet.ipynb`

1. kaggle.com → **Code → New Notebook**.
2. Right sidebar → **Accelerator → GPU T4**. Leave **Internet off** — the
   Kaggle image ships torch/pydicom/sklearn, and repo.zip carries the weights.
3. **Add Input** (right panel):
   - *Competitions* → RSNA Pneumonia Detection Challenge → `stage_2_train_images.zip`;
   - *Your Datasets* → `bikash24795/aiml02-code`.
4. Import the notebook: in the editor **File → Import Notebook** → upload
   `notebooks/03_train_densenet.ipynb` from the repo. (If import is not
   offered, copy-paste all cells into a fresh notebook instead.)
5. **Run All.** Watch for, in order:
   - `RUN_ON: kaggle` and the CONFIG printout;
   - zip found (`/kaggle/input/...`), repo extracted, split CSVs loaded with
     label rates;
   - the decode phase (26,684 DICOM → 448 px JPEG) — the long CPU part;
   - epoch rows printing; final cell with **test metrics** and the saved-file
     list (`densenet_metrics.json`, `densenet121_lungopacity.pt`, figure).
6. **Save Version** (top right) → **Save & Run All**. This is what stores
   `/kaggle/working` as the kernel's Output — Session B depends on it.
7. **Screenshots for the report:** the test-metrics cell, the saved-files cell,
   and the Accelerator setting showing T4.

## 3. Session B — `04_gradcam_fairness.ipynb`

1. New notebook, **GPU T4**, Internet off.
2. **Add Input:**
   - RSNA competition data (test images are decoded again);
   - the `bikash24795/aiml02-code` dataset;
   - **Your Code → the notebook-03 kernel** (`bikash24795/aiml-02-notebook-03-densenet121-chest-x-ray`)
     — its Output (checkpoint `densenet121_lungopacity.pt` + `densenet_metrics.json`) lands
     under `/kaggle/input/notebooks/<owner>/<kernel-slug>/`, which is where this notebook hunts;
   - `dicom_headers.csv` **if** you have it — grab it from your Drive `aiml02`
     folder and attach it the same way. Without it the notebook still runs,
     falling back to demographics inside the split CSVs (the cached scan is
     better; plan a scan session later with `tools/scan_dicom_headers.py`).
3. Import `notebooks/04_gradcam_fairness.ipynb`, **Run All**.
4. **Save Version → Save & Run All** so the deliverables persist.
5. **Screenshots for the report:** the Grad-CAM overlay figure, the subgroup
   fairness table, the missing-data audit, the final summary cell.

## 4. Bring the results home

1. Open the kernel's **Output** tab → Download:
   - from notebook 03: `densenet_metrics.json`, the figure PNG, and
     `densenet121_lungopacity.pt`;
   - from notebook 04: the Grad-CAM PNGs, `focus.csv`,
     `test_predictions_with_metadata.csv`, `missing_data_audit.csv`, and the
     summary JSON.
2. Drop the metrics/CSVs/PNGs into the repo's `results/` folder, and the
   checkpoint into `artifacts/`.
3. Tell the chat — it will `dvc add` + `dvc push` the checkpoint (git stores
   only the pointer; `*.pt` is gitignored on purpose) and commit the rest.

## 4b. The local Kaggle CLI is not a working path (2026-10-26)

Do every status check and every output pull in the browser. `kaggle` is not on `PATH`, and the local Python
install cannot be relied on to provide it: `py` points at a missing Python 3.13 interpreter, `python3` is the
Microsoft Store stub, and Python 3.12 has no `kaggle` CLI of its own. Do not build a workflow that assumes
`kaggle kernels status` or `kaggle kernels output` will run on this machine.

Repairing the interpreter (installing 3.13 properly, or a pinned venv with the CLI) is a machine-level fix and
is deliberately **not** part of this repo. Until it is done, the runbook's steps are manual: the kernel page shows
status and the Output tab is the download. The credential file, when the CLI is restored, belongs at
`~/.kaggle/access_token` for the 2.x CLI (the newer single-token format), not `kaggle.json`.

## 5. Gotchas

- **No GPU?** Phone verification missing. Settings → verify phone → re-open the
  notebook.
- **RSNA not in Add Input?** Competition rules not accepted yet.
- **Run died mid-training?** Re-run via Save Version. Outputs already in
  `/kaggle/working` only survive in a saved version.
- **Weekly quota** resets weekly; a full 03+04 cycle is a few GPU-hours, well
  inside it.
- **Never quote a `host: local` run** as a result — the notebooks record the
  host inside their metrics JSON for exactly this reason.

## 6. Session C — `05_segment_spleen.ipynb` (MONAI spleen, Phase 3b)

The notebook now exists (built 2026-10-25): DynUNet + deep supervision + sliding-window
inference, rewritten onto `src/data/ct_volume.py` and `src/models/unet3d.py`, resumable
checkpoints, per-case Dice table with a failure-case overlay. Session shape is the same:
**GPU T4**, attach `repo.zip`, **Run All**, **Save Version**.

One difference — the dataset: MSD Task09_Spleen is *not* a Kaggle competition input. Either
attach `Task09_Spleen.tar` as a private dataset (URL + expected md5 are pinned in
`results/msd_spleen_manifest.json`), or leave Internet **on** for the dataset cell and let
`tools/get_msd_spleen.py` download + md5-verify it itself. Everything else (MONAI/nibabel
wheels, ImageNet weights) still works offline.

Two things were learned by running it on 2026-10-26, and the notebook now carries both:

- **192³ crops at batch 2 do not fit a T4.** The first attempt allocated 14.28 GiB of the
  14.56 GiB card on training step 1 and died in `DiceCELoss`. The `CONFIG` default is now
  `roi = [128, 128, 128]`, the patch size MONAI's own spleen tutorial trains on.
- **Whole-volume load + resample dominated the epoch.** `patch_stream` pulled a fresh CT per
  patch; `load_cached` now keeps the 28 training volumes in RAM (about 2.7 GB at
  334×334×181 float32 + mask), because loading one volume takes seconds and cropping it
  does not.

The budget run is `epochs_total = 90` at 64 steps/epoch; the full schedule is 100–200
epochs, and `spleen_metrics.json` records the config so a 90-epoch number is never quoted
as a longer one.

### 6.1 The 40-epoch run scored 0.329 Dice — and the bug was the sampler, not the GPU

The 2026-10-26 T4 run finished green in 50 GPU-minutes and reported mean test Dice **0.329**
(predicted volumes 737–1,777 ml against 55–361 ml of reference: 7–20× over-segmentation,
centroids 114–160 mm away). Three local checks, in this order, located it:

1. **A 128³ crop that contains the spleen scores 0.87**, and still 0.88 with the organ pushed
   40 voxels off-centre. The checkpoint is fine; whole volumes are what break.
2. **A crop with no spleen in it predicts 0.2 % positive** — the net does not hallucinate on
   empty windows, so the usual "background windows cause false positives" story is wrong.
3. **`predict_volume` on one 128³ window is bit-identical to a direct `net(x)`** (Dice 0.874,
   mean |p_direct − p_sw| = 0.0000), so the sliding-window inferer is not the culprit either.

That leaves the training distribution. `sample_patch` was drawing every positive crop centred
on an organ voxel — the spleen is always near the middle of a positive window — and every
negative crop from the *volume centre*, which is one fixed liver/stomach view per case. A net
can fit that with "paint a spleen-shaped blob near the middle of whatever window I get", which
is exactly the 0.87-crop / 0.33-volume split.

Fixed in `src/data/ct_volume.py` (`sample_patch`, 19 tests still pass): positive centres are
jittered by up to a quarter of the ROI so the organ is not always centred, and negatives are
drawn uniformly from the volume instead of the centre. 40 epochs was also far too few —
2,560 optimizer steps for a 3D U-Net — so the next run uses 90 (~1.8 GPU-hours).

### 6.2 The 90-epoch rerun: 0.8586, and the sampler was the whole story

Kernel version 8 ran on 2026-10-27 (T4, 119.3 GPU-min wall clock, 90 epochs at 128 patches each) and
reported mean test Dice **0.8586** — median 0.9124, range 0.5972–0.9480, `n_zero_dice` 0, `n_below_0_5` 0.
Quick-val peaked at 0.9357 on epoch 79, which is the checkpoint the test section loaded.

| epoch | 9 | 19 | 29 | 39 | 49 | 59 | 69 | 79 | 89 |
|---|---|---|---|---|---|---|---|---|---|
| quick-val Dice | 0.305 | 0.665 | 0.885 | 0.906 | 0.695 | 0.837 | 0.540 | **0.936** | 0.899 |

Three things to take from it:

- **Only the sampler changed between 0.329 and 0.8586.** No architecture, ROI, loss or inference change;
  the epoch count went 40 → 90, but the diagnostic above showed a correctly-sampled 40-epoch checkpoint
  already scoring 0.87 on spleen-containing crops. If a future run regresses, re-check `sample_patch`
  before touching hyperparameters.
- **The val curve is noisy, not monotone** (0.906 → 0.695 → 0.837 → 0.540 on 3 cases). Three validation
  volumes cannot rank checkpoints reliably; the best-checkpoint rule is what kept 0.9357, and the final
  test score on 6 unseen volumes is the number to quote.
- **Failure mode moved from "wrong shape" to "under-segmentation".** The worst case, `spleen_31`
  (Dice 0.597), predicts 159 ml against a 357 ml reference with its centroid 8.1 mm off — it finds the
  organ and labels part of it, instead of painting a blob in the middle of the window as the old model did.

Pushing a rerun takes three steps, all used here: rebuild `repo.zip` from the working tree, upload it as
a new version of the private dataset (`kaggle datasets version -p <stage> -m ...`), then
`kaggle kernels push -p <stage>` — the kernel picks up the new dataset version at the next run. Logs are
only served after the run finishes, so poll `kaggle kernels status` rather than trying to stream.

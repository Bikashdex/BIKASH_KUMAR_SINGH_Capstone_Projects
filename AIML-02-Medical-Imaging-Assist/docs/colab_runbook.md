# Colab runbook — notebooks 03 and 04

> **Status 2026-10-25: the Colab quota is exhausted — use
> [`docs/kaggle_runbook.md`](kaggle_runbook.md) (Kaggle T4) as the primary
> runbook for now.** Both notebooks auto-detect the host, so everything below
> stays valid when the quota resets.

How to take the training notebooks from "written" to "ran on a real GPU", and what
evidence to bring back. Everything below matches the paths the notebooks already
search, so there is nothing to edit in the notebooks themselves.

---

## 0. One-time prep (do once, ~30–60 min + upload time)

- [ ] **Google Drive folder:** create `aiml02` in My Drive.
- [ ] **The dataset zip.** Notebook 03 wants `stage_2_train_images.zip`
      (3,534,067,864 bytes — the exact size is checked) in **My Drive root** or
      `aiml02/`. It is not on the laptop anymore (only the DVC pointer is), so:
      - Kaggle account → Account → "Create New API Token" → save `kaggle.json`.
      - `kaggle competitions download -c rsna-pneumonia-detection-challenge`
        (or the big Download button on the competition page).
      - Upload the zip to Google Drive `aiml02/`. One-time 3.3 GB upload.
- [ ] **The code zip.** Upload the repo as `repo.zip` to `aiml02/`.
      The notebooks also accept it dragged into the Colab Files panel, or at the
      My Drive root — those are the only choices, no path edits needed:
      - `MyDrive/repo.zip`, `MyDrive/aiml02/repo.zip`, or `MyDrive/Colab Notebooks/repo.zip`.
      - Make the zip *after* the latest push so it matches git: zip the repo
        folder without `.git`, `data/`, `artifacts/`, `frontend/node_modules`.
- [ ] **`dicom_headers.csv`** for notebook 04's fairness step: check it exists in
      `aiml02/`. If it is missing, tell the chat — we run
      `tools/scan_dicom_headers.py` on Colab first, because notebook 04 reads
      `MyDrive/aiml02/dicom_headers.csv` directly.

## 1. Session A — `03_train_densenet.ipynb` (DenseNet fine-tune)

1. https://colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU.**
3. Drag `03_train_densenet.ipynb` into the Files panel, open it.
4. **Runtime → Run all.** When the Drive-mount cell runs, click the link, sign in,
   paste the code.
5. What you should see, in order:
   - `RUN_ON: colab` and the CONFIG json with the Drive paths.
   - The zip found, repo extracted, `split_train/val/test.csv` loaded with label
     rates printed.
   - Decode phase (26,684 DICOM → 448 px JPEG, 4 workers) — the long CPU part.
   - Training epochs printing one json row each; `densenet_last.pt` saved each
     epoch into the Drive artifacts folder, so a disconnect is not fatal.
   - Final cell: **test metrics printed**, then the saved-file list.
6. **Evidence to capture (screenshots):**
   - the test-metrics cell (AUROC etc.),
   - the final "saved files" cell showing `results/densenet_metrics.json` and
     `artifacts/densenet121_lungopacity.pt`,
   - Runtime → Change runtime type window showing T4 (proof it was a GPU run —
     `host: colab` is also recorded *inside* `densenet_metrics.json`).
7. Verify in Drive: `aiml02/results/densenet_metrics.json`,
   `aiml02/artifacts/densenet121_lungopacity.pt`, and the figure PNG exist.

**Budget:** decode ~20–40 min + training a few hours on T4. A free Colab day is
enough. Never quote a `host: local` run as a result — the notebook itself says so.

## 2. Session B — `04_gradcam_fairness.ipynb`

Needs Session A's checkpoint in `aiml02/artifacts/` (its search path includes
`/content/drive/MyDrive/aiml02/artifacts/densenet121_lungopacity.pt`).

1. New Colab notebook, **T4 GPU**, drag notebook 04 in, **Run all**, authorise Drive.
2. What you should see:
   - checkpoint found and its provenance (which run produced it) printed,
   - test-split decode (only the images it needs),
   - overall metrics, Grad-CAM figure(s), fairness subgroup AUROC table with the
     largest gap, missing-data audit table.
3. **Evidence to capture (screenshots):**
   - Grad-CAM overlay figure (does the heat sit on the lungs?),
   - the subgroup fairness table,
   - the final summary cell.
4. Verify in Drive `aiml02/results/`: gradcam PNG, `focus.csv`,
   `test_predictions_with_metadata.csv`, `missing_data_audit.csv`, and the
   summary json the last cell writes.

## 3. Bring the results home (after each session)

The files are already in Drive; get them into the repo like this:

1. From Colab's Files panel (or Google Drive) download the new files from
   `aiml02/results/` and drop them into the repo's `results/` folder.
2. The checkpoint `densenet121_lungopacity.pt` goes to `artifacts/` **via DVC**,
   not git: `dvc add artifacts/densenet121_lungopacity.pt` then
   `dvc push` (git only stores the pointer — `*.pt` is gitignored on purpose).
3. Tell the chat — it will commit `results/` and push, and check the figures
   render in the README/report.

## 4. What is NOT covered here yet

- **Phase 3b (MONAI spleen, DynUNet):** there is no notebook 05 yet. That is a
  coding task done in the repo first (port MONAI's spleen tutorial onto
  `src/models/unet3d.py`), then a Colab session the same way as above. The
  dataset URL + expected MD5 are already pinned in
  `results/msd_spleen_manifest.json`, and `tools/get_msd_spleen.py` downloads it.
- **API/Docker end-to-end test** (Phase 4): no Python on the laptop, so this
  happens with `docker compose up` once Docker Desktop is running — separate
  checklist.

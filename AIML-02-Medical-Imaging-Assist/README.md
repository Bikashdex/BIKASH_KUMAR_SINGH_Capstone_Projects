# AIML-02 — AI-Assisted Medical Imaging and Diagnosis System

> ## ⚠ For education only. Not for clinical use.
>
> This system is an internship capstone project. It is **not** a medical device, has **no regulatory clearance**, and
> must **never** be used to diagnose, screen, or treat a patient. Any output shown here is illustrative only.

[![ci](https://github.com/Bikashdex/AIML-02-Medical-Imaging-Assist/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bikashdex/AIML-02-Medical-Imaging-Assist/actions/workflows/ci.yml)

**Intern:** Bikash Kumar Singh · **ID:** NEX-INT-2026-0134 · **Track:** AI / Machine Learning
**Company:** Nexvia Technologies · **Project:** AIML-02 (of 2) · **Deadline:** 10 November 2026

## What this is

A radiologist uploads a chest X-ray to a DICOM server. A Python service notices the new study, reads it, runs a deep
learning classifier and returns the probability with the threshold that produced it; a small ingest worker does that
unattended for every instance Orthanc receives. A React viewer shows the scan, the score and **Approve / Reject**
controls, and every decision is appended to an audit log in SQLite. Grad-CAM maps exist, but they are produced
**offline** for the fairness study rather than served per request — the viewer's overlay toggle is disabled and says so.

A second, independent track segments the spleen from abdominal CT volumes using a **3D U-Net in MONAI**.

## Two model tracks

| Track | Task | Dataset | Model | Headline metric |
|---|---|---|---|---|
| A | Lung opacity vs not (binary classification; opacity is the sign RSNA used as the pneumonia surrogate) | [RSNA Pneumonia Detection, stage_2](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) — Option 4 in the brief; 26,684 DICOM files | DenseNet121, ImageNet transfer learning | AUROC |
| B | Spleen segmentation from CT | [Medical Segmentation Decathlon Task09](http://medical-segmentation-decathlon.org/) — Option 5 | 3D U-Net (`monai.networks.nets.DynUNet`) | Dice score |

## Tech stack

Required by the brief, and where each one is used:

| Component | Used for |
|---|---|
| **PyTorch** | Both model tracks |
| **MONAI** | 3D U-Net, NIfTI loading, CT intensity windowing |
| **Grad-CAM** | Classification explainability |
| **Orthanc** | DICOM server that receives and serves studies |
| **DVC** | Data and model versioning, remote on Google Drive |
| **Docker** | Containerising the inference API and Orthanc |
| **React** | Radiologist review viewer |

## Repository layout

```
├── README.md                  you are here
├── docs/
│   ├── architecture.md        system diagram + data flow + API contract
│   ├── dataset_card.md        provenance, licence, label policy, splits
│   ├── model_card.md          limits, bias, intended use  ← required by the brief
│   ├── demo_script.md         10-minute recording script: shot list, verified commands, narration
│   ├── reflection.md          individual reflection (draft to personalise)
│   └── presentation/          the 13-slide deck (aiml02-presentation/AIML-02-Medical-Imaging-Assist.pptx)
├── notebooks/                 01-02 run locally, 03-05 on Kaggle (also Colab; CPU-tested locally)
├── src/
│   ├── data/                  DICOM I/O, patient-level splits, datasets
│   ├── models/                classifier, 3D U-Net, Grad-CAM
│   ├── api/                   FastAPI inference service
│   ├── ingest/                Orthanc poller → inference → audit log
│   └── monitor/               input drift detection
├── frontend/                  React viewer
├── tests/                     pytest suite (runs in CI)
├── configs/                   YAML config, hyperparameters
├── artifacts/                 model weights (git-ignored, tracked by DVC)
├── weights/                   pretrained ImageNet backbones, .pth only (git-ignored, shipped inside repo.zip
│                              so a Kaggle kernel with no internet can still load DenseNet121)
├── wheelhouse/                Linux wheels for the API image. Empty in git (.gitkeep); fill it before
│                              `docker compose up --build` on a fresh clone — see "Fresh clone on Windows"
├── data/                      raw + processed data (git-ignored)
├── results/                   metrics.json (index of the headline numbers), per-track JSONs, plots/, split CSVs
├── Dockerfile
└── docker-compose.yml         orthanc + api + ingest
```

## Quick start

> **Use Python 3.11, not 3.13.** The pinned `torch==2.4.1` has no cp313 wheels, so `pip install -r requirements.txt`
> fails on a 3.13 interpreter. 3.11 also matches the `python:3.11-slim` image used by `Dockerfile`. If 3.11 is not
> installed, get it from python.org or `winget install Python.Python.3.11` and create the venv with
> `py -3.11 -m venv .venv`.
>
> If `python`, `pip`, `dvc` and `jupyter` all stop working at once on Windows with "The system cannot find the file
> specified" while `pythonw.exe` and `Lib\site-packages` are still present, the interpreter binary itself has gone
> missing — this project's development machine hit exactly that. Look for the cause in *Windows Security → Protection
> History* (this machine's Recycle Bin was empty, so a delete is not what did it), add an exclusion for the Python
> install folder, then run the original installer and choose **Repair**. Repair restores the binary and keeps every
> installed package; uninstalling would throw away the whole environment.
>
> **Read this before you reproduce the numbers.** The executed notebooks in `notebooks/` and
> `results/baseline_metrics.json` were produced on this machine's existing Python 3.13.1 with torch 2.11.0+cpu and
> numpy 2.4.3 — *not* the pinned environment above, which cannot be installed on 3.13. The exact versions are
> captured in `results/environment.json`. Building the 3.11 venv below gives you a working deployment environment,
> but it may reproduce the baselines to a few decimals rather than exactly. This is disclosed, not hidden.

```bash
# 1. Environment
py -3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 2. Launch the Docker daemon (Docker Desktop must be *running* on Windows)
docker compose up --build
# 3. Inference API docs (host port is ${API_PORT:-8000}; this machine runs 8010 — see "Fresh clone on Windows")
open http://localhost:8000/docs
# 4. Orthanc UI
open http://localhost:8042
# 5. Tests
pytest -q
```

### Fresh clone on Windows — what the 2026-11-03 redeployment added

The quick start above describes the development machine: filled wheelhouse, weights present, port 8000 free. A clean
clone onto a second Windows machine (2026-11-03) hit five things in order, each recorded because it cost real time:

1. **The wheelhouse ships empty.** `wheelhouse/` holds only `.gitkeep`, and the `Dockerfile` installs with
   `--no-index --find-links=/wheelhouse`, so a fresh clone dies at `torch==2.4.1` with `from versions: none`.
   Fill it from the host (Python 3.11) before building:

   ```bash
   py -3.11 -m pip download -r requirements-api.txt -d wheelhouse --find-links wheelhouse \
       --only-binary=:all: --platform manylinux2014_x86_64 --platform linux_x86_64 \
       --python-version 311 --implementation cp \
       --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
   ```

   Both `--platform` values are required: numpy and pydantic-core are tagged `manylinux2014_x86_64` while the torch
   CPU wheel is tagged `linux_x86_64`. `--find-links wheelhouse` stops pip re-downloading wheels already present.
   The closure is 28 wheels; fetching only torch+torchvision fails later on torch's own dependencies (`filelock`
   first).
2. **Weights do not travel with the clone.** `artifacts/` is git-ignored; copy `densenet121_lungopacity.pt` (~27 MB)
   from the Kaggle kernel output, the Drive `aiml02/artifacts/` folder, or an older checkout. `/health` reports
   `model_loaded: false` until it arrives; no container restart is needed — `load_bundle()` re-checks the path on
   every request.
3. **Container names are global.** `container_name:` pins `aiml02-*`; containers left behind by an earlier compose
   *project* on the same host keep those names, and `docker compose down` will not reap them (different project
   label). `docker rm -f aiml02-orthanc aiml02-api aiml02-ingest` clears the way.
4. **Port 8000 can be held by Docker itself.** On Windows the publish proxies (`com.docker.backend.exe`,
   `wslrelay.exe`) can keep a port after the container that owned it is gone, and a stale host-level API process can
   answer `/health` with the old shape (`{"status":"healthy",…}`, no `device` field) — check any `/health` you trust
   for `device` and `disclaimer` before believing it. Diagnose with `netstat -ano | findstr :8000`; restart Docker
   Desktop, or move the host port, which the compose file supports: `.env` → `API_PORT=8010` and `frontend\.env` →
   `VITE_API_TARGET=http://localhost:8010`. A container that *started* with a failed bind never re-applies it: free
   the port, then `docker compose up -d --force-recreate`.
5. **Vite reads `VITE_API_TARGET` once at startup.** Editing `frontend\.env` without restarting `npm run dev` leaves
   the viewer proxying at the old target; the symptom is a badge built from a foreign `/health` ("model loaded on
   undefined"). Restart the dev server after changing it.

Re-verified end to end on that clone (2026-11-03): viewer → `/predict` → Approve/Reject → audit rows 1–2 (`0.9434`
approved, `0.1826` rejected, read back out of SQLite with `docker compose exec`); `curl -X POST --data-binary
@positive.dcm http://localhost:8042/instances` → poller → `[ingest] 350cda31-… -> opacity p=0.943 (358.12ms)` after
three retried forwards (only a successful forward counts as seen, so a transient API timeout costs a retry, not a
study). Demo files by patient id: positive = `0022073f-cec8-42ec-ab5f-bc2314649235` (label 1), negative =
`000924cf-0f8d-42bd-9158-1af53881a557` (label 0). The negative reproduces the graded `0.1826` exactly — same file,
same checkpoint, same transform, new stack. The `0.8685` quoted in the Phase 4/5 rows belongs to a different positive
file, `019d950b-dd38-4cf3-a686-527a75728be6` (the one in `docs/architecture.md`'s example response).

## Getting the data

Track A uses the RSNA stage_2 archives. `data/` is git-ignored, so the raw files never enter the repo; they are
tracked by DVC instead. Measured on disk: the competition download is 3.5 GB of zips and extracts to about 3.4 GB,
so allow ~7 GB while unzipping and get back to ~3.4 GB once the archives are deleted.

```bash
pip install kaggle
mkdir -p data/raw/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/raw/rsna
cd data/raw/rsna
unzip stage_2_train_labels.csv.zip && unzip stage_2_detailed_class_info.csv.zip
unzip stage_2_train_images.zip -d stage_2_train_images
# ~26,684 .dcm files. Keep stage_2_train_images.zip - it is both the single DVC transport object and
# the one file Colab can actually receive (26,684 loose uploads is not a plan).

# Cache every DICOM header once, so the notebooks never re-read 26,684 files:
python tools/scan_dicom_headers.py --images data/raw/rsna/stage_2_train_images \
    --out data/raw/rsna/dicom_headers.csv

# Then run, in order:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_logreg.ipynb
```

`notebooks/01_eda.ipynb` writes `results/eda_summary.json`, `results/split_{train,val,test}.csv` and
`results/train_pixel_stats.json`. The split CSVs store image paths relative to the data folder, so the same split
works locally and in Colab — point `RSNA_DATA_DIR` at the new location and nothing else has to change.

## Running the GPU notebooks (Kaggle, Colab, or verified on CPU)

Notebooks 03 and 04 need a GPU, not this laptop. Both are written to run unchanged on Kaggle, Colab or this
laptop: cell 4 sets `RUN_ON = "kaggle" if os.path.isdir("/kaggle") else ("colab" if os.path.isdir("/content") else "local")`
and picks a path table from it, so there is one notebook and no forked copy to keep in sync. `local` is not
a shortcut — it is how every code cell in both notebooks got executed on CPU before handing them over, with tiny
subsample sizes and `device = cpu` (the notebooks only allow that when `n_predict <= 600`).

They also import the real package (`from src.data.dicom_io import to_uint8`) rather than pasting a second copy of
the windowing code, so the hosted run needs the code as well as the images. It gets that from
`repo.zip` (~32 MB, 58 files) — `src/`, `notebooks/`, `configs/`, `tools/`, `results/`, `tests/`, the requirement
files, and `weights/densenet121-a639ec97.pth`. Never include `data/`, `.dvc/config.local`, `.git`, `kaggle.json` or
`__pycache__`; the build script re-opens the archive and asserts those are absent.

**Kaggle (the route in use now — Colab's free tier stopped handing out T4 sessions after ~4 disconnects):**

- Kaggle → Create → Notebook. In the right sidebar set **Accelerator → GPU T4 or P100** and leave **Internet off**
  — the pretrained backbone comes from `repo.zip`, which is why it is 32 MB rather than 2 MB.
- **Add Input → Competitions → rsna-pneumonia-detection-challenge.** That mounts the 3.3 GB
  `stage_2_train_images.zip` under `/kaggle/input/`, so nothing has to be uploaded.
- Attach `repo.zip` too — either Add Input → Files (upload it), or turn on Settings → Google Drive and keep
  it in `My Drive/aiml02/`. Cell 6 globs `/kaggle/input/**/repo.zip` and `/kaggle/working/repo.zip` rather than
  assuming one location.
- Download `03_train_densenet.ipynb` from `My Drive/aiml02/` and import it (File → Import Notebook in the
  kernel's `...` menu) — do not paste cells by hand, because pasting silently loses the comment that explains
  which host is which. **Save version before leaving**; outputs land in `/kaggle/working`
  (`densenet_metrics.json`, `densenet121_lungopacity.pt`, `densenet_last.pt`, `densenet_training.png`) and Kaggle
  keeps them as the kernel's output. Scratch (extracted DICOMs, JPEG cache) goes to `/tmp` because `/kaggle/working`
  is what gets saved.
- Kaggle free tier gives ~30 h/week of GPU, and a session is capped at 12 h — enough for the ~2 h run below.
- Notebook 04 is the same competition input plus notebook 03's output: after saving 03's version, reopen
  **Add Input → Your Code → Accelerated Computed Tomography Scanning… (the 03 kernel) > the saved version**.
  Its files appear under `/kaggle/input/<kernel-slug>/` (the slug is the kernel's title in lower-case hyphens — the
  default is `quick-start-notebook` until you rename it), which is exactly where cell 8 looks for
  `densenet121_lungopacity.pt` and `densenet_metrics.json`; a missing checkpoint stops the run with that menu
  spelled out instead of failing later. Outputs: `gradcam_gallery.png`, `gradcam_focus.csv`,
  `test_predictions_with_metadata.csv`, `missing_data_audit.csv`, `gradcam_fairness_summary.json`.

**Colab, if the quota comes back:** open the Drive copy of the notebook, Runtime → Change runtime type → T4 GPU,
upload `repo.zip` to `My Drive/aiml02/` (or drag it into Files), make sure `stage_2_train_images.zip` is in My Drive,
and run every cell. Updating the notebook through right-click → Manage versions → Upload new version keeps the
same link; a plain re-upload creates a second file and it is easy to run the stale one.

Time budget on a T4, measured: unzip ~10 min, DICOM → 448 px JPEG cache ~25 min, then roughly 10-15 min per epoch with
patience 3 stopping it early. `densenet_last.pt` is rewritten after every epoch because a session disconnected at
epoch 8 would otherwise lose the whole run.

If the download is blocked to you, `python tools/make_synthetic_rsna.py` generates 108 valid DICOM files with the
same schema so the pipeline can still be smoke-tested. Say plainly in the dataset card that any figure produced that
way is synthetic — it is not a result.

## Versioning the data with DVC

`data/` never enters git; the `.dvc` pointer files next to it do. Google Drive is the remote target because this
machine has ~24 GB free and Track B adds another ~2.3 GB. What is done, and what is not:

| step | state |
|---|---|
| `dvc init` | done |
| `dvc add` of the 26,684 images + 3 CSVs | done — dir hash `3fd56c2857f39d9f9df084ddadd7c8de.dir` |
| `dvc add` of `stage_2_train_images.zip` | done — md5 `a8307fbc6d9138a7760dafde21383714`, 3,534,067,864 B. This is the *transport* object; see "one object, not 26,688" below |
| `.dvc` pointers committed | done — git alone tells a reviewer which exact 26,684 bytes (md5 + total size) every number was computed on. Getting the pixels themselves still needs the remote below |
| `dvc remote add gdrive …` | done — dedicated Drive folder `aiml02-dvc-remote`, id `1_MZ3txKpgRucQ5Ep8YxZAbKovIzDJFiL`, held in `.dvc/config` (so it is public once the repo is pushed; keep that folder private) |
| `dvc push` | done and verified — 4 objects, 3,546,066,217 B in the remote: the archive `a8307fbc6d9138a7760dafde21383714` (3,534,067,864 B), `stage_2_train_labels.csv` (1,490,034 B), `stage_2_detailed_class_info.csv` (1,647,396 B) and `dicom_headers.csv` (8,860,923 B). `dvc status -c` reports *Cache and remote 'gdrive' are in sync* and each Drive object's size was read back over the API. The image **folder** is deliberately not pushed — 26,688 objects for bytes already inside the archive. How it was unblocked: "What it actually took" below |

**Disk cost, measured rather than assumed.** DVC's docs promise the cache is free because it hardlinks. Here it is
free for *single-file* targets and not free for *directories*. After `dvc config cache.type hardlink,copy` and a full
cache rebuild: 26,689 cache objects, 6.62 GiB, and exactly **4** of them have `st_nlink > 1` — the zip (3,534,067,864 B)
and the three CSVs, which is precisely the set of individually-tracked files. The 26,684 images inside the tracked
folder are still byte-for-byte *copies*, so `dvc add` of that folder really did cost ~3.1 GB. `os.link()` succeeds on
those `.dcm` files when called directly, so this is DVC's per-file fallback inside directory imports, not a filesystem
limitation. Two consequences worth knowing: the archive is the only thing *worth pushing*, and freeing the folder's
copies means untracking the folder — which this project deliberately does not do, because the folder pointer is the
per-image md5 provenance record.

**One object, not 26,688.** `find .dvc/cache -type f | wc -l` → 26,688. `dvc push` uploads one Drive object per
cache file, so pushing the image folder means ~26,700 HTTP requests and Drive's rate limiter will make that hours.
The archive is therefore what gets pushed; the folder pointer stays in git purely as the per-image md5 record.

The commands that produced that state, for anyone reproducing from a clean clone:

```bash
dvc init
dvc config cache.type hardlink,copy
dvc add data/raw/rsna/stage_2_train_images data/raw/rsna/stage_2_train_labels.csv \
        data/raw/rsna/stage_2_detailed_class_info.csv data/raw/rsna/dicom_headers.csv
dvc add data/raw/rsna/stage_2_train_images.zip     # the single object that actually gets pushed
git add data/raw/rsna/*.dvc && git commit -m "chore: track RSNA data with DVC"

# One-time: create a folder in your Drive, open it, and copy its id from the URL.
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify gdrive gdrive_acknowledge_abuse true   # chest X-rays trip Drive's abuse scan
dvc push data/raw/rsna/stage_2_train_images.zip.dvc      # push the archive; the folder is provenance only
```

**Why the default `dvc push` is blocked, from the installed source rather than from a forum.** `dvc-gdrive` ships one shared OAuth
client that every user of the plugin inherits (`dvc_gdrive/__init__.py:23-24`, client id
`710796635688-…apps.googleusercontent.com`), and `pydrive2` requests the full `drive` scope with it
(`pydrive2/fs/spec.py:26-29`). Google classifies that scope as sensitive and refuses to grant it to an unverified
third-party app, so the consent screen ends at **"This app is blocked"**. Retrying does not help: the block is on the
shared client, not on your account.

Two ways forward.

**Option A — register your own OAuth client. Done, and this is the exact sequence that worked.**
In Google Cloud Console, as the same Google account that owns the Drive: create/open a project → enable the
Google Drive API → OAuth consent screen (type External) → add your own Google address under **Test users**
(this is the step that gets past the block) → Credentials → OAuth client ID → application type **Desktop app**.
Then, in the repo:

```bash
dvc remote modify --local gdrive gdrive_client_id     <CLIENT_ID>
dvc remote modify --local gdrive gdrive_client_secret <CLIENT_SECRET>
dvc push data/raw/rsna/stage_2_train_images.zip.dvc
```

Three things cost time and are worth writing down:

- **Enabling the API is not the same as creating the client.** With the client configured but `drive.googleapis.com`
  still off, consent succeeds and then the first API call answers `accessNotConfigured`: "Google Drive API has not
  been used in project … before or it is disabled."
- **The folder in `gdrive://<FOLDER_ID>` must already exist, and the id must be its real one.** A nonexistent id does
  not produce a friendly error: `pydrive2` asks the folder for its `driveId`, gets HTTP 404, and turns that into
  `PermissionError` (`pydrive2/fs/spec.py:284-300`). Create the folder in Drive, open it, and copy the long id from
  the URL.
- **Consent can report a false rejection.** `dvc push` starts a server on `localhost:8080` and waits for the redirect;
  if a stale tab hits that port without a `code`, pydrive2 raises `AuthenticationRejected` ("User rejected
  authentication") even though nothing was refused. Close stray `localhost` tabs and retry — the second attempt on
  this machine returned `Authentication successful.`

**Desktop app, not Web — this is the part that costs an hour if you guess.** pydrive2 authenticates by starting a
local web server on `http://localhost:8080/` (falling back to 8090) and having Google redirect the code to it
(`pydrive2/auth.py:249-269`). Loopback redirect URIs are allowed for installed client types and refused for Web
application clients, so a Web client fails the flow before consent with `redirect_uri_mismatch`. That error is
reproducible without a browser:

```bash
curl -s -D - -o /dev/null "https://accounts.google.com/o/oauth2/auth?client_id=<ID>&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2F&response_type=code&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive&access_type=offline" | grep -i '^location'
```

A `Location:` header containing `oauth/error?authError=` means the client is not usable yet; a redirect straight to
the consent page means it is.

`--local` is not a style preference: it writes `.dvc/config.local`, which is git-ignored, whereas `.dvc/config` is
committed. Putting the client secret there would publish it in the same repo whose CI workflow exists to keep
credentials out. With an unverified app the consent page still warns "Google hasn't verified this app" — choose
Advanced → Go to … (unsafe) — and refresh tokens stop working after 7 days, so expect to re-authenticate weekly.

**Option B — no third-party OAuth at all (the route `notebooks/03_train_densenet.ipynb` uses).** DVC keeps doing the
versioning job locally, and transport to the GPU machine goes through Colab's own Drive mount, a first-party
integration that needs no app registration:

```bash
# laptop: upload the original 3.3 GB stage_2_train_images.zip (or a zip of .dvc/cache) to MyDrive/aiml02/
# colab:  drive.mount("/content/drive"), then either unzip, or use the mount as a *local* DVC remote:
dvc remote add --default drivecache /content/drive/MyDrive/aiml02/dvc-cache
dvc pull
```

`notebooks/03_train_densenet.ipynb` implements Option B's unzip path. This substitution is recorded in
`docs/dataset_card.md` as well, because "the DVC remote is not reachable from the submission environment" is a
limitation of the work, not a detail.

Two other things to know before you type `dvc` commands:

- `requirements.txt` pins `pathspec==0.12.1` on purpose. DVC 3.55 imports a symbol pathspec 1.x deleted, so on a
  fresh environment without that pin every `dvc` command dies with `ImportError: cannot import name '_DIR_MARK'`.
  CI would not catch it — CI installs a deliberately light dependency set and skips torch.
- `dvc push` / `dvc pull` also need `pydrive2[fsspec]==1.21.1` with `pyOpenSSL==26.4.0`; newer pydrive2 caps pyOpenSSL
  at 24.2.1, which cannot coexist with the `cryptography>=48` that `scmrepo` / `asyncssh` require, and `dvc push` dies
  with `module 'lib' has no attribute 'GEN_EMAIL'`.

Training runs on a free hosted GPU - Kaggle now, Colab if its T4 quota comes back — see `notebooks/`. This
project's development machine has 8 GB RAM and no GPU, so nothing here should be trained locally. Notebooks 01 and
02 do run locally (CPU, tabular work); 03 and 04 detect the host and treat a laptop as a smoke test only.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo scaffold | ✅ done |
| 1 | EDA + leakage-safe splits | ✅ done — `notebooks/01_eda.ipynb`, executed on the real 26,684 images |
| 1 | DVC versioning | ✅ `dvc init` + `dvc add` + committed pointers + `dvc push` of the archive and 3 CSVs, verified by reading each object's size back from the Drive API and by `dvc status -c` reporting sync. Needed a personal **Desktop app** OAuth client and a real Drive folder id; both are recorded above so the next person does not rediscover them |
| 2 | Baselines → DenseNet121 | ✅ trained on a **Kaggle T4** (2026-10-26, 8 epochs, early stop at 4): **test AUROC 0.8819**, val 0.8882, PR-AUC 0.7085, at threshold 0.5108 sens 0.7661 / spec 0.8055. Beats the 0.740 pixel baseline by +0.142 (`results/densenet_metrics.json`), confusion matrix at that threshold in `results/plots/confusion_matrix.png` (TN 2,498 / FP 603 / FN 211 / TP 691 over the 4,003 test images, rebuilt from the saved per-image predictions). `results/metrics.json` is the one-file index of every headline number with its source artifact beside it, and `tests/test_metrics_summary.py` fails if the index drifts from those files. The frozen-ResNet row scoring 0.7250 is the measured argument for fine-tuning |
| 3 | Grad-CAM + fairness | ✅ run on the **Kaggle T4** over all 4,003 test images: AUROC 0.8819 (CI 0.8697–0.8916), 300 Grad-CAM maps scored for attention inside the body (43 % under half), subgroup tables for sex, age band and label source, missing-data audit all zero (`results/gradcam_fairness_summary.json`). Calibration measured post hoc: over-confident, ECE 0.1256 |
| 3b | MONAI 3D U-Net | ✅ rerun on the **Kaggle T4** (2026-10-27, 90 epochs, 119.3 GPU-min, kernel version 8): **mean test Dice 0.8586** — median 0.9124, range 0.5972–0.9480 over the 6 held-out volumes, no zero-Dice cases, `n_below_0_5` = 0, best quick-val 0.9357 at epoch 79 (`results/spleen_metrics.json`, per-case table in `results/spleen_per_case.csv`, overlays in `results/plots/spleen_overlays.png`). The first 40-epoch run scored 0.329 because the sampler drew positives from a voxel inside the organ and negatives from the volume centre, teaching the net a blob near the middle of the window; jittered positive centres plus uniformly drawn negatives is the whole difference between 0.329 and 0.8586 |
| 4 | FastAPI + Orthanc + audit log + Docker | ✅ **three containers up together** (`docker compose up -d`, 2026-10-27): `aiml02-orthanc` (Orthanc 1.13, healthy on 8042/4242), `aiml02-api` (healthy on 8010) and `aiml02-ingest`. Proven end to end: a DICOM POSTed to `/instances` arrives as a *merged patient change* in Orthanc's `/changes`, the poller resolves patient → study → instance, fetches the file and posts it to `/predict` — logged as `[ingest] 07cd2950-… -> no_opacity p=0.184 (1006.53ms)`; a repeat listing of the same patient is not scored twice. The audit trail is wired the same way: `POST /audit` appends to `audit.db` in the `audit-data` volume (verified by reading the table back with `docker exec`), `GET /audit/{study_uid}` returns the history, and the viewer's Approve names the row it wrote. `/predict` returns **0.8685** for `positive.dcm` and **0.1826** for `negative.dcm` at threshold 0.5108; latency is **1.0–1.9 s** on an idle machine and **7–14 s** while Docker is pulling or building, versus 0.42–0.48 s in the host venv. Four bugs/gaps found and fixed while wiring this, all recorded because each cost real time: (1) `fetch_dcm` never called `raise_for_status`, so a failed fetch posted Orthanc's 500 body to `/predict` as a study; (2) Orthanc 1.13 rejects a boolean `StableAge` — it must be integer seconds, and the container crash-looped with `OrthancException` until it was; (3) the official image ships neither `curl` nor `wget`, so the healthcheck uses `python3 -c urllib.request…`; (4) the `audit-data` volume came up root-owned and SQLite could not create `audit.db` until the image `chown`ed `/data` to the app user. Still open: no authentication on `/audit`, and `ingest` scores every instance rather than filtering to chest X-rays |
| 4a | Fresh-clone redeployment | ✅ re-verified 2026-11-03 on a clean Windows clone: wheelhouse filled from the host (28 wheels), stack on `API_PORT=8010`, weights copied from an older checkout; viewer → audit rows 1–2 (0.9434 / 0.1826), Orthanc → ingest `p=0.943 (358.12ms)` with retry-on-timeout observed live. Five deployment gotchas recorded in Quick start |
| 5 | React viewer | ✅ exercised 2026-10-27 **against the running full stack**: `positive.png` → **86.9 % opacity**, `negative.png` → **18.3 % no_opacity**, `positive.dcm` → 86.9 % *with its real tags* (patient id, CR, 1024×1024); the 60 % overlay slider and the education banner render, the heatmap toggle stays disabled with `"(none from API yet)"` because `/predict` returns no `overlay_url`. **Approve now writes an audit row**: the panel shows *"Decision written to the audit log as row 4."* and the row is in `decisions` in the volume (`study_uid` falls back to the filename for a PNG, which has no StudyInstanceUID). **Reject was exercised on 2026-11-03** against the redeployed stack (row 2, 0.1826, reason stored), so both decision paths are now verified live; on that run the positive demo file was test-split image `0022073f-cec8-42ec-ab5f-bc2314649235` (scored 0.9434) and the negative was `000924cf-0f8d-42bd-9158-1af53881a557` (0.1826 — the same file behind the 0.1826 quoted above, reproduced exactly on the new stack). Note: a `.dcm` upload cannot be rendered as an image in the browser (`"Browser preview not available"`), so the picture itself comes from the PNG path |
| 6 | Docs, tests, demo, presentation | 🟡 **97 tests pass** in the project venv (`C:\Users\kumar\venvs\aiml02-py312`, pytest 9.1.1, 4 min 57 s under Docker load) and the same suite runs in GitHub Actions on Python 3.11 with CPU-only torch (`.github/workflows/ci.yml`, ~2 min of tests). CI had been red for five pushes without anyone noticing: it installed only the DICOM libraries, but `src/api/main.py` imports torch at module level, so 7 of 10 test modules failed at *collection* and no test ever ran — fixed by installing the CPU stack and MONAI rather than by skipping tests. A new `pytest.ini` pins `testpaths = tests` — a bare `pytest` used to walk `frontend/node_modules` before collecting, which looked like a hang. `results/metrics.json` consolidates the headline numbers and `tests/test_metrics_summary.py` re-reads all five source artifacts to prove the copy has not drifted. Dataset card, both halves of the model card and both runbooks carry measured numbers. **13-slide deck** in `docs/presentation/` follows the brief's narrative (two tracks → numbers → explainability → live service → the sampler bug → limits); `docs/demo_script.md` is a 10-minute shot list re-verified on 2026-10-27 against the running stack (0.8685 / 0.1826, Orthanc → ingest, viewer → audit row, viewer proxying 5173 → 8010); `docs/reflection.md` is a first-person draft marked for personalisation. Outstanding as of 2026-11-04: the recording itself (shot list ready, not yet recorded) — and slide 10 still says Orthanc never pulled, which was true when the deck was built and has since been contradicted by two live runs (2026-10-27 and 2026-11-03, the latter logging `[ingest] 350cda31-… -> opacity p=0.943 (358.12ms)` after three retried forwards); fix the slide before recording |


## 🎬 Demo Video

A complete 10-minute end-to-end demonstration of the system is available here:  
👉 **[Watch the AIML-02 Demo Video](https://drive.google.com/file/d/18_7wiSiSdOaxHUAlI2IAksqW84c1fCmZ/view?usp=sharing)**

*(The video covers: API health check, live DICOM upload via the React viewer, human-in-the-loop Approve/Reject decisions, SQLite audit log verification, and the unattended Orthanc → ingest polling path.)*

## Licence and data terms

RSNA Pneumonia Detection Challenge data is licensed for non-commercial use only. CheXpert and MIMIC alternatives
carry their own data-use agreements. No data in this repository may be redistributed commercially.
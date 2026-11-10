# Individual reflection — AIML-02

> **Draft.** This is written from the project's own work log and commit history, in first person, so it is factually
> accurate about what happened. Read it, cut anything that does not sound like you, and replace the two marked
> `[PERSONAL: …]` lines with your own words before submitting.

**Bikash Kumar Singh** · NEX-INT-2026-0134 · AI/ML Track · AIML-02

## What I set out to build

The brief offered five data options and two tracks. I picked one from each — RSNA chest X-rays for classification and
MSD spleen for segmentation — because I wanted to find out whether the same problems show up in a 2D picture and a 3D
volume. They do, and that turned out to be the most useful thing I learned.

## What I actually did

**Track A — classification.** DenseNet121 with ImageNet transfer learning on 26,684 chest X-rays, grouped split by
patient so no patient appears in two splits. Final test AUROC **0.8819** (CI 0.8697–0.8916) against a 0.740 pixel
baseline. The comparison I am proudest of is not the headline: frozen backbone features with a logistic-regression
head scored 0.7250 on a 1,000-image subset — a small subset, so it is a direction rather than a ranking — and it is
barely above the pixel baseline. Fine-tuning did the work, not the backbone.

**Track B — segmentation.** MONAI DynUNet on 41 abdominal CTs. The first run scored **0.329** Dice. The second scored
**0.8586**. The difference was one function — the sampler.

**The system around them.** A FastAPI service with `/health`, `/model-info` and `/predict`; a React viewer showing the
scan, the probability and an Approve/Reject decision; a Grad-CAM path; an Orthanc ingest poller and a SQLite audit
trail. 83 tests pass.

## The four things that actually taught me something

**1. A number can be true and still be useless.** 0.329 Dice is a real result. It also describes a model that learned
"put a blob near the middle of the crop", because my sampler took positives from a voxel inside the spleen and
negatives from the exact centre of the volume. I only found it by opening the predictions instead of reading the
metric. Jittered positive centres and uniformly drawn negatives moved the same architecture, data and seed from 0.329
to 0.8586. I now look at outputs before I look at scores, and it is not a habit I had a month ago.

**2. The platform is part of the work.** I have no GPU — 8 GB of RAM and CPU only — so everything trained on free
hosted GPU. Colab stopped handing out T4s after about four disconnects, so I moved to Kaggle and had to learn its rules
the hard way: it silently unzips any archive you upload as a dataset, mounts inputs four directories deep, derives the
kernel slug from the notebook's *title* (so a re-push with a mismatched id fails with a 409), and serves empty logs
until the run finishes. I spent more time on that than on model architecture. I would rather have known it upfront,
but knowing it now is worth something: the runbook in `docs/` is written so the next person does not repeat it.

**3. Reading the source beats searching the error.** `dvc push` failed with "This app is blocked" because the
`dvc-gdrive` plugin ships a shared OAuth client that requests the full Drive scope, which Google will not grant to an
unverified app. I found that by reading `dvc_gdrive/__init__.py` and `pydrive2/fs/spec.py`, not by forum-hunting. The
same approach explained a `PermissionError` that was really a 404 on a folder id, and an `AuthenticationRejected` that
was really a stale browser tab holding port 8080. Registering my own Desktop-app OAuth client fixed it. The lesson
generalises: when a library fails, the answer is usually in the library.

**4. Honesty is a deliverable, not a disclaimer.** My README lists what is *not* done, in the same table as what is.
The Orthanc ingest poller and the audit database are written and have never pulled a DICOM, because Docker Desktop's
engine on this machine will not start — the named pipe is missing and `_ping` returns 503; I restarted the daemon, the
WSL distro and the whole app, and it did not come back. The HTML viewer was verified at DOM level because the in-app
browser was hidden, so there is no screenshot, and I say so instead of implying one exists. Two of the findings I
report are unflattering to my own model: 43 % of Grad-CAM maps put less than half their attention inside the body, and
the same model scores about 0.99 AUROC on "opacity" labels and 0.67 on "no opacity" labels. Cutting those would have
made the deck look better and the work worse.

## Where I fell short

- **Time allocation.** I let infrastructure (GPU platforms, Docker, DVC's OAuth) eat days that should have gone to
  experiments. A 30-minute reconnaissance of a platform before committing to it would have saved most of that.
- **Scale.** Six held-out CT volumes is too few to describe a distribution; the mean Dice hides one bad case
  (0.597, predicting 159 ml against 357). A larger test set was feasible and I did not do it.
- **Calibration.** I measured the ECE (0.1256) but never fixed it. Temperature scaling is an evening's work and would
  have made the confidence number honest rather than merely reported.
- **End-to-end integration.** The single most valuable thing a medical imaging demo can show is a scan arriving at a
  server and a prediction appearing without a human touching a file. I have every part of that and none of it
  connected, for a reason outside my control — but I could have run the stack on a machine that could pull images
  instead of only on this one.

## What this internship changed

[PERSONAL: one or two sentences on what you want to do next — e.g. whether this made you want to work on regulated
medical AI, or made you want to work on the evaluation side of it.]

The concrete shift is that I no longer think of a model as the deliverable. The deliverable is a claim someone else can
check: the split, the threshold, the label semantics, the failure mode, and the number that would prove me wrong. The
spleen bug is the clearest example — 0.329 was checkable and wrong, and that is why I caught it.

## What I would do with another month

1. **Finish the wiring** — Orthanc container, ingest poller, audit database, one command to bring the stack up.
2. **Calibrate** — temperature scaling on the held-out split, then re-report ECE and Brier.
3. **Train the spleen model to convergence** — validation was still improving at epoch 79 of 90; a 100–200 epoch
   schedule is the cheapest route to 0.90.
4. **External validation** — a second institution's chest X-rays, because one hospital's images are one distribution.
5. **A real error analysis** — read the 50 worst cases by hand and write down what each one has in common.

[PERSONAL: replace this section with something you actually want to work on, even if it overlaps with the list above.]

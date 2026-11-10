"""Track B's input contract: where Task09_Spleen lives, how the 41 cases split, and what one case becomes.

The reason this is a module and not notebook cells is the same as for `transforms.py` on the chest track: if
the CT windowing lives in one place and the resampled spacing lives in another, the mask a model is scored
on is no longer the mask it was trained on, and nothing raises. Every number needed to re-derive a Dice
score is defined once, here.
"""

from __future__ import annotations

import os
import random
from typing import Any, Sequence

import numpy as np

# --- the intensity contract, in Hounsfield units --------------------------------------------------------
# Spleen parenchyma is ~40-60 HU unenhanced and ~90-130 HU in the portal-venous phase that most of Task09
# is acquired in. [-150, 200] therefore keeps the organ and drops lung, fat and bone detail; clipping stops
# one unusually dense scan from shifting the whole histogram.
HU_MIN = -150.0
HU_MAX = 200.0
SPACING_MM = 1.5  # isotropic resample target; Task09 originals are ~1.1-1.6 mm in-plane, 3-5 mm slices
ROI: tuple[int, int, int] = (192, 192, 192)
NUM_CLASSES = 2  # background + spleen

IMAGE_DIR = "imagesTr"
LABEL_DIR = "labelsTr"
# The MSD tarballs were packed on macOS, so every real volume is accompanied by an AppleDouble resource
# fork: imagesTr holds 82 entries for 41 spleens. They are not data and nibabel cannot read them.
JUNK_PREFIXES = ("._",)


def _is_case_file(name: str) -> bool:
    return name.endswith(".nii.gz") and not name.startswith(JUNK_PREFIXES)


def list_cases(msd_dir: str) -> list[dict[str, str]]:
    """Pair `imagesTr/spleen_N.nii.gz` with `labelsTr/spleen_N.nii.gz`, sorted by case *number*.

    Numerically, not lexicographically: `spleen_10` sorts before `spleen_2`, so a lexicographic list makes a
    split described as "the first 28 ids" impossible for a reviewer to reproduce by eye.
    """
    img_dir = os.path.join(msd_dir, IMAGE_DIR)
    lab_dir = os.path.join(msd_dir, LABEL_DIR)
    if not os.path.isdir(img_dir) or not os.path.isdir(lab_dir):
        raise FileNotFoundError(
            f"{msd_dir} needs {IMAGE_DIR}/ and {LABEL_DIR}/ - download Task09_Spleen first "
            "(tools/get_msd_spleen.py does it, or attach it as a Kaggle input)")
    cases = []
    for name in sorted(os.listdir(img_dir)):
        if not _is_case_file(name):
            continue
        label = os.path.join(lab_dir, name)
        if not os.path.exists(label):
            print(f"skipping {name}: no matching label in {LABEL_DIR}")
            continue
        cases.append({"case_id": name[: -len(".nii.gz")],
                      "image": os.path.join(img_dir, name),
                      "label": label})
    cases.sort(key=lambda c: _case_number(c["case_id"]))
    if not cases:
        raise FileNotFoundError(f"no .nii.gz cases found under {img_dir}")
    skipped = [n for n in os.listdir(img_dir) if n.endswith(".nii.gz") and not _is_case_file(n)]
    if skipped:
        print(f"ignored {len(skipped)} AppleDouble entries (._*) in {img_dir}")
    return cases


def _case_number(case_id: str) -> int:
    digits = "".join(ch for ch in case_id if ch.isdigit())
    return int(digits) if digits else 0


def split_cases(cases: Sequence[dict[str, str]], n_val: int = 7, n_test: int = 6,
                seed: int = 7) -> dict[str, list[dict[str, str]]]:
    """Deterministic train/val/test split of the released Task09 cases.

    This is *our* split, not MSD's: Task09 publishes labels for all 41 training volumes and keeps the
    challenge test images offline entirely, so there is no public test set to inherit - the notebook has to
    say so rather than imply a benchmark split. One case is one patient, so splitting by case is also
    splitting by patient: the chest track cannot make that claim, because RSNA's de-identification gave
    every file its own PatientID and the multi-film grouping was destroyed.
    """
    total = len(cases)
    if n_val + n_test >= total:
        raise ValueError(f"n_val + n_test must be < number of cases ({n_val}+{n_test} vs {total})")
    order = list(range(total))
    random.Random(seed).shuffle(order)
    test_idx = set(order[:n_test])
    val_idx = set(order[n_test:n_test + n_val])
    out: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for i, case in enumerate(cases):
        out["test" if i in test_idx else "val" if i in val_idx else "train"].append(case)
    ids = {k: set(c["case_id"] for c in v) for k, v in out.items()}
    assert len(ids["train"] | ids["val"] | ids["test"]) == total, "split lost or duplicated a case"
    assert not (ids["train"] & ids["test"] or ids["train"] & ids["val"] or ids["val"] & ids["test"])
    print(f"split (seed={seed}): {len(out['train'])} train / {len(out['val'])} val / "
          f"{len(out['test'])} test of {total} cases")
    return out


def window_ct(arr: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """HU -> float32 in [0, 1], clipped. Pure numpy so a test can pin the exact mapping the net saw."""
    x = np.asarray(arr, dtype=np.float32)
    return np.clip((x - hu_min) / (hu_max - hu_min), 0.0, 1.0)


def load_case(image_path: str, label_path: str, spacing_mm: float = SPACING_MM,
              reader: str = "NibabelReader") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One case as (windowed image, 0/1 label, provenance), both arrays shaped (D, H, W).

    Channel order follows MONAI: a NIfTI loads as (H, W, D), `EnsureChannelFirst` makes it (1, H, W, D),
    `Orientation("RAS")` puts the axes in the scanner convention, and `Spacing` resamples from the affine -
    bilinear for the image, nearest for the label, because interpolating a mask invents voxels that belong to
    no organ. The affine and both shapes come back so a predicted mask can be written into the same space as
    the scan it came from and checked by eye in the viewer.
    """
    from monai.transforms import (EnsureChannelFirst, LoadImage, Orientation, ScaleIntensityRange,
                                  Spacing)

    load = LoadImage(image_only=True, dtype=np.float32, reader=reader)
    img = EnsureChannelFirst()(load(str(image_path)))
    lab = EnsureChannelFirst()(load(str(label_path)))
    if getattr(img, "affine", None) is None:
        raise RuntimeError("MONAI returned no affine for this volume; resampling would be guessing at spacing")

    shape_before = [int(v) for v in img.shape[1:]]
    spacing_before = [float(v) for v in np.asarray(img.pixdim[:3]).tolist()]
    affine = np.asarray(img.affine, dtype=np.float64).copy()
    label_values = sorted(float(v) for v in np.unique(np.asarray(lab)).tolist())

    img, lab = Orientation(axcodes="RAS")(img), Orientation(axcodes="RAS")(lab)
    img = Spacing(pixdim=(spacing_mm,) * 3, mode="bilinear")(img)
    lab = Spacing(pixdim=(spacing_mm,) * 3, mode="nearest")(lab)
    img = ScaleIntensityRange(a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True)(img)

    x = np.asarray(img[0], dtype=np.float32)
    y = (np.round(np.asarray(lab[0], dtype=np.float32)) > 0.5).astype(np.uint8)
    if x.shape != y.shape:
        raise AssertionError(f"image {x.shape} and label {y.shape} diverged during resampling")
    meta = {
        "case_id": os.path.basename(str(image_path)).replace(".nii.gz", ""),
        "shape_before_resample": shape_before,
        "shape_after_resample": [int(v) for v in x.shape],
        "spacing_before_mm": spacing_before,
        "spacing_after_mm": [float(spacing_mm)] * 3,
        "affine": affine.tolist(),
        "label_values": label_values,
        "label_voxels": int(y.sum()),
        "image_min_max": [float(np.min(x)), float(np.max(x))],
        "foreground_fraction": float(y.mean()),
    }
    return x, y, meta


def to_tensors(x: np.ndarray, y: np.ndarray):
    """(D,H,W) numpy -> ((1,D,H,W), (1,D,H,W)) float32 / float32 tensors, channel-first for MONAI.

    The label becomes a float tensor because `DiceCELoss(to_onehot_y=True)` expects integer class ids in a
    single channel; keeping it float and rounding at the boundary is what lets the same tensor go to the
    loss and to `DiceMetric` without a cast in the notebook.
    """
    import torch

    return (torch.from_numpy(np.ascontiguousarray(x))[None],
            torch.from_numpy(np.ascontiguousarray(y).astype(np.float32))[None])


def sample_patch(x: np.ndarray, y: np.ndarray, roi: Sequence[int] = ROI, foreground_ratio: float = 0.5,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw one `roi`-sized crop out of a whole volume, foreground-biased.

    This is the part of a 3D segmentation pipeline that is easy to get silently wrong, so it is a named
    function with a returned record rather than three notebook lines:

    * A spleen is a few percent of an abdominal CT. Cropping uniformly at random gives mostly empty patches,
      and a Dice/CE loss on empty patches drives the model to predict nothing. `foreground_ratio` of the
      crops are centred on a *jittered* labelled voxel instead: jittered so the organ is not always in the
      middle of the window, and the remaining crops are drawn uniformly from the whole volume so the
      negatives are every organ and not just the one behind the volume centre.
    * A crop that falls off the edge is zero-padded, and the label is padded with 0, which is background.
      Without the `padded` flag in the record, an evaluation would score those synthetic background voxels
      as if they were real anatomy and inflate Dice.
    * The returned record says which happened, so the notebook can report how much of training actually saw
      the whole organ.

    `rng` is a `numpy.random.Generator` so a run is reproducible from one seed in the CONFIG.
    """
    if x.shape != y.shape:
        raise ValueError(f"image {x.shape} and label {y.shape} must match")
    rng = rng or np.random.default_rng()
    roi = tuple(int(v) for v in roi)
    if len(roi) != x.ndim:
        raise ValueError(f"roi {roi} has {len(roi)} dims, volume has {x.ndim}")

    centre: Sequence[int] | None = None
    fg_voxels = np.argwhere(y > 0.5)
    if len(fg_voxels) and rng.random() < foreground_ratio:
        # The jitter is load-bearing, not decoration. Centred exactly on an organ voxel, every positive crop
        # in the epoch has the spleen within a few voxels of the window centre, and a net is free to learn
        # "spleen = blob near the middle of whatever I am shown" - which scores 0.87 on a centred crop and
        # 0.33 on the whole volume, where most windows are a *shifted* organ it was never asked to locate.
        jitter = rng.integers(-(np.asarray(roi) // 4), np.asarray(roi) // 4 + 1, size=len(roi))
        centre = (np.asarray(rng.choice(fg_voxels)) + jitter).clip(0, np.asarray(x.shape) - 1).tolist()
        on_foreground = True
    else:
        # Uniform, not the volume centre: centring every negative on the middle of the abdomen handed the
        # model one background (liver/stomach) per case and left every kidney free to be called a spleen.
        centre = [int(rng.integers(0, s)) for s in x.shape]
        on_foreground = False

    starts, padded = [], False
    for c, size, extent in zip(centre, roi, x.shape):
        half = size // 2
        start = c - half
        if start < 0 or start + size > extent:
            padded = True
        starts.append(int(np.clip(start, 0, max(0, extent - size))))

    slices = tuple(slice(s, min(s + size, e)) for s, size, e in zip(starts, roi, x.shape))
    xp = x[slices]
    yp = y[slices]
    if xp.shape != roi:  # volume smaller than the crop on some axis: pad, background = 0
        pad_width = [(0, max(0, r - v)) for r, v in zip(roi, xp.shape)]
        xp = np.pad(xp, pad_width, mode="constant", constant_values=0.0)
        yp = np.pad(yp, pad_width, mode="constant", constant_values=0)

    record = {"centred_on_foreground": on_foreground, "zero_padded": padded,
              "foreground_fraction": float(yp.mean()), "start": list(starts),
              "patch_shape": [int(v) for v in xp.shape]}
    return xp, yp, record


def flip_patch(x: np.ndarray, y: np.ndarray, axes: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Random flips applied identically to image and label. Returns (x, y).

    Flipping only one of the pair is the classic 3D augmentation bug: the volume still looks fine, the mask
    still looks fine, and every Dice score computed from the pair is meaningless.
    """
    out_x, out_y = x, y
    for ax in axes:
        if rng.random() < 0.5:
            out_x = np.flip(out_x, axis=ax)
            out_y = np.flip(out_y, axis=ax)
    return out_x, out_y

"""CPU tests for track B's input contract. Every volume here is synthetic, written by the test itself, so
the data-loading logic is checked without the MSD download - and the resampling test uses an anisotropic
voxel size on purpose, because that is the case a spleen mask breaks on.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.data.ct_volume import (HU_MAX, HU_MIN, IMAGE_DIR, LABEL_DIR, SPACING_MM, flip_patch, list_cases,
                                load_case, sample_patch, split_cases, to_tensors, window_ct, _case_number)


def write_fake_case(root: str, case_id: str, shape=(12, 10, 8), spacing=(3.0, 0.8, 0.8)) -> dict:
    """A tiny NIfTI pair: a bright ellipsoid and the matching mask, in an anisotropic voxel grid."""
    import nibabel as nib

    os.makedirs(os.path.join(root, IMAGE_DIR), exist_ok=True)
    os.makedirs(os.path.join(root, LABEL_DIR), exist_ok=True)
    z, y, x = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    cz, cy, cx = [s / 2 for s in shape]
    inside = (((z - cz) / (shape[0] / 3)) ** 2 + ((y - cy) / (shape[1] / 3)) ** 2
              + ((x - cx) / (shape[2] / 3)) ** 2) < 1.0
    img = np.where(inside, 90.0, -300.0).astype(np.float32)   # ~spleen HU vs air
    lab = inside.astype(np.uint8)
    affine = np.diag(list(spacing) + [1.0])
    ip, lp = os.path.join(root, IMAGE_DIR, f"{case_id}.nii.gz"), os.path.join(root, LABEL_DIR, f"{case_id}.nii.gz")
    nib.save(nib.Nifti1Image(img, affine), ip)
    nib.save(nib.Nifti1Image(lab, affine), lp)
    return {"case_id": case_id, "image": ip, "label": lp}


# --- the intensity contract -----------------------------------------------------------------------------

def test_window_ct_maps_hu_endpoints_exactly():
    x = np.array([HU_MIN - 500, HU_MIN, (HU_MIN + HU_MAX) / 2, HU_MAX, HU_MAX + 900], dtype=np.float32)
    w = window_ct(x)
    assert w.dtype == np.float32
    assert w.tolist() == [0.0, 0.0, pytest.approx(0.5), 1.0, 1.0]


def test_window_ct_is_the_same_window_load_case_uses(tmp_path):
    # isotropic spacing so Spacing() is a no-op: this test is about the HU->[0,1] mapping, and bilinear
    # resampling at the phantom's edges would introduce intermediate levels that belong to the next test.
    case = write_fake_case(str(tmp_path), "spleen_1", spacing=(1.5, 1.5, 1.5))
    x, y, meta = load_case(case["image"], case["label"])
    assert meta["image_min_max"][0] >= 0.0 and meta["image_min_max"][1] <= 1.0
    # the phantom is either -300 HU (clips to 0) or 90 HU (inside the window). Exact values, because a
    # window that is off by a few HU still looks like a plausible picture and a plausible Dice score.
    uniq = sorted(np.unique(np.round(x, 4)).tolist())
    assert uniq[0] == 0.0
    assert uniq[-1] == pytest.approx((90.0 - HU_MIN) / (HU_MAX - HU_MIN), abs=1e-3)
    assert len(uniq) == 2, f"the phantom should window to exactly two levels, got {uniq}"


# --- finding the files ----------------------------------------------------------------------------------

def test_list_cases_sorts_numerically_not_lexicographically(tmp_path):
    for n in (1, 2, 10, 11):
        write_fake_case(str(tmp_path), f"spleen_{n}")
    cases = list_cases(str(tmp_path))
    assert [c["case_id"] for c in cases] == ["spleen_1", "spleen_2", "spleen_10", "spleen_11"]
    assert _case_number("spleen_10") < _case_number("spleen_11")


def test_list_cases_skips_unpaired_and_raises_without_data(tmp_path):
    write_fake_case(str(tmp_path), "spleen_1")
    # an image with no label must not become a training sample
    with open(os.path.join(tmp_path, IMAGE_DIR, "spleen_99.nii.gz"), "wb") as fh:
        fh.write(b"not a nifti")
    cases = list_cases(str(tmp_path))
    assert [c["case_id"] for c in cases] == ["spleen_1"]
    with pytest.raises(FileNotFoundError):
        list_cases(str(tmp_path / "nothing_here"))


# --- the split ------------------------------------------------------------------------------------------

def test_split_is_deterministic_disjoint_and_exhaustive():
    cases = [{"case_id": f"spleen_{i}"} for i in range(1, 42)]
    a = split_cases(cases, n_val=7, n_test=6, seed=7)
    b = split_cases(cases, n_val=7, n_test=6, seed=7)
    assert [c["case_id"] for c in a["train"]] == [c["case_id"] for c in b["train"]]
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (28, 7, 6)
    ids = {k: {c["case_id"] for c in v} for k, v in a.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert ids["train"] | ids["val"] | ids["test"] == {c["case_id"] for c in cases}


def test_split_depends_on_the_seed():
    cases = [{"case_id": f"spleen_{i}"} for i in range(1, 42)]
    assert {c["case_id"] for c in split_cases(cases)["test"]} != {
        c["case_id"] for c in split_cases(cases, seed=11)["test"]}


def test_split_refuses_to_eat_the_dataset():
    cases = [{"case_id": f"spleen_{i}"} for i in range(1, 11)]
    with pytest.raises(ValueError):
        split_cases(cases, n_val=5, n_test=5)


# --- one case through the real MONAI pipeline -----------------------------------------------------------

@pytest.mark.parametrize("spacing", [(3.0, 0.8, 0.8), (1.5, 1.5, 1.5)])
def test_load_case_resamples_to_isotropic_and_keeps_image_and_label_aligned(tmp_path, spacing):
    case = write_fake_case(str(tmp_path), "spleen_7", shape=(12, 10, 8), spacing=spacing)
    x, y, meta = load_case(case["image"], case["label"])
    assert x.shape == y.shape
    assert x.dtype == np.float32 and y.dtype == np.uint8
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert meta["spacing_after_mm"] == [SPACING_MM] * 3
    assert meta["shape_before_resample"] == [12, 10, 8]
    # each axis moves by original/1.5, rounded by MONAI; a 3 mm axis halves, a 0.8 mm axis nearly doubles
    assert meta["shape_after_resample"][0] == pytest.approx(12 * spacing[0] / SPACING_MM, rel=0.2)
    assert meta["shape_after_resample"][2] == pytest.approx(8 * spacing[2] / SPACING_MM, rel=0.2)
    assert meta["label_values"] == [0.0, 1.0]
    assert 0.02 < meta["foreground_fraction"] < 0.5, meta
    assert len(meta["affine"]) == 4


def test_load_case_mask_sits_where_the_bright_blob_is(tmp_path):
    """Alignment, not just shapes: the label must cover the high-intensity voxels it was drawn over."""
    case = write_fake_case(str(tmp_path), "spleen_3")
    x, y, _ = load_case(case["image"], case["label"])
    bright = x > 0.5
    assert y[bright].mean() > 0.9, "the mask no longer follows the phantom it was rasterised from"
    assert y[~bright].mean() < 0.1


def test_to_tensors_shapes_and_dtypes():
    x, y = np.random.default_rng(0).random((6, 6, 6)).astype(np.float32), np.zeros((6, 6, 6), np.uint8)
    xt, yt = to_tensors(x, y)
    assert tuple(xt.shape) == (1, 6, 6, 6) and tuple(yt.shape) == (1, 6, 6, 6)
    assert xt.dtype is torch.float32 and yt.dtype is torch.float32


def test_load_case_uses_the_labels_that_exist(tmp_path):
    # Task09 labels are 0/1; a stray 2 would still binarise to foreground, which must be explicit, not silent
    case = write_fake_case(str(tmp_path), "spleen_5")
    import nibabel as nib

    p = os.path.join(str(tmp_path), LABEL_DIR, "spleen_5.nii.gz")
    arr = (np.asarray(nib.load(p).dataobj) * 2).astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, nib.load(p).affine), p)
    _, y, meta = load_case(case["image"], p)
    assert meta["label_values"] == [0.0, 2.0]
    assert y.sum() > 0, "label value 2 must count as spleen, not vanish"


# --- crop drawing: the part of a 3D pipeline that goes silently wrong -----------------------------------

def _phantom(shape=(40, 44, 48)):
    """A volume with one bright blob off-centre, and the matching mask."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]]
    inside = ((z - 30) / 6) ** 2 + ((y - 8) / 6) ** 2 + ((x - 40) / 6) ** 2 < 1.0
    return np.where(inside, 0.9, 0.1).astype(np.float32), inside.astype(np.uint8)


def test_patch_is_always_roi_shaped():
    x, y = _phantom()
    rng = np.random.default_rng(0)
    for i in range(12):
        xp, yp, rec = sample_patch(x, y, roi=(16, 16, 16), foreground_ratio=0.5, rng=rng)
        assert xp.shape == yp.shape == (16, 16, 16), rec
        assert rec["patch_shape"] == [16, 16, 16]


def test_foreground_ratio_controls_empty_patches():
    x, y = _phantom()
    rng = np.random.default_rng(1)
    always = [sample_patch(x, y, (16, 16, 16), foreground_ratio=1.0, rng=rng)[2] for _ in range(8)]
    assert all(r["centred_on_foreground"] for r in always)
    assert all(r["foreground_fraction"] > 0.01 for r in always), "a foreground-centred crop must contain spleen"
    never = [sample_patch(x, y, (16, 16, 16), foreground_ratio=0.0, rng=rng)[2] for _ in range(4)]
    assert all(not r["centred_on_foreground"] for r in never)


def test_crop_off_the_edge_is_flagged_not_sneaky():
    x, y = _phantom(shape=(40, 44, 48))
    # the blob sits at z=30, y=8, x=40; a 32^3 crop centred there runs past the end of two axes
    rng = np.random.default_rng(2)
    padded = [sample_patch(x, y, (32, 32, 32), foreground_ratio=1.0, rng=rng)[2] for _ in range(20)]
    assert any(r["zero_padded"] for r in padded), "a crop this big against the edge must report padding"
    for r in padded:
        assert r["patch_shape"] == [32, 32, 32], "padding must still fill the requested ROI"


def test_patch_rejects_mismatched_inputs():
    x, y = _phantom()
    with pytest.raises(ValueError):
        sample_patch(x, y[:-2], (16, 16, 16))
    with pytest.raises(ValueError):
        sample_patch(x, y, (16, 16))


def test_sample_patch_is_seeded_and_reproducible():
    x, y = _phantom()
    a = sample_patch(x, y, (16, 16, 16), 0.7, np.random.default_rng(9))[0]
    b = sample_patch(x, y, (16, 16, 16), 0.7, np.random.default_rng(9))[0]
    assert np.array_equal(a, b)


def test_flip_moves_image_and_label_together():
    x, y = _phantom()
    # force every axis to flip: with a real RNG a seed can legitimately draw "no flip", and then this test
    # would fail while the function is correct. Alignment is the property under test, not RNG luck.
    always_flip = type("_AlwaysFlip", (), {"random": lambda self: 0.0})()
    xf, yf = flip_patch(x, y, (0, 1, 2), always_flip)
    assert xf.shape == x.shape
    # a flipped pair must score itself exactly like the unflipped pair
    from src.models.unet3d import binary_dice

    assert binary_dice(xf > 0.5, yf) == pytest.approx(binary_dice(x > 0.5, y))
    # the blob is off-centre, so a flip on every axis must move both arrays
    assert not np.array_equal(yf, y) and not np.array_equal(xf, x)


def test_flip_is_deterministic_from_the_generator():
    x, y = _phantom()
    a = flip_patch(x, y, (0, 1, 2), np.random.default_rng(11))
    b = flip_patch(x, y, (0, 1, 2), np.random.default_rng(11))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

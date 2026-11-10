import numpy as np
import pandas as pd
import pytest

from src.data.splits import assert_no_patient_leakage, patient_level_split


def make_frame(n_patients: int = 60, films: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n_patients):
        patient_label = int(rng.random() < 0.3)
        for film in range(films):
            rows.append(
                {
                    "patient_id": f"P{patient:04d}",
                    "dicom": f"P{patient:04d}_{film}.dcm",
                    "label": patient_label,
                }
            )
    return pd.DataFrame(rows)


def test_no_patient_appears_in_two_folds():
    df = make_frame()
    train, val, test = patient_level_split(df, "patient_id", "label")
    assert_no_patient_leakage(train, val, test, patient_col="patient_id")


def test_every_row_is_kept_exactly_once():
    df = make_frame()
    train, val, test = patient_level_split(df, "patient_id", "label")
    ids = pd.concat([train, val, test])["dicom"]
    assert len(ids) == len(df)
    assert ids.nunique() == len(df)


def test_all_patients_of_one_label_land_in_the_same_fold():
    df = make_frame()
    train, val, test = patient_level_split(df, "patient_id", "label")
    for frame in (train, val, test):
        counts = frame.groupby("patient_id")["label"].nunique()
        assert (counts == 1).all()


def test_split_is_deterministic_for_a_given_seed():
    df = make_frame()
    a = patient_level_split(df, "patient_id", "label", seed=11)
    b = patient_level_split(df, "patient_id", "label", seed=11)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right)


def test_different_seeds_produce_different_assignments():
    df = make_frame()
    a, _, _ = patient_level_split(df, "patient_id", "label", seed=1)
    b, _, _ = patient_level_split(df, "patient_id", "label", seed=2)
    assert set(a["dicom"]) != set(b["dicom"])


def test_fractions_are_roughly_respected():
    df = make_frame(n_patients=300)
    train, val, test = patient_level_split(df, "patient_id", "label")
    n_patients = df["patient_id"].nunique()
    got = [
        frame["patient_id"].nunique() / n_patients for frame in (train, val, test)
    ]
    for expected, actual in zip((0.7, 0.15, 0.15), got):
        assert abs(expected - actual) < 0.06


def test_leakage_guard_actually_catches_a_bad_split():
    df = make_frame()
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(df, df, patient_col="patient_id")


def test_rejects_fractions_that_do_not_sum_to_one():
    df = make_frame()
    with pytest.raises(ValueError):
        patient_level_split(df, "patient_id", "label", fractions=(0.5, 0.2, 0.2))


def test_rejects_missing_patient_column():
    df = make_frame()
    with pytest.raises(KeyError):
        patient_level_split(df, "subject", "label")


def test_image_level_split_would_have_leaked_this_dataset():
    df = make_frame()
    rng = np.random.default_rng(0)
    mask = rng.random(len(df)) < 0.7
    train, test = df[mask], df[~mask]
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(train, test, patient_col="patient_id")

"""Patient-level train/val/test splitting that cannot leak a patient across splits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Sequence

import pandas as pd


def _dominant_label(labels: Iterable[object]) -> object:
    values = [label for label in labels if pd.notna(label)]
    if not values:
        return "unknown"
    counts: dict[object, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts.items(), key=lambda item: (item[1], str(item[0])))[0]


def _stable_rank(token: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{token}".encode()).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def patient_level_split(
    df: pd.DataFrame,
    patient_col: str,
    label_col: str | None = None,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) frames with every patient_id present in exactly one frame."""
    if len(fractions) != 3 or not 0.999 < sum(fractions) < 1.001:
        raise ValueError(f"fractions must be 3 values summing to 1, got {fractions}")
    if patient_col not in df.columns:
        raise KeyError(f"{patient_col!r} not in columns {list(df.columns)}")

    patients = pd.Index(df[patient_col].unique())
    if len(patients) < 3:
        raise ValueError(f"need at least 3 distinct patients, found {len(patients)}")

    if label_col is None:
        buckets = {None: patients}
    else:
        per_patient = df.groupby(patient_col)[label_col].apply(list)
        buckets = defaultdict(list)
        for patient, labels in per_patient.items():
            buckets[_dominant_label(labels)].append(patient)
        buckets = {key: pd.Index(value) for key, value in buckets.items()}

    assigned: dict[object, int] = {}

    for bucket_patients in buckets.values():
        ordered = sorted(bucket_patients, key=lambda p: _stable_rank(str(p), seed))
        n = len(ordered)
        n_train = max(1, int(round(n * fractions[0])))
        n_val = max(1, int(round(n * fractions[1]))) if n - n_train >= 2 else 0
        cuts = {
            **{p: 0 for p in ordered[:n_train]},
            **{p: 1 for p in ordered[n_train : n_train + n_val]},
            **{p: 2 for p in ordered[n_train + n_val :]},
        }
        assigned.update(cuts)

    fold_of_patient = df[patient_col].map(assigned)
    train = df[fold_of_patient == 0].copy()
    val = df[fold_of_patient == 1].copy()
    test = df[fold_of_patient == 2].copy()
    return train, val, test


def assert_no_patient_leakage(
    *frames: pd.DataFrame, patient_col: str
) -> None:
    """Raise if any patient appears in more than one of the given frames."""
    seen: dict[object, int] = {}
    for index, frame in enumerate(frames):
        for patient in frame[patient_col].unique():
            if patient in seen:
                raise AssertionError(
                    f"patient {patient!r} appears in both fold {seen[patient]} and fold {index}"
                )
            seen[patient] = index

"""results/metrics.json is hand-assembled, so this file is the guard that keeps it honest.

Every assertion compares the summary against the per-track artefact the number actually came from. If a notebook is
re-run and one of those files changes, this fails rather than leaving a quotable number in two places.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"
SUMMARY = RESULTS / "metrics.json"

pytestmark = pytest.mark.skipif(
    not SUMMARY.exists(), reason="results/ is not shipped in the API image, only tick metrics there"
)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def summary() -> dict:
    return load("metrics.json")


def test_counts_and_rates_match_the_track_a_artefact(summary):
    den = load("densenet_metrics.json")
    test = summary["track_a_classification"]["test"]
    source = den["evaluations"]["L3 DenseNet121 - TEST"]

    assert test["confusion"] == source["confusion"]
    for field in ("n", "auroc", "pr_auc", "threshold", "sensitivity", "specificity", "precision", "f1"):
        assert test[field] == source[field], field
    assert summary["track_a_classification"]["val"]["auroc"] == den["evaluations"]["L3 DenseNet121 - VAL"]["auroc"]
    assert summary["track_a_classification"]["threshold"] == den["threshold_by_youden_on_val"]


def test_confusion_counts_account_for_every_test_image(summary):
    test = summary["track_a_classification"]["test"]
    assert sum(test["confusion"].values()) == test["n"] == 4003


def test_baselines_match_the_baseline_artefact(summary):
    base = load("baseline_metrics.json")["baselines"]
    assert summary["baselines"]["L1_pixel_logreg_test"]["auroc"] == base["L1_pixel_test"]["auroc"]
    assert summary["baselines"]["L2_frozen_resnet18_logreg_test"]["auroc"] == base["L2_frozen_test"]["auroc"]
    assert summary["baselines"]["L0_constant"]["auroc"] == base["L0_constant"]["auroc"]
    assert summary["track_a_classification"]["bar_to_beat_test_auroc"] == base["L1_pixel_test"]["auroc"]


def test_calibration_matches_the_calibration_artefact(summary):
    cal = load("calibration.json")
    for field in ("n", "auroc", "brier", "log_loss", "ece_10bin", "mce_10bin"):
        assert summary["calibration"][field] == cal[field], field
    assert summary["calibration"]["ece_95ci_bootstrap"] == cal["ece_95ci_bootstrap"]


def test_fairness_matches_the_gradcam_artefact(summary):
    fair = load("gradcam_fairness_summary.json")
    assert summary["fairness"]["test_subset"] == fair["test_subset"]
    assert summary["fairness"]["sex"]["table"] == fair["fairness"]["sex"]["table"]
    assert summary["fairness"]["sex"]["max_gap"] == fair["fairness"]["sex"]["max_gap"]
    assert summary["fairness"]["age_band_max_gap"] == fair["fairness"]["age_band"]["max_gap"]
    assert summary["fairness"]["gradcam"]["maps"] == fair["gradcam"]["maps"]


def test_spleen_dice_matches_the_segmentation_artefact(summary):
    spl = load("spleen_metrics.json")
    assert summary["track_b_segmentation"]["test"] == spl["test_report"]
    assert summary["track_b_segmentation"]["test"]["n"] == 6


def test_every_listed_source_exists(summary):
    for relative in summary["sources"]:
        assert (RESULTS.parent / relative).exists(), relative


def test_disclaimer_survives_the_copy(summary):
    assert summary["disclaimer"] == load("densenet_metrics.json")["disclaimer"]
    assert "education" in summary["disclaimer"].lower()

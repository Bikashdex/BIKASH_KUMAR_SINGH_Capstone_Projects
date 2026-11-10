"""Post-hoc reliability of the saved DenseNet test probabilities.

Reads the predictions CSV written by notebook 04 (prob + label columns) and writes
results/calibration.json plus results/plots/calibration.png. No model or GPU needed:
the scores were already produced by the Kaggle run, this only measures them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "lo": round(float(edges[b]), 2),
                "hi": round(float(edges[b + 1]), 2),
                "n": int(m.sum()),
                "mean_prob": round(float(p[m].mean()), 4),
                "observed": round(float(y[m].mean()), 4),
                "gap": round(float(abs(p[m].mean() - y[m].mean())), 4),
            }
        )
    ece = sum(r["n"] / len(p) * r["gap"] for r in rows)
    mce = max(r["gap"] for r in rows)
    return rows, float(ece), float(mce)


def bootstrap_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, n_boot: int = 500, seed: int = 7):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        vals.append(reliability(y[idx], p[idx], n_bins)[1])
    return [round(float(v), 4) for v in np.quantile(vals, [0.025, 0.975])]


def plot(rows, y, p, ece, brier, out_path: Path):
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect calibration")
    ax.plot(
        [r["mean_prob"] for r in rows],
        [r["observed"] for r in rows],
        "o-",
        color="#d62728",
        label=f"DenseNet121 (ECE {ece:.3f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title("Reliability, held-out test split")
    ax.legend(loc="upper left")

    axh.hist(p, bins=np.linspace(0, 1, 21), color="#4c72b0")
    axh.set_yscale("log")
    axh.set_xlabel("predicted probability")
    axh.set_ylabel("images (log scale)")
    axh.set_title(f"Score distribution (n={len(p):,})")

    fig.suptitle(
        f"Raw sigmoid scores are over-confident: mean score {p.mean():.3f} vs prevalence "
        f"{y.mean():.3f}, Brier {brier:.3f} - education only",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path,
                    default=Path("results/test_predictions_with_metadata.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--threshold", type=float, default=0.5108,
                    help="decision threshold used by notebook 04 (validation Youden's J)")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    missing = {"prob", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.predictions} is missing columns: {sorted(missing)}")
    y = df["label"].to_numpy(dtype=float)
    p = df["prob"].to_numpy(dtype=float)

    rows, ece, mce = reliability(y, p)
    payload = {
        "source": str(args.predictions),
        "n": int(len(df)),
        "prevalence": round(float(y.mean()), 4),
        "mean_probability": round(float(p.mean()), 4),
        "auroc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "ece_10bin": round(ece, 4),
        "ece_95ci_bootstrap": bootstrap_ece(y, p),
        "mce_10bin": round(mce, 4),
        "threshold": args.threshold,
        "bins": rows,
        "reading": (
            "Raw sigmoid scores are systematically higher than the observed positive rate "
            "(mean score 0.351 vs 22.5% prevalence; ECE 0.126). The training loss used "
            "pos_weight=3.4387, so the score ranks well (AUROC 0.8819) but must not be read "
            "as a probability. The 0.5108 threshold was chosen on validation for this reason; "
            "a Platt/isotonic calibration layer is future work, not a clinical claim."
        ),
        "disclaimer": "Education only. Not for clinical use.",
    }

    out_json = args.out_dir / "calibration.json"
    out_png = args.out_dir / "plots" / "calibration.png"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    plot(rows, y, p, ece, payload["brier"], out_png)

    print(json.dumps({k: payload[k] for k in
                      ("n", "prevalence", "mean_probability", "auroc", "brier",
                       "ece_10bin", "ece_95ci_bootstrap", "mce_10bin")}, indent=1))
    print("wrote", out_json)
    print("wrote", out_png)


if __name__ == "__main__":
    main()

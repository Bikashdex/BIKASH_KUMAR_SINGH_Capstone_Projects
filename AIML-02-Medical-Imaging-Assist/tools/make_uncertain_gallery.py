"""Render the uncertain-band cases the Grad-CAM gallery does not picture.

Notebook 04's gallery covers confident hits, misses and false positives. This tool renders two
test cases whose score sits inside threshold +/- 0.05, through the same cache render (448 px),
transform and checkpoint, so the model card can show the band rather than only describe it.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dicom_io import to_uint8
from src.data.transforms import gray_to_rgb, make_transform
from src.models.classifier import build_densenet121, load_checkpoint
from src.models.gradcam import gradcam, overlay

CACHE_SIZE = 448
MODEL_INPUT = 224


def render(dicom_path: str) -> Image.Image:
    """The same 448 px JPEG cache step notebooks 03/04 used, minus the JPEG round-trip."""
    img = Image.fromarray(to_uint8(pydicom.dcmread(dicom_path, force=True)))
    return img.resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR).convert("L")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=os.path.join(ROOT, "results", "test_predictions_with_metadata.csv"))
    ap.add_argument("--dicom-dir", default=os.path.join(ROOT, "data", "raw", "rsna"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "artifacts", "densenet121_lungopacity.pt"))
    ap.add_argument("--ids", nargs="+", default=["b8fd583c-2159-494a-bed3-88a3c0370efa",
                                                 "0d4192cd-b111-45df-86f6-712b04dad4a9"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "plots", "gradcam_uncertain.png"))
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    net = build_densenet121(num_classes=1, pretrained=False, repo_dir=ROOT)
    load_checkpoint(net, args.checkpoint, map_location="cpu")
    net.eval()

    fig, axes = plt.subplots(len(args.ids), 2, figsize=(9, 4.0 * len(args.ids)), squeeze=False)
    for row, file_id in enumerate(args.ids):
        rec = pred[pred.file_id == file_id].iloc[0]
        grey = np.asarray(render(os.path.join(args.dicom_dir, rec.dicom_rel)))
        x = make_transform(MODEL_INPUT, train=False)(gray_to_rgb(grey))[None]
        heat = gradcam(net, x, image_size=(CACHE_SIZE, CACHE_SIZE))
        axes[row, 0].imshow(grey, cmap="gray")
        axes[row, 0].set_title(f"label={int(rec.label)} p={rec.prob:.3f} | {rec.class_source}", fontsize=8)
        axes[row, 1].imshow(overlay(grey, heat, alpha=0.5, threshold=0.3))
        axes[row, 1].set_title("Grad-CAM overlay", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
        print(f"{file_id}: label={int(rec.label)} p={rec.prob:.4f} source={rec.class_source}")

    fig.suptitle("Uncertain band (|p - 0.511| <= 0.05) - education only, not a diagnosis", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("wrote", args.out, f"({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()

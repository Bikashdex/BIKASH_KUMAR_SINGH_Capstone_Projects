"""Dataset over the decoded JPEG cache - the layer that decides what a missing render means."""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import gray_to_rgb


class ChestXrayDataset(Dataset):
    """One split of the split CSVs, read out of the 448 px JPEG cache notebook 03 builds.

    Paths are derived from the CSV's `dicom_rel` column by swapping the extension, so a split CSV is
    reusable across machines without rewriting it.
    """

    def __init__(self, frame, root: str, tf, cache_size: int, blank_missing: bool = True):
        self.root = root
        self.cache_size = int(cache_size)
        self.blank_missing = blank_missing
        self.paths = [os.path.join(root, os.path.basename(r).replace(".dcm", ".jpg"))
                      for r in frame["dicom_rel"]]
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def missing_paths(self) -> list[str]:
        """Which renders never made it to disk - notebook 04 audits this instead of hiding it."""
        return [p for p in self.paths if not os.path.exists(p)]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, float]:
        path = self.paths[i]
        if not os.path.exists(path):
            if not self.blank_missing:
                raise FileNotFoundError(path)
            # A single failed decode is treated as blank rather than crashing a 10-minute epoch; the
            # count is reported by missing_paths() and written into densenet_metrics.json.
            img = Image.new("L", (self.cache_size, self.cache_size), 0)
        else:
            img = Image.open(path)
            img.load()
        return self.tf(gray_to_rgb(np.asarray(img, dtype=np.uint8))), float(self.labels[i])

"""Grad-CAM attention maps for the lung-opacity classifier, plus the numbers that make them checkable.

Why this exists rather than `pytorch-grad-cam`: the graded claim is "the model looks at lung parenchyma,
and it looks at the same place regardless of the patient's sex or age". That needs the raw map, so the
overlay can be judged *and* measured (``region_stats``). Also, a frozen backbone produces a graph with no
gradient-carrying leaves, and off-the-shelf helpers fail there - this module enables grad on the
parameters for the duration of the backward pass and restores them afterwards.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: the last concatenation inside DenseNet121's feature stack: 1024 channels, 7x7 for a 224 px input
TARGET_LAYER = "features.denseblock4"


def resolve_module(model: nn.Module, dotted_name: str) -> nn.Module:
    """Look up "features.denseblock4" by attribute path.

    torchvision's DenseNet names its blocks `denselayer1` as modules but `denselayer1_` in state_dict
    keys, so accept either spelling of a segment rather than trusting one.
    """
    found = dict(model.named_modules())
    if dotted_name in found:
        return found[dotted_name]
    current = model
    walked = []
    for segment in dotted_name.split("."):
        # accept torchvision's `denselayer16_` as well as `denselayer16`
        match = next((name for name in current._modules if name.rstrip("_") == segment.rstrip("_")), None)
        if match is None:
            raise AttributeError(
                f"no submodule {segment!r} under '{'.'.join(walked) or type(current).__name__}'. "
                f"Children: {sorted(current._modules)}"
            )
        walked.append(match)
        current = current._modules[match]
    return current


@contextlib.contextmanager
def _params_requiring_grad(model: nn.Module):
    was = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(True)
    try:
        yield
    finally:
        for p, was_flag in zip(model.parameters(), was):
            p.requires_grad_(was_flag)


def class_index_for(logit_row: torch.Tensor, requested: int | None) -> int:
    if requested is not None:
        if not 0 <= requested < logit_row.numel():
            raise IndexError(f"class_index {requested} outside a {logit_row.numel()}-logit head")
        return requested
    if logit_row.numel() != 1:
        raise ValueError(
            f"the head emits {logit_row.numel()} logits; pass class_index=<the opacity class> so the map "
            "is not silently tied to whichever class happens to score highest"
        )
    return 0


def gradcam(
    model: nn.Module,
    image: torch.Tensor,
    *,
    target_layer: str = TARGET_LAYER,
    class_index: int | None = None,
    image_size: Sequence[int] | None = None,
) -> np.ndarray:
    """Return a Grad-CAM map for one preprocessed image, as float32 in [0, 1].

    `image` is (1, 3, H, W) - the same tensor the DataLoader yields after the eval transform, so the map
    describes what the network actually saw. Pass `image_size=(h, w)` to get the map back at input
    resolution; without it you get the feature-map grid (7x7 for a 224 px image), which is too coarse to
    read as anatomy but is what `region_stats` was designed on.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"expected one image shaped (1, 3, H, W), got {tuple(image.shape)}")
    layer = resolve_module(model, target_layer)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError(f"{target_layer} outputs {type(output).__name__}, not a tensor")
        captured.append(output)

    handle = layer.register_forward_hook(hook)
    try:
        with _params_requiring_grad(model), torch.enable_grad():
            logits = model(image.detach())
            index = class_index_for(logits.detach()[0], class_index)
            score = logits.flatten()[index]        # (1, 1) or (1, C) heads: exactly one scalar here
            if not captured:
                raise RuntimeError(f"{target_layer} did not run - is it really in this model's forward path?")
            acts = captured[-1]
            grads = torch.autograd.grad(score, acts)[0]
    finally:
        handle.remove()

    # the original Grad-CAM formulation: every channel's gradient, averaged over its own spatial grid
    weights = grads.mean(dim=(2, 3), keepdim=True)
    map_16 = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)
    if image_size is not None and tuple(image_size) != tuple(map_16.shape[-2:]):
        map_16 = F.interpolate(map_16, size=tuple(image_size), mode="bilinear", align_corners=False)
    return normalize_heatmap(map_16[0, 0].detach().float().cpu().numpy())


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """ReLU already happened; this only rescales to [0, 1] and keeps an all-zero map all-zero."""
    hm = np.asarray(hm, dtype=np.float64)
    lo, hi = hm.min(), hm.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(hm.shape, dtype=np.float32)
    return ((hm - lo) / (hi - lo)).astype(np.float32)


def resize_heatmap(hm: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """Bilinear upsample to (h, w) for an overlay; nearest would draw staircases on a 7x7 map."""
    t = torch.from_numpy(np.asarray(hm, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=tuple(size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def jet(gray: np.ndarray) -> np.ndarray:
    """matplotlib's 'jet' anchors, so an overlay here matches the figure in the notebook."""
    xs = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    channels = [
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5]),   # red
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),   # green
        np.array([0.5, 1.0, 1.0, 0.0, 0.0, 0.0]),   # blue
    ]
    flat = np.clip(np.asarray(gray, dtype=np.float64).ravel(), 0.0, 1.0)
    rgb = np.stack([np.interp(flat, xs, c) for c in channels], axis=1)
    return rgb.reshape(np.asarray(gray).shape + (3,))


def overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.5,
    threshold: float = 0.3,
) -> np.ndarray:
    """Tint a windowed X-ray with attention and return uint8 RGB.

    Below `threshold` the pixels stay pure grey. Without that cut-off the whole field of view is
    washed in blue and the radiological content - the thing a reviewer is meant to check - disappears.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] not in (1, 3):
            raise ValueError(f"unexpected image channels {image.shape}")
        grey = image[..., 0]
    else:
        grey = image
    hm = resize_heatmap(np.asarray(heatmap, dtype=np.float32), grey.shape[-2:])
    out8 = image.dtype == np.uint8
    base = grey.astype(np.float64) / (255.0 if out8 else 1.0)
    base = np.repeat(base[..., None], 3, axis=2)
    tinted = (1.0 - alpha) * base + alpha * jet(hm)
    tinted = np.where(hm[..., None] >= threshold, tinted, base)
    if out8:
        return np.clip(tinted * 255.0, 0, 255).astype(np.uint8)
    return np.clip(tinted, 0.0, 1.0).astype(np.float32)


def otsu_threshold(gray: np.ndarray) -> float:
    """Foreground cut-off for a windowed 8-bit image: the air around a chest is near zero."""
    hist, _ = np.histogram(np.clip(np.asarray(gray), 0, 255).astype(np.uint8).ravel(), bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    levels = np.arange(256, dtype=np.float64)
    weight_below = np.cumsum(hist) / total
    mean_below = np.cumsum(hist * levels) / np.maximum(weight_below * total, 1e-12)
    mean_all = (hist * levels).sum() / total
    mean_above = (mean_all - weight_below * mean_below) / np.maximum(1.0 - weight_below, 1e-12)
    variance = weight_below * (1.0 - weight_below) * (mean_below - mean_above) ** 2
    return float(np.nanargmax(np.where(np.isfinite(variance), variance, 0.0)))


def body_mask(gray: np.ndarray) -> np.ndarray:
    """True where the exposure is above an Otsu cut-off - a stand-in for "inside the patient"."""
    return np.asarray(gray) > otsu_threshold(np.asarray(gray))


def region_stats(
    heatmap: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    top_fraction: float = 0.2,
) -> dict[str, object]:
    """Summarise where the attention actually went, so a heat-map gallery is not the only evidence.

    Keys: mass, centroid, vertical/horizontal spread, in_body_fraction, lateral_bias.
    `lateral_bias` is viewer's-left minus viewer's-right over the strongest `top_fraction` of the map,
    in [-1, 1]; remember the radiological convention, where the patient's right lung is on the
    viewer's left.
    """
    hm = np.asarray(heatmap, dtype=np.float64)
    h, w = hm.shape
    total = hm.sum()
    if total <= 0:
        # every key present, values None: a caller building a CSV from these dicts must not KeyError
        # because one image rendered blank
        return {"mass": 0.0, "centroid": None, "spread": None, "peak": None,
                "in_body_fraction": None, "lateral_bias": None, "note": "empty map"}

    rows = np.arange(h, dtype=np.float64)[:, None] * np.ones((1, w))
    cols = np.arange(w, dtype=np.float64)[None, :] * np.ones((h, 1))
    centroid = (float((rows * hm).sum() / total), float((cols * hm).sum() / total))
    var = (float((((rows - centroid[0]) ** 2) * hm).sum() / total),
           float((((cols - centroid[1]) ** 2) * hm).sum() / total))

    stats: dict[str, object] = {
        "mass": float(total),
        # normalised so maps from different backbones or input sizes are comparable
        "centroid": (round(centroid[0] / (h - 1), 4) if h > 1 else 0.0,
                     round(centroid[1] / (w - 1), 4) if w > 1 else 0.0),
        "spread": (round(float(np.sqrt(var[0])) / h, 4), round(float(np.sqrt(var[1])) / w, 4)),
        "peak": tuple(int(v) for v in np.unravel_index(np.argmax(hm), hm.shape)),
    }

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != hm.shape:
            m = body_mask_resized(m, hm.shape)
        stats["in_body_fraction"] = round(float(hm[m].sum() / total), 4)

    cut = np.quantile(hm, 1.0 - top_fraction)
    top = np.where(hm >= cut, hm, 0.0)
    top_total = top.sum()
    if top_total > 0:
        left = float(top[:, : w // 2].sum())
        right = float(top[:, w - w // 2:].sum())
        stats["lateral_bias"] = round((left - right) / top_total, 4)
    else:
        stats["lateral_bias"] = None
    return stats


def body_mask_resized(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Downsample a pixel mask to the map grid with area averaging, never nearest - a thin border
    lost to nearest sampling would silently inflate in_body_fraction."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    t = F.interpolate(t, size=tuple(shape), mode="area")
    return (t[0, 0].numpy() > 0.5)

"""3D U-Net (MONAI `DynUNet`) for the spleen track: build, loss, Dice, sliding-window inference, checkpoints.

Why this is a module and not notebook cells: the training notebook, the FastAPI service and the tests must
construct the *same* network. A hand-copied `DynUNet(...)` in a fourth place is how a checkpoint silently
stops matching the architecture that loaded it - `strict=True` raises, and the tempting fix is
`strict=False`, which turns a real mismatch into random weights.

Two deliberate choices are encoded here rather than left to the reader:

* `out_channels=2`, not 1. A single sigmoid logit can be trained, but MONAI's `DiceCELoss` is written for
  the one-hot/softmax convention, and `DiceMetric` wants the same. Two channels keep loss and metric on the
  same path as every other MONAI example, so the numbers are comparable to published MSD results.
* Dice is reimplemented here in numpy (`binary_dice`) even though MONAI ships one. The tests assert the two
  agree, so a scoring bug shows up as a failing test instead of a slightly optimistic report.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.data.ct_volume import HU_MAX, HU_MIN, NUM_CLASSES, ROI, SPACING_MM

__all__ = ["HU_MIN", "HU_MAX", "SPACING_MM", "ROI", "NUM_CLASSES"]

# The HU window, spacing, ROI and class count are imported from src/data/ct_volume.py rather than repeated
# here: they are the input contract, and a model file that redefines them is a second source of truth.


def dynunet_strides(num_downsample: int) -> list[list[int]]:
    """Per-stage encoder strides for `DynUNet`: one full-resolution stage, then `num_downsample` halvings.

    MONAI wants one entry per stage and each entry is a 3-vector, so `[[1,1,1],[2,2,2],...]`. Flat lists
    raise `length of kernel_size in block 0 should be the same as spatial_dims`.
    """
    if not 1 <= num_downsample <= 6:
        raise ValueError(f"num_downsample must be 1..6, got {num_downsample}")
    return [[1, 1, 1]] + [[2, 2, 2]] * num_downsample


def dynunet_filters(num_downsample: int, initial_filters: int = 32,
                    max_filters: int = 320) -> list[int]:
    """Per-stage channel widths, doubling until `max_filters` - the shape MSD's own baselines use.

    MONAI 1.5.2 has no `initial_filters` argument on `DynUNet`, so the widths are computed here and passed
    explicitly; leaving `filters=None` makes the architecture depend on whatever the installed version
    defaults to, which is exactly what a checkpoint has to be immune to.
    """
    widths = [initial_filters]
    for _ in range(num_downsample):
        widths.append(min(widths[-1] * 2, max_filters))
    return widths


def build_dynunet(num_downsample: int = 5, initial_filters: int = 32, num_aux_heads: int = 2,
                  in_channels: int = 1, num_classes: int = NUM_CLASSES,
                  deep_supervision: bool = True, spatial_dims: int = 3,
                  probe_input: tuple[int, ...] | None = None) -> nn.Module:
    """MONAI `DynUNet`, the 3D U-Net variant the Medical Segmentation Decathlon baselines used.

    `deep_supervision=True` makes a training forward pass return the main head plus `num_aux_heads`
    auxiliary heads (MONAI's `deep_supr_num` counts the *extra* ones, so 2 gives 3 outputs), each upsampled
    back to full resolution and stacked on a new axis. `eval()` returns one tensor only. That asymmetry is a
    real trap - the loss must zip over the stacked axis and inference must not - so
    `unpack_deep_supervision` below is the only place that knows it.
    """
    from monai.networks.nets import DynUNet

    if num_aux_heads >= num_downsample:
        # MONAI enforces this and says so badly ("deep_supr_num should be less than the number of up sample
        # layers"), so the constraint is restated here against the names the notebook's CONFIG actually uses.
        raise ValueError(f"num_aux_heads must be < num_downsample: {num_aux_heads} vs {num_downsample} "
                         "(a miniature CPU net can only supervise a few heads)")
    if spatial_dims != 3 and probe_input is None:
        raise ValueError("only 2D/3D helpers are wired; pass probe_input to build a non-3D net")
    filters = dynunet_filters(num_downsample, initial_filters)
    net = DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=num_classes,
        kernel_size=[[3] * spatial_dims] * (num_downsample + 1),
        strides=dynunet_strides(num_downsample),
        upsample_kernel_size=[[2] * spatial_dims] * num_downsample,
        filters=filters,
        norm_name="instance",
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout=None,
        deep_supervision=deep_supervision,
        deep_supr_num=num_aux_heads,
        res_block=True,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DynUNet: stages={num_downsample + 1} filters={filters} in={in_channels} "
          f"classes={num_classes} deep_supervision={deep_supervision} aux_heads={num_aux_heads} "
          f"params={n_params:,}")

    if probe_input is not None:
        # A forward pass on a real-sized input is the only check that strides, filters and the upsample list
        # agree with each other; a wrong combination otherwise surfaces as a crash at training step 1.
        net.eval()
        with torch.no_grad():
            out = net(torch.zeros(1, in_channels, *probe_input))
        main = out[:, 0] if out.ndim == 6 else out
        assert tuple(main.shape) == (1, num_classes, *probe_input), \
            f"eval forward on {probe_input} gave {tuple(out.shape)}"
        net.train()
        stacked = net(torch.zeros(1, in_channels, *probe_input))
        assert stacked.ndim == 6 and stacked.shape[1] == num_aux_heads + 1, \
            f"training output {tuple(stacked.shape)} does not carry {num_aux_heads + 1} heads"
        print(f"probe ok: eval() -> 1 tensor {tuple(main.shape)}, train() -> {stacked.shape[1]} stacked heads")
    return net


def make_loss():
    """Dice + cross-entropy on the foreground channel only."""
    from monai.losses import DiceCELoss

    return DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)


def unpack_deep_supervision(output: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Split DynUNet's training output into the full-resolution head plus its auxiliary heads.

    MONAI 1.5 stacks them: a training forward pass on a batch of 2 with `num_aux_heads=2` returns one tensor
    shaped `(2, 3, 2, D, H, W)` = (batch, head, class, spatial). `eval()` returns `(1, 2, D, H, W)` - a
    single head. Handling only the list form would crash on the first training step, and handling only the
    stacked form would crash at inference, so both are accepted and split the same way.
    """
    if isinstance(output, (list, tuple)):
        levels = list(output)
    elif output.ndim == 6:
        levels = [output[:, i] for i in range(output.shape[1])]
    else:
        levels = [output]
    return levels[0], levels[1:]


def deep_supervision_loss(output: Any, target: torch.Tensor, loss_fn, weight: float = 0.4):
    """Main-head loss plus a decaying weight on each auxiliary head.

    Every DynUNet head is upsampled back to full resolution, so all of them compare against the same label
    tensor - no downsampled targets needed. `eval()` gives one head and this returns exactly the plain loss,
    which is why the validation loop can call it unchanged.
    """
    main, aux = unpack_deep_supervision(output)
    total = loss_fn(main, target)
    for i, extra in enumerate(aux):
        total = total + (weight ** (i + 1)) * loss_fn(extra, target)
    return total


def binary_dice(pred: np.ndarray, ref: np.ndarray, smooth: float = 1e-6) -> float:
    """Symmetric Dice on two boolean/0-1 masks of identical shape. No MONAI, no GPU, easy to test.

    A both-empty pair returns 1.0 by the epsilon convention. That is *not* a good score and callers
    reporting an average over cases must keep empty-organ cases visible - see `dice_report`.
    """
    p = np.asarray(pred).astype(bool)
    r = np.asarray(ref).astype(bool)
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs ref {r.shape}")
    intersection = np.logical_and(p, r).sum()
    denominator = p.sum() + r.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (denominator + smooth))


def volume_metrics(pred: np.ndarray, ref: np.ndarray, spacing_mm: float = SPACING_MM) -> dict[str, Any]:
    """Dice plus the two numbers a radiologist actually asks for when a mask looks wrong.

    Volumes in mm^3 (a voxel is spacing^3 for isotropic data) and the centroid distance in mm. A mask can
    have Dice 0.9 and still be shifted onto the wrong lobe, and Dice alone does not show that.
    """
    p = np.asarray(pred).astype(bool)
    r = np.asarray(ref).astype(bool)
    voxel = float(spacing_mm) ** 3
    out: dict[str, Any] = {
        "dice": binary_dice(p, r),
        "pred_voxels": int(p.sum()),
        "ref_voxels": int(r.sum()),
        "pred_volume_mm3": float(p.sum() * voxel),
        "ref_volume_mm3": float(r.sum() * voxel),
        "pred_empty": bool(p.sum() == 0),
        "ref_empty": bool(r.sum() == 0),
    }
    if out["pred_empty"] or out["ref_empty"]:
        out["centroid_dist_mm"] = None
    else:
        pc = np.argwhere(p).mean(axis=0)
        rc = np.argwhere(r).mean(axis=0)
        out["centroid_dist_mm"] = float(np.linalg.norm(pc - rc) * spacing_mm)
    return out


def dice_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean, min and the count of dice==0 cases. Reported together on purpose.

    On 41 volumes the mean flatters a couple of disasters, and a segmentation system that quietly misses
    one spleen is worse than a classifier that gets the case wrong: nothing downstream flags it.
    """
    d = [r["dice"] for r in rows]
    if not d:
        raise ValueError("no cases to score")
    return {
        "n": len(d),
        "mean_dice": float(np.mean(d)),
        "median_dice": float(np.median(d)),
        "min_dice": float(min(d)),
        "max_dice": float(max(d)),
        "n_zero_dice": int(sum(1 for x in d if x < 1e-9)),
        "n_below_0_5": int(sum(1 for x in d if x < 0.5)),
    }


def predict_volume(net: nn.Module, image: torch.Tensor, roi: Sequence[int] = ROI,
                   sw_batch_size: int = 2, device: str = "cpu", overlap: float = 0.5,
                   num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Sliding-window spleen probability map for one whole volume, returned as (D, H, W) float32.

    `image` is (1, D, H, W) or (1, 1, D, H, W), already windowed to [0, 1]. A whole abdominal CT does not fit
    through a 192^3 network in one pass, so `sliding_window_inference` crops, predicts and blends the
    overlaps with a Gaussian window - and hands the result back in the *original* grid, which is what lets
    the mask be written to NIfTI and opened beside the scan it came from.

    `overlap=0.5` rather than MONAI's default 0.25: with a 192^3 window on a ~350^3 volume the seam between
    windows falls straight through soft tissue, and a half-overlap is the cheap fix that stops the seam
    showing up as a strip of low Dice.
    """
    from monai.inferers import sliding_window_inference

    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5:
        raise ValueError(f"expected (1,1,D,H,W), got {tuple(image.shape)}")
    net = net.to(device).eval()
    x = image.to(device)
    with torch.no_grad():
        out = sliding_window_inference(x, tuple(roi), sw_batch_size, net,
                                       overlap=overlap, mode="gaussian", device=device)
    if out.ndim != 5 or out.shape[1] != num_classes:
        raise AssertionError(f"expected (1, {num_classes}, D, H, W) logits, got {tuple(out.shape)} - "
                             "a deep-supervision output reaching inference means the net is not in eval()")
    probs = torch.softmax(out, dim=1)
    return probs[0, 1].detach().cpu().numpy().astype(np.float32)


def argmask(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Probability map -> boolean mask. Kept separate so the threshold is a reported number, not a default."""
    return np.asarray(probs) > float(threshold)


def save_segmentation_checkpoint(net: nn.Module, path: str, config: dict[str, Any],
                                 epoch: int | None = None, history: list | None = None,
                                 extra: dict[str, Any] | None = None) -> str:
    """Same payload shape as `classifier.save_checkpoint`, detached to CPU.

    A CUDA-tensor checkpoint is unloadable on the laptop that has to review it, so every stored tensor is
    moved to CPU here rather than at each call site.
    """
    payload: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "config": config,
        "epoch": epoch,
        "history": history or [],
        "architecture": "monai DynUNet (3D U-Net), res_block, deep_supervision="
                        f"{config.get('deep_supervision')}",
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"checkpoint: {path} ({os.path.getsize(path):,} bytes)")
    return path


def load_segmentation_checkpoint(net: nn.Module, path: str,
                                 map_location: str = "cpu") -> dict[str, Any]:
    """Strict load, and return the payload so the caller can read the run's CONFIG and history."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(f"{path} has keys {sorted(ckpt)} - expected a 'state_dict' entry")
    missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"loaded {os.path.basename(path)}: epoch={ckpt.get('epoch')} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()
    return ckpt

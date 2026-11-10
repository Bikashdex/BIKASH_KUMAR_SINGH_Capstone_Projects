"""DenseNet121 for the binary lung-opacity head: build, freeze, save, load - in one place.

Notebooks 03 and 04 both need this, and the version tolerance is the reason it is not left inline:
torchvision <= 0.19 hands back `classifier = Sequential(Linear, LogSoftmax)` while >= 0.20 (Colab and
Kaggle run 0.26) hands back a bare `nn.Linear`. Replacing only the inner Linear and leaving the
LogSoftmax makes every logit 0.0, AUROC 0.5, and nothing raises - the bug this function exists to
prevent.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_WEIGHTS_URL = models.DenseNet121_Weights.IMAGENET1K_V1.url
WEIGHTS_FILENAME = os.path.basename(IMAGENET_WEIGHTS_URL)


def stage_pretrained_weights(repo_dir: str | None = None) -> str:
    """Put the ImageNet .pth where torchvision looks, so `weights=IMAGENET1K_V1` needs no internet.

    A Kaggle kernel has no outbound network unless the toggle is turned on (and turning it on needs
    phone verification), so repo.zip carries the file and this copies it into the torch hub cache.
    """
    target = os.path.join(torch.hub.get_dir(), "checkpoints", WEIGHTS_FILENAME)
    shipped = os.path.join(repo_dir, "weights", WEIGHTS_FILENAME) if repo_dir else None
    if os.path.exists(target):
        print(f"weights already in the torch hub cache: {os.path.getsize(target):,} bytes")
    elif shipped and os.path.exists(shipped):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        import shutil

        shutil.copyfile(shipped, target)
        print(f"staged {WEIGHTS_FILENAME} from the code archive into {target}")
    else:
        print(f"WARNING: no {WEIGHTS_FILENAME} in the cache or in repo.zip - this will try to download it")
    return target


def head_in_features(net: nn.Module) -> tuple[int, str]:
    """Channels feeding the classifier head, plus a description of what the head actually was."""
    cls = net.classifier
    if isinstance(cls, nn.Linear):
        return cls.in_features, "Linear"
    if isinstance(cls, nn.Sequential):
        linears = [m for m in cls if isinstance(m, nn.Linear)]
        if linears:
            return linears[-1].in_features, "Sequential" + str([type(m).__name__ for m in cls])
    raise TypeError(f"unexpected DenseNet classifier: {cls}")


def build_densenet121(num_classes: int = 1, pretrained: bool = True,
                      repo_dir: str | None = None, probe_size: int = 224) -> nn.Module:
    """Pretrained DenseNet121 with its head replaced by `num_classes` logits, checked before it returns.

    The probe is the part worth keeping: it asserts the new head emits distinct finite logits for
    distinct inputs, which is exactly what fails when a LogSoftmax survives the swap.
    """
    if pretrained:
        stage_pretrained_weights(repo_dir)
    net = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    num_features, kind = head_in_features(net)
    net.classifier = nn.Linear(num_features, num_classes)
    net.eval()

    gen = torch.Generator().manual_seed(0)
    probe = torch.randn(2, 3, probe_size, probe_size, generator=gen)
    with torch.no_grad():
        out = net(probe)
    assert out.shape == (2, num_classes), out.shape
    assert torch.isfinite(out).all(), "non-finite logits from a fresh head"
    if num_classes == 1:
        assert not torch.allclose(out[0], out[1]), \
            "constant logit for different inputs - the LogSoftmax trap is back"
    print(f"pretrained head was {kind} on {num_features} features; now {net.classifier}")
    return net


def set_frozen(net: nn.Module, frozen: bool) -> int:
    """Train only the head, or the whole backbone. Returns the trainable parameter count."""
    for p in net.features.parameters():
        p.requires_grad = not frozen
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"frozen={frozen} | trainable params {trainable:,}")
    return trainable


def save_checkpoint(net: nn.Module, path: str, config: dict[str, Any],
                    epoch: int | None = None, history: list | None = None,
                    extra: dict[str, Any] | None = None) -> str:
    """Store detached CPU tensors - a checkpoint that remembers CUDA device ids is not loadable elsewhere.

    The whole CONFIG travels with the weights so a reviewer can tell which run produced which numbers.
    """
    payload: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "config": config,
        "epoch": epoch,
        "history": history or [],
        "architecture": "torchvision densenet121, classifier replaced by nn.Linear(_, "
                        f"{net.classifier.out_features})",
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    print(f"checkpoint: {path} ({os.path.getsize(path):,} bytes)")
    return path


def load_checkpoint(net: nn.Module, path: str, map_location: str = "cpu") -> dict[str, Any]:
    """Strict load, returning the payload so the caller can read the stored CONFIG and history."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(f"{path} has keys {sorted(ckpt)} - expected a 'state_dict' entry")
    missing, unexpected = net.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"loaded {os.path.basename(path)}: epoch={ckpt.get('epoch')} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    net.eval()
    return ckpt

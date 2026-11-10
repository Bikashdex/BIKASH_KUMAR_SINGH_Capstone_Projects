"""One definition of what the network actually sees, shared by notebooks 03, 04 and the API.

Duplicating the transform per notebook is how an explained image quietly stops being the image the
model was trained on: a Grad-CAM map computed under a different resize or normalisation is decoration,
not evidence.
"""

from __future__ import annotations

import numpy as np
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gray_to_rgb(x: np.ndarray) -> np.ndarray:
    """(H, W) uint8 -> (H, W, 3): ImageNet backbones want three channels and a chest X-ray has one."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"expected a single-channel image, got shape {x.shape}")
    return np.repeat(x[..., None], 3, axis=2)


def make_transform(size: int, train: bool = False, margin: float = 1.05) -> transforms.Compose:
    """Train or eval pipeline for uint8 HxWx3 arrays coming out of the JPEG cache."""
    ops = [transforms.ToPILImage()]
    if train:
        # Horizontal flip is safe here: the target is "any opacity anywhere", not a laterality-specific
        # finding. No brightness or contrast jitter - notebook 02 measured a -0.061 correlation between
        # mean brightness and the label, so brightness augmentation is likelier to teach a shortcut than
        # to help.
        ops += [transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.85, 1.15))]
    else:
        ops += [transforms.Resize(int(size * margin)), transforms.CenterCrop(size)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

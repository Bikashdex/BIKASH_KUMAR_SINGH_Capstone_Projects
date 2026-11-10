"""Grad-CAM must work on a frozen backbone, under no_grad, and produce stats a reviewer can check.

Everything here runs on a hand-built two-layer network: no ImageNet weights, no GPU, no dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.gradcam import (
    TARGET_LAYER,
    body_mask,
    class_index_for,
    gradcam,
    jet,
    normalize_heatmap,
    otsu_threshold,
    overlay,
    region_stats,
    resolve_module,
)

H = W = 16


class TinyNet(nn.Module):
    """(1, 3, H, W) -> one logit, with a named intermediate the size of a feature map.

    The head is all-ones so the map is literally "the sum of the channel responses": with random head
    weights the four channels cancel each other out and a legitimately empty map fails the shape tests.
    """

    def __init__(self, out_features: int = 1):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("block1", nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1))
        self.head = nn.Linear(4 * H * W // 4, out_features)
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            self.head.bias.zero_()

    def forward(self, x):
        a = self.features.block1(x)
        return self.head(torch.relu(a).flatten(1))


def _image(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, H, W, generator=g)
    x[..., : H // 2, :] += 1.0          # a bright upper half: the map should find it
    return x


def test_map_shape_range_and_spatial_signal():
    net = TinyNet()
    hm = gradcam(net, _image(), target_layer="features.block1", image_size=(H, W))
    assert hm.shape == (H, W)
    assert hm.dtype == np.float32
    assert hm.min() >= 0.0 and hm.max() <= 1.0
    assert hm.max() > hm.min(), "a constant map means the hook or the weighting is broken"


def test_map_without_image_size_stays_on_the_feature_grid():
    net = TinyNet()
    hm = gradcam(net, _image(), target_layer="features.block1")
    assert hm.shape == (H // 2, W // 2)          # stride-2 conv


def test_frozen_backbone_still_yields_a_map_and_keeps_its_flags():
    net = TinyNet()
    for p in net.parameters():
        p.requires_grad_(False)
    hm = gradcam(net, _image(), target_layer="features.block1", image_size=(H, W))
    assert hm.max() > 0
    assert all(not p.requires_grad for p in net.parameters()), \
        "gradcam must put the requires_grad flags back the way it found them"


def test_works_inside_no_grad_because_the_caller_is_evaluating():
    net = TinyNet()
    with torch.no_grad():
        hm = gradcam(net, _image(), target_layer="features.block1", image_size=(H, W))
    assert hm.max() > 0


def test_dead_network_gives_an_empty_map_not_nan():
    net = TinyNet()
    with torch.no_grad():
        net.features.block1.weight.zero_()
        net.features.block1.bias.zero_()
    hm = gradcam(net, _image(), target_layer="features.block1", image_size=(H, W))
    assert np.isnan(hm).sum() == 0
    assert hm.max() == 0.0
    assert region_stats(hm)["note"] == "empty map"


def test_multi_logit_head_demands_an_explicit_class():
    net = TinyNet(out_features=5)
    with pytest.raises(ValueError, match="class_index"):
        gradcam(net, _image(), target_layer="features.block1")
    hm = gradcam(net, _image(), target_layer="features.block1", class_index=2, image_size=(H, W))
    assert hm.shape == (H, W)
    assert class_index_for(torch.zeros(3), 0) == 0
    with pytest.raises(IndexError):
        class_index_for(torch.zeros(3), 3)


def test_bad_input_shape_is_refused():
    net = TinyNet()
    for bad in (torch.rand(3, H, W), torch.rand(2, 3, H, W)):
        with pytest.raises(ValueError, match=r"\(1, 3, H, W\)"):
            gradcam(net, bad, target_layer="features.block1")


def test_resolve_module_tolerates_torchvision_underscore_names():
    net = TinyNet()
    assert resolve_module(net, "features.block1") is net.features.block1
    quirky = TinyNet()
    quirky.features.add_module("denselayer16_", nn.Identity())
    assert resolve_module(quirky, "features.denselayer16") is quirky.features._modules["denselayer16_"]
    with pytest.raises(AttributeError, match="no submodule"):
        resolve_module(net, "features.not_there")


def test_default_target_layer_exists_on_the_real_backbone():
    # if torchvision renames denseblock4, notebook 04 dies on its first call - catch it here instead
    from torchvision import models

    net = models.densenet121(weights=None)
    layer = resolve_module(net, TARGET_LAYER)
    assert type(layer).__name__ == "_DenseBlock"
    probe = torch.zeros(1, 3, 224, 224)
    for p in net.parameters():
        p.requires_grad_(False)          # exactly what notebook 04 does before asking for a map
    grid = gradcam(net, probe, class_index=0)      # weights=None leaves the 1000-way ImageNet head on
    assert grid.shape == (7, 7), "224 px in, /32 = 7x7 - anything else means the hook fired elsewhere"
    assert gradcam(net, probe, class_index=0, image_size=(224, 224)).shape == (224, 224)


def test_normalize_and_resize():
    assert normalize_heatmap(np.zeros((4, 4))).max() == 0.0
    flat = normalize_heatmap(np.full((4, 4), 7.0))
    assert flat.max() == 0.0, "a constant map has no information to show"
    ramp = normalize_heatmap(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert ramp[0, 0] == 0.0 and ramp[2, 3] == 1.0
    big = normalize_heatmap(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32))
    up = np.asarray(overlay(np.zeros((8, 8), np.uint8), big), dtype=np.float32)
    assert up.shape == (8, 8, 3)


def test_overlay_tints_only_where_attention_is_strong():
    grey = np.full((8, 8), 120, dtype=np.uint8)
    hm = np.zeros((8, 8), np.float32)
    hm[0:2, 0:2] = 1.0
    out = overlay(grey, hm, alpha=0.5, threshold=0.3)
    assert out.dtype == np.uint8 and out.shape == (8, 8, 3)
    quiet = out[6, 6]
    assert (quiet == np.array([120, 120, 120])).all(), "low-heat pixels must stay pure grey"
    hot = out[0, 0]
    assert not (hot == quiet).all(), "the peak has to be visibly tinted"
    # the jet ramp: cold end dark blue, hot end dark red
    assert jet(np.array([0.0]))[0].tolist() == [0.0, 0.0, 0.5]
    assert jet(np.array([1.0]))[0].tolist() == [0.5, 0.0, 0.0]


def test_body_mask_finds_a_bright_chest_on_a_dark_background():
    # background 8, not 0: on a pure 0/180 image every cut-off between them maximises the between-class
    # variance equally and Otsu legitimately returns the lowest one
    img = np.full((64, 64), 8, dtype=np.uint8)
    img[16:48, 16:48] = 180
    m = body_mask(img)
    assert 0.9 * 32 * 32 <= m.sum() <= 1.1 * 32 * 32
    assert 8 <= otsu_threshold(img) < 180, "the cut-off lands on the background level; body_mask uses >"


def test_region_stats_see_where_the_attention_went():
    mask = np.zeros((16, 16), bool)
    mask[:, 2:14] = True
    inside = np.zeros((16, 16), np.float32)
    inside[8, 8] = 1.0
    stats = region_stats(inside, mask)
    assert stats["in_body_fraction"] == pytest.approx(1.0)
    assert stats["centroid"] == pytest.approx((0.5333, 0.5333), abs=1e-3)
    assert stats["spread"][0] == 0.0

    outside = np.zeros((16, 16), np.float32)
    outside[0, 0] = 1.0
    assert region_stats(outside, mask)["in_body_fraction"] == pytest.approx(0.0)

    left = np.zeros((16, 16), np.float32)
    left[:, 0] = 1.0
    right = np.zeros((16, 16), np.float32)
    right[:, -1] = 1.0
    # viewer's-left mass is positive bias: the patient's RIGHT lung, by radiological convention
    assert region_stats(left, mask)["lateral_bias"] == pytest.approx(1.0)
    assert region_stats(right, mask)["lateral_bias"] == pytest.approx(-1.0)

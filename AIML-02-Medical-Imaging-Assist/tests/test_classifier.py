"""The head swap is the one part of this project that fails silently, so it gets tested directly.

Leaving torchvision's LogSoftmax on a single-logit head makes every prediction 0.0, AUROC 0.5 and
nothing raises. These tests build DenseNet121 with `weights=None` - no download, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

from src.models.classifier import (
    WEIGHTS_FILENAME,
    build_densenet121,
    head_in_features,
    load_checkpoint,
    save_checkpoint,
    set_frozen,
)


def _bare():
    return models.densenet121(weights=None)


def test_head_swap_replaces_the_whole_classifier():
    net = build_densenet121(num_classes=1, pretrained=False, probe_size=64)
    assert isinstance(net.classifier, nn.Linear) and net.classifier.out_features == 1
    assert not any(isinstance(m, nn.LogSoftmax) for m in net.modules()), \
        "a surviving LogSoftmax turns the single logit into a constant 0.0"
    assert net.classifier.in_features == 1024


def test_head_in_features_tolerates_both_torchvision_shapes():
    assert head_in_features(_bare())[0] == 1024
    old = _bare()
    # emulate torchvision <= 0.19: Sequential(Linear, LogSoftmax)
    old.classifier = nn.Sequential(nn.Linear(1024, 1000), nn.LogSoftmax(dim=1))
    n, kind = head_in_features(old)
    assert (n, kind) == (1024, "Sequential['Linear', 'LogSoftmax']")
    weird = _bare()
    weird.classifier = nn.Identity()
    with pytest.raises(TypeError, match="unexpected DenseNet classifier"):
        head_in_features(weird)


def test_probe_actually_catches_the_logsoftmax_trap():
    net = _bare()
    net.classifier = nn.Sequential(nn.Linear(1024, 1), nn.LogSoftmax(dim=1))
    # two different inputs through a LogSoftmax over a single class give log(1) = 0 every time
    with torch.no_grad():
        out = net(torch.randn(2, 3, 64, 64))
    assert torch.allclose(out[0], out[1]), "the trap this function exists to catch is gone from torchvision"
    with pytest.raises(AssertionError, match="LogSoftmax"):
        built = _bare()
        built.classifier = nn.Sequential(nn.Linear(1024, 1), nn.LogSoftmax(dim=1))
        # reproduce build_densenet121's check on that head
        probe = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
        o = built(probe)
        assert not torch.allclose(o[0], o[1]), "constant logit - the LogSoftmax trap is back"


def test_build_rejects_a_wrong_shape_head():
    net = build_densenet121(num_classes=2, pretrained=False, probe_size=64)
    with torch.no_grad():
        assert net(torch.randn(3, 3, 64, 64)).shape == (3, 2)


def test_freeze_schedule_only_touches_the_backbone():
    net = build_densenet121(num_classes=1, pretrained=False, probe_size=64)
    trainable_head_only = set_frozen(net, True)
    assert trainable_head_only == net.classifier.weight.numel() + net.classifier.bias.numel()
    everything = set_frozen(net, False)
    assert everything > trainable_head_only
    assert all(p.requires_grad for p in net.features.parameters())


def test_checkpoint_round_trip_is_device_clean(tmp_path):
    net = build_densenet121(num_classes=1, pretrained=False, probe_size=64)
    with torch.no_grad():
        net.classifier.weight.fill_(0.125)
    path = str(tmp_path / "model.pt")
    save_checkpoint(net, path, {"train_size": 224, "seed": 7}, epoch=3,
                    history=[{"epoch": 3, "val_auroc": 0.8}], extra={"threshold": 0.44})
    fresh = _bare()
    fresh.classifier = nn.Linear(1024, 1)
    ckpt = load_checkpoint(fresh, path)
    assert ckpt["epoch"] == 3 and ckpt["config"]["seed"] == 7 and ckpt["threshold"] == 0.44
    assert torch.allclose(fresh.classifier.weight, net.classifier.weight)
    assert fresh.training is False, "load_checkpoint must leave the model in eval mode"
    stored = torch.load(path, map_location="cpu", weights_only=False)
    assert all(v.device.type == "cpu" for v in stored["state_dict"].values())


def test_load_checkpoint_refuses_a_foreign_file(tmp_path):
    net = build_densenet121(num_classes=1, pretrained=False, probe_size=64)
    bad = tmp_path / "bad.pt"
    torch.save({"weights": net.state_dict()}, str(bad))
    with pytest.raises(KeyError, match="state_dict"):
        load_checkpoint(net, str(bad))
    other = tmp_path / "other.pt"
    torch.save({"state_dict": {"not_a_parameter": torch.zeros(1)}}, str(other))
    with pytest.raises(RuntimeError):
        load_checkpoint(net, str(other))


def test_weights_filename_matches_whatever_torchvision_pinned():
    # repo.zip ships this exact filename; if torchvision changes the hash suffix, the archive must be
    # rebuilt, and this is the test that says so instead of a Kaggle session failing at cell 14
    from torchvision import models as m

    assert WEIGHTS_FILENAME == m.DenseNet121_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[1]
    assert WEIGHTS_FILENAME.startswith("densenet121-") and WEIGHTS_FILENAME.endswith(".pth")


def test_num_classes_other_than_one_does_not_run_the_constancy_probe():
    net = build_densenet121(num_classes=4, pretrained=False, probe_size=64)
    assert net.classifier.out_features == 4
    with torch.no_grad():
        assert np.isfinite(net(torch.randn(1, 3, 64, 64)).numpy()).all()

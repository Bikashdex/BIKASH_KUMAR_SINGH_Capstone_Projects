"""CPU tests for the spleen 3D U-Net module. Synthetic volumes and a miniature DynUNet only - nothing here
needs the 1.6 GB MSD download, which is why the segmentation track can be tested on a laptop without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.unet3d import (binary_dice, build_dynunet, deep_supervision_loss, dice_report,
                               dynunet_filters, dynunet_strides, load_segmentation_checkpoint,
                               make_loss, predict_volume, save_segmentation_checkpoint,
                               unpack_deep_supervision, volume_metrics)


# --- the geometry helpers -------------------------------------------------------------------------------

def test_strides_are_one_entry_per_stage():
    s = dynunet_strides(5)
    assert len(s) == 6
    assert s[0] == [1, 1, 1]
    assert all(v == [2, 2, 2] for v in s[1:])
    with pytest.raises(ValueError):
        dynunet_strides(0)
    with pytest.raises(ValueError):
        dynunet_strides(7)


def test_filters_match_the_msd_recipe():
    # 32 -> doubling -> capped at 320, which is what MONAI itself defaults to. Asserting the numbers means
    # a MONAI upgrade that changes its default cannot silently change this project's architecture.
    assert dynunet_filters(5) == [32, 64, 128, 256, 320, 320]
    assert dynunet_filters(2) == [32, 64, 128]
    assert dynunet_filters(3, initial_filters=16) == [16, 32, 64, 128]


# --- Dice, reimplemented on purpose ---------------------------------------------------------------------

def test_dice_known_values():
    a = np.zeros((4, 4, 4), dtype=bool)
    a[1:3, 1:3, 1:3] = True          # 8 voxels
    b = np.zeros_like(a)
    b[1:3, 1:3, 1:3] = True          # identical
    assert binary_dice(a, b) == pytest.approx(1.0, abs=1e-5)
    c = np.zeros_like(a)
    c[0, 0, 0] = True                # disjoint
    assert binary_dice(a, c) == pytest.approx(0.0, abs=1e-4)
    d = np.zeros_like(a)
    d[1:3, 1:3, :2] = True           # half of a overlaps -> 2*4/(8+8) = 0.5
    assert binary_dice(a, d) == pytest.approx(0.5, abs=1e-4)


def test_dice_symmetric_and_shape_checked():
    rng = np.random.default_rng(0)
    p, r = rng.random((6, 7, 8)) > 0.5, rng.random((6, 7, 8)) > 0.6
    assert binary_dice(p, r) == pytest.approx(binary_dice(r, p))
    with pytest.raises(ValueError):
        binary_dice(p, r[:, :5, :])


def test_dice_both_empty_is_one_and_stays_visible():
    z = np.zeros((3, 3, 3), dtype=bool)
    assert binary_dice(z, z) == 1.0
    m = volume_metrics(z, z)
    assert m["pred_empty"] and m["ref_empty"] and m["centroid_dist_mm"] is None


def test_dice_agrees_with_monai():
    from monai.metrics import compute_dice

    rng = np.random.default_rng(3)
    for _ in range(3):
        ref = rng.random((16, 20, 24)) > 0.8
        pred = np.logical_and(ref, rng.random(ref.shape) > 0.25)
        if not ref.any():
            continue
        mine = binary_dice(pred, ref)
        # MONAI wants (B, C, spatial) one-hot-ish channel tensors with the foreground in channel 1.
        stack = torch.from_numpy(np.stack([~pred, pred])[None]).float()
        target = torch.from_numpy(np.stack([~ref, ref])[None]).float()
        theirs = float(compute_dice(stack, target, include_background=False))
        assert mine == pytest.approx(theirs, abs=1e-4), (mine, theirs)


def test_volume_metrics_reports_shift_not_just_dice():
    a = np.zeros((20, 20, 20), dtype=bool)
    a[4:12, 4:12, 4:12] = True
    b = np.zeros_like(a)
    b[8:16, 8:16, 8:16] = True          # the same 8^3 cube, translated by 4 voxels on every axis
    m = volume_metrics(b, a, spacing_mm=2.0)
    assert 0.0 < m["dice"] < 1.0
    assert m["pred_volume_mm3"] == pytest.approx(m["ref_volume_mm3"])
    assert m["pred_voxels"] == m["ref_voxels"] == 512
    # centroids differ by 4 voxels on each axis -> 4*sqrt(3)*2 mm
    assert m["centroid_dist_mm"] == pytest.approx(4 * np.sqrt(3) * 2.0, rel=1e-3)


def test_dice_report_keeps_the_worst_case_visible():
    rows = [{"dice": d} for d in (0.95, 0.91, 0.4, 0.0, 0.8)]
    r = dice_report(rows)
    assert r == {"n": 5, "mean_dice": pytest.approx(np.mean([0.95, 0.91, 0.4, 0.0, 0.8])),
                 "median_dice": pytest.approx(0.8), "min_dice": 0.0, "max_dice": 0.95,
                 "n_zero_dice": 1, "n_below_0_5": 2}
    with pytest.raises(ValueError):
        dice_report([])


# --- the network ----------------------------------------------------------------------------------------

def test_build_dynunet_probe_returns_full_resolution():
    # 3 downsample stages on a 48^3 input: small enough for a laptop, and the probe is what proves strides,
    # filters, upsample_kernel_size and the head count all agree. MONAI only refuses at the first forward.
    net = build_dynunet(num_downsample=3, num_aux_heads=2, deep_supervision=True,
                        probe_input=(48, 48, 48))
    assert sum(p.numel() for p in net.parameters()) > 1e5


def test_build_dynunet_rejects_more_heads_than_layers():
    # MONAI's own message for this is "deep_supr_num should be less than the number of up sample layers",
    # which means nothing to someone reading a notebook traceback six cells later.
    with pytest.raises(ValueError, match="num_aux_heads must be < num_downsample"):
        build_dynunet(num_downsample=2, num_aux_heads=2)


def test_deep_supervision_shapes_train_and_eval_differ():
    net = build_dynunet(num_downsample=3, num_aux_heads=2, deep_supervision=True)
    x = torch.zeros(2, 1, 48, 48, 48)
    net.train()
    out = net(x)
    assert tuple(out.shape) == (2, 3, 2, 48, 48, 48), tuple(out.shape)
    main, aux = unpack_deep_supervision(out)
    assert main.shape == (2, 2, 48, 48, 48), tuple(main.shape)
    assert len(aux) == 2, "two auxiliary heads means three outputs"
    net.eval()
    with torch.no_grad():
        one = net(x)
    main_eval, aux_eval = unpack_deep_supervision(one)
    assert main_eval.shape == (2, 2, 48, 48, 48) and aux_eval == []


def test_deep_supervision_loss_weights_auxiliary_heads():
    loss_fn = make_loss()
    target = torch.zeros(2, 1, 32, 32, 32)
    target[:, 0, 4:12, 4:12, 4:12] = 1.0
    stacked = torch.zeros(2, 2, 2, 32, 32, 32)
    stacked[:, :, 1, 4:12, 4:12, 4:12] = 4.0
    main_only = loss_fn(stacked[:, 0], target)
    whole = deep_supervision_loss(stacked, target, loss_fn, weight=0.5)
    assert float(whole) > float(main_only), "the auxiliary head must add loss, not remove it"
    plain = deep_supervision_loss(stacked[:, 0], target, loss_fn, weight=0.5)
    assert float(plain) == pytest.approx(float(main_only))


def test_loss_reaches_the_network_and_dice_improves():
    """A 3-step overfit on a synthetic blob: the only test that proves the plumbing can learn at all."""
    torch.manual_seed(0)
    net = build_dynunet(num_downsample=2, num_aux_heads=1, deep_supervision=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = make_loss()

    img = np.zeros((1, 1, 40, 40, 40), dtype=np.float32)
    img[0, 0, 10:26, 12:30, 8:24] = 0.8
    mask = np.zeros((1, 1, 40, 40, 40), dtype=np.float32)
    mask[0, 0, 10:26, 12:30, 8:24] = 1.0
    x, y = torch.from_numpy(img), torch.from_numpy(mask)

    net.train()
    first = None
    for step in range(4):
        opt.zero_grad()
        out = net(x)
        loss = deep_supervision_loss(out, y, loss_fn)
        assert torch.isfinite(loss), f"loss went non-finite at step {step}"
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    net.eval()
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0, 1].numpy() > 0.5
    dice = binary_dice(probs, mask[0, 0] > 0.5)
    assert last < first, f"loss did not fall ({first:.4f} -> {last:.4f})"
    assert dice > 0.5, f"a trainable net should overfit a solid blob; dice {dice:.3f}"


def test_predict_volume_shape_and_range():
    net = build_dynunet(num_downsample=2, num_aux_heads=1, deep_supervision=True)
    img = torch.zeros(1, 40, 40, 40, dtype=torch.float32)
    probs = predict_volume(net, img, roi=(24, 24, 24), sw_batch_size=1, device="cpu")
    assert probs.shape == (40, 40, 40)
    assert probs.dtype == np.float32
    assert 0.0 <= float(probs.min()) and float(probs.max()) <= 1.0


# --- checkpoints ----------------------------------------------------------------------------------------

def test_checkpoint_round_trip_is_cpu_clean(tmp_path):
    net = build_dynunet(num_downsample=2, num_aux_heads=1, deep_supervision=True)
    path = str(tmp_path / "dynunet.pt")
    payload = save_segmentation_checkpoint(net, path, {"roi": [64, 64, 64], "deep_supervision": True},
                                           epoch=3, history=[{"dice": 0.1}], extra={"m": 1})
    import os
    assert os.path.getsize(path) > 1e5
    assert all(v.device.type == "cpu" for v in
               torch.load(path, map_location="cpu", weights_only=False)["state_dict"].values())

    fresh = build_dynunet(num_downsample=2, num_aux_heads=1, deep_supervision=True)
    ck = load_segmentation_checkpoint(fresh, path)
    assert ck["config"]["roi"] == [64, 64, 64] and ck["epoch"] == 3 and ck["m"] == 1
    net.eval(), fresh.eval()
    x = torch.zeros(1, 1, 40, 40, 40)
    with torch.no_grad():
        assert torch.allclose(net(x), fresh(x), atol=1e-5)


def test_load_rejects_a_foreign_checkpoint(tmp_path):
    from torchvision.models import densenet121

    path = str(tmp_path / "wrong.pt")
    torch.save({"state_dict": densenet121(num_classes=1).state_dict()}, path)
    net = build_dynunet(num_downsample=2, num_aux_heads=1, deep_supervision=False)
    with pytest.raises(RuntimeError):
        load_segmentation_checkpoint(net, path)
    bad = str(tmp_path / "not_a_ckpt.pt")
    torch.save({"weights": 1}, bad)
    with pytest.raises(KeyError):
        load_segmentation_checkpoint(net, bad)

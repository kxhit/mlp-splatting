from __future__ import annotations

import torch


def test_feature_only_loss_does_not_launch_rgb_backward(monkeypatch):
    from mip360 import cumodel

    calls = {"rgb_backward": 0, "feature_backward": 0}

    class FakeExtension:
        @staticmethod
        def render_tiles_forward_tilecenter_with_feat(*args):
            rays_o = args[0]
            feat_attr = args[15]
            batch = rays_o.shape[0]
            channels = feat_attr.shape[-1]
            return (
                torch.zeros(batch, 3),
                torch.zeros(batch, 1),
                torch.zeros(batch, channels),
            )

        @staticmethod
        def render_tiles_backward_tilecenter_full(*args):
            calls["rgb_backward"] += 1
            raise AssertionError("unused RGB output must not launch RGB backward")

        @staticmethod
        def render_tiles_backward_featattr_tilecenter(*args):
            calls["feature_backward"] += 1
            return torch.zeros_like(args[15])

    fake = FakeExtension()
    monkeypatch.setattr(
        cumodel.NeuralPrimitiveSceneTileCUDA,
        "_ensure_tile_ext_static",
        staticmethod(lambda _tile_size=None: fake),
    )

    rays_o = torch.zeros(1, 3)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    positions = torch.zeros(1, 3, requires_grad=True)
    log_scales = torch.zeros(1, 3, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    candidates = torch.zeros(1, dtype=torch.int32)
    offsets = torch.tensor([0, 1], dtype=torch.int32)
    frequency = torch.ones(1)
    attenuation = torch.ones(1)
    w1 = torch.zeros(1, 2, 12, requires_grad=True)
    b1 = torch.zeros(1, 2, requires_grad=True)
    w2 = torch.zeros(1, 2, 2, requires_grad=True)
    b2 = torch.zeros(1, 2, requires_grad=True)
    w3 = torch.zeros(1, 4, 2, requires_grad=True)
    b3 = torch.zeros(1, 4, requires_grad=True)
    feat_attr = torch.zeros(1, 4, 2, requires_grad=True)

    _rgb, feature = cumodel._TileRenderWithFeatAutogradFn.apply(
        rays_o,
        rays_d,
        positions,
        log_scales,
        quats,
        candidates,
        offsets,
        frequency,
        attenuation,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        feat_attr,
        1.5,
        2.0,
        1,
        1,
        1,
        1,
        1.0e-3,
        None,
    )
    feature.sum().backward()

    assert calls == {"rgb_backward": 0, "feature_backward": 1}
    assert feat_attr.grad is not None
    for unrelated in (positions, log_scales, quats, w1, b1, w2, b2, w3, b3):
        assert unrelated.grad is None

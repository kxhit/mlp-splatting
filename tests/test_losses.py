from __future__ import annotations

import pytest
import torch

from mip360.utils.loss_utils import l1_loss, l2_loss, ssim


def test_basic_losses_and_identical_image_ssim():
    prediction = torch.tensor([0.0, 0.5, 1.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(l1_loss(prediction, target), torch.tensor(0.5))
    torch.testing.assert_close(l2_loss(prediction, target), torch.tensor(1.25 / 3.0))

    image = torch.rand(2, 3, 16, 16)
    torch.testing.assert_close(ssim(image, image), torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    per_image = ssim(image, image, size_average=False)
    assert per_image.shape == (2,)
    torch.testing.assert_close(per_image, torch.ones(2), rtol=1e-5, atol=1e-5)


def test_ssim_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        ssim(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 7, 8))
    with pytest.raises(ValueError, match="NCHW"):
        ssim(torch.zeros(3, 8, 8), torch.zeros(3, 8, 8))

"""Small image-loss helpers used by training and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean()


def l2_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean()


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Return a normalized, centered 1D Gaussian kernel."""

    coordinates = torch.arange(window_size, dtype=torch.float32)
    coordinates -= window_size // 2
    kernel = torch.exp(-coordinates.square() / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create a depthwise 2D Gaussian window for SSIM."""

    kernel_1d = gaussian(window_size, 1.5)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size).expand(
        channel, 1, window_size, window_size
    ).contiguous()


def ssim(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """Compute structural similarity for NCHW tensors with values in [0, 1]."""

    if image_a.shape != image_b.shape:
        raise ValueError(
            f"SSIM inputs must have identical shapes, got {image_a.shape} and {image_b.shape}"
        )
    if image_a.ndim != 4:
        raise ValueError(f"SSIM expects NCHW inputs, got shape {image_a.shape}")
    channel = int(image_a.shape[-3])
    window = create_window(window_size, channel).to(
        device=image_a.device,
        dtype=image_a.dtype,
    )
    return _ssim(
        image_a,
        image_b,
        window,
        window_size,
        channel,
        size_average,
    )


def _ssim(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    padding = window_size // 2
    mean_a = F.conv2d(image_a, window, padding=padding, groups=channel)
    mean_b = F.conv2d(image_b, window, padding=padding, groups=channel)
    mean_a_sq = mean_a.square()
    mean_b_sq = mean_b.square()
    mean_ab = mean_a * mean_b

    variance_a = (
        F.conv2d(image_a.square(), window, padding=padding, groups=channel)
        - mean_a_sq
    )
    variance_b = (
        F.conv2d(image_b.square(), window, padding=padding, groups=channel)
        - mean_b_sq
    )
    covariance = (
        F.conv2d(image_a * image_b, window, padding=padding, groups=channel)
        - mean_ab
    )

    stabilizer_luminance = 0.01**2
    stabilizer_contrast = 0.03**2
    score = (
        (2.0 * mean_ab + stabilizer_luminance)
        * (2.0 * covariance + stabilizer_contrast)
        / (
            (mean_a_sq + mean_b_sq + stabilizer_luminance)
            * (variance_a + variance_b + stabilizer_contrast)
        )
    )
    if size_average:
        return score.mean()
    return score.mean(dim=(1, 2, 3))

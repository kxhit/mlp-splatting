import torch
import torch.nn as nn


class FeatureUpsampleDecoder(nn.Module):
    """Feature-3DGS-style 3-layer conv decoder: 1x1 -> 3x3 -> 1x1."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

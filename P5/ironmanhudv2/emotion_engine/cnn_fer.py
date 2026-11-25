from collections.abc import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional feature block with two Conv-BN-ReLU layers, pooling, and dropout."""

    def __init__(self, in_ch: int, out_ch: int, drop: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNFer(nn.Module):
    """FER-2013 grayscale CNN architecture matching the trained PyTorch checkpoints."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 7,
        filters: Sequence[int] = (32, 64, 128, 256),
        dropouts: Sequence[float] = (0.10, 0.15, 0.20, 0.25),
        head_units: int = 256,
        head_dropout: float = 0.40,
        aug_rotation_deg: float = 10.8,
        aug_zoom: float = 0.10,
    ):
        super().__init__()
        self.aug_rotation_deg = aug_rotation_deg
        self.aug_zoom = aug_zoom

        blocks = []
        ch = in_channels
        for f, d in zip(filters, dropouts):
            blocks.append(ConvBlock(ch, f, d))
            ch = f
            
        self.features = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, head_units, bias=False),
            nn.BatchNorm1d(head_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_units, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)
            
        x = x.to(dtype=torch.float32) / 255.0
        x = self.features(x)
        x = self.gap(x)
        return self.head(x)

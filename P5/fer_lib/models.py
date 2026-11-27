"""Arquitecturas PyTorch (CNNFer, TransferFER, EnsembleFER) y carga unificada de checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2

from .config import TL_IMG_SIZE


class ConvBlock(nn.Module):
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
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 7,
        filters: list[int] | None = None,
        dropouts: list[float] | None = None,
        head_units: int = 256,
        head_dropout: float = 0.40,
        aug_rotation_deg: float = 10.8,
        aug_zoom: float = 0.10,
    ):
        super().__init__()
        filters = filters if filters is not None else [48, 96, 192, 384]
        dropouts = dropouts if dropouts is not None else [0.05, 0.10, 0.15, 0.20]

        self.augment = nn.Sequential(
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(
                degrees=aug_rotation_deg,
                translate=(0.08, 0.08),
                scale=(1.0 - aug_zoom, 1.0 + aug_zoom),
            ),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        )

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
        if self.training:
            x = self.augment(x)
        x = self.features(x)
        x = self.gap(x)
        return self.head(x)


class TransferFER(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        n_classes: int = 7,
        tl_img_size: int = TL_IMG_SIZE,
        aug_rotation_deg: float = 10.8,
        aug_zoom: float = 0.10,
        head_dropout: float = 0.30,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.tl_img_size = tl_img_size

        self.augment = nn.Sequential(
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(
                degrees=aug_rotation_deg,
                translate=(0.08, 0.08),
                scale=(1.0 - aug_zoom, 1.0 + aug_zoom),
            ),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        )

        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            in_feats = net.fc.in_features
            net.fc = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_feats, n_classes),
            )
            self.head_module = net.fc
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_feats = net.classifier[1].in_features
            net.classifier = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_feats, n_classes),
            )
            self.head_module = net.classifier
        else:
            raise ValueError(f"Backbone no soportado: {backbone_name}")

        self.backbone = net

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        for p in self.head_module.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)
        x = x.to(dtype=torch.float32) / 255.0
        if self.training:
            x = self.augment(x)
        if x.shape[-1] != self.tl_img_size or x.shape[-2] != self.tl_img_size:
            x = F.interpolate(
                x, size=(self.tl_img_size, self.tl_img_size), mode="bilinear", align_corners=False
            )
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x)


class EnsembleFER(nn.Module):
    def __init__(self, submodels: list[nn.Module]):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)

    def forward_probs(self, x: torch.Tensor, use_tta: bool = True) -> torch.Tensor:
        probs_sum = None
        for net in self.submodels:
            p = F.softmax(net(x), dim=-1)
            if use_tta:
                p_flip = F.softmax(net(torch.flip(x, dims=[-1])), dim=-1)
                p = 0.5 * (p + p_flip)
            probs_sum = p if probs_sum is None else (probs_sum + p)
        return probs_sum / len(self.submodels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward_probs(x, use_tta=False)
        return torch.log(probs.clamp_min(1e-8))


def build_from_arch(arch: dict, state_dict: dict, device: torch.device | str) -> nn.Module:
    arch_cfg = dict(arch)
    if "backbone_name" in arch_cfg:
        arch_cfg["pretrained"] = False
        net = TransferFER(**arch_cfg).to(device)
    else:
        net = CNNFer(**arch_cfg).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def load_checkpoint(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, list[str], dict]:
    dev = torch.device(device)
    map_loc = dev if dev.type == "cuda" else "cpu"
    ckpt = torch.load(Path(ckpt_path), map_location=map_loc, weights_only=True)
    class_names = ckpt["class_names"]

    if "models" in ckpt:
        submodels = [
            build_from_arch(sub["arch"], sub["model_state_dict"], dev)
            for sub in ckpt["models"].values()
        ]
        model = EnsembleFER(submodels).to(dev)
        model.eval()
        return model, class_names, ckpt

    model = build_from_arch(ckpt["arch"], ckpt["model_state_dict"], dev)
    return model, class_names, ckpt

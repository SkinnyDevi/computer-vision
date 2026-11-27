import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock matching torchvision's parameter layout."""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class _ResNet18(nn.Module):
    """Pure-PyTorch ResNet-18 matching torchvision's state_dict keys."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [_BasicBlock(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class TransferFER(nn.Module):
    """Transfer-learning FER architecture matching trained PyTorch checkpoints."""

    def __init__(
        self,
        backbone_name: str,
        n_classes: int = 7,
        tl_img_size: int = 112,
        aug_rotation_deg: float = 10.8,
        aug_zoom: float = 0.10,
        head_dropout: float = 0.30,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.tl_img_size = tl_img_size
        self.aug_rotation_deg = aug_rotation_deg
        self.aug_zoom = aug_zoom

        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if backbone_name == "resnet18":
            if pretrained:
                try:
                    from torchvision import models
                except ImportError as exc:
                    raise ImportError(
                        "Loading pretrained resnet18 weights requires torchvision to be installed."
                    ) from exc
                net: nn.Module = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                net = _ResNet18()
            in_feats = net.fc.in_features
            net.fc = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_feats, n_classes),
            )
            self.head_module = net.fc
        elif backbone_name == "efficientnet_b0":
            try:
                from torchvision import models
            except ImportError as exc:
                raise ImportError(
                    "Loading efficientnet_b0 requires torchvision to be installed."
                ) from exc
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_feats = net.classifier[1].in_features
            net.classifier = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_feats, n_classes),
            )
            self.head_module = net.classifier
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)

        x = x.to(dtype=torch.float32) / 255.0
        if x.shape[-1] != self.tl_img_size or x.shape[-2] != self.tl_img_size:
            x = F.interpolate(
                x,
                size=(self.tl_img_size, self.tl_img_size),
                mode="bilinear",
                align_corners=False,
            )
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.imagenet_mean) / self.imagenet_std
        return self.backbone(x)

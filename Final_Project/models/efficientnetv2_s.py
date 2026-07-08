"""Pure-torch EfficientNetV2-S style classifier without pretrained weights.

The goal here is not to reproduce every detail of the original architecture.
Instead, we keep the same design principles:
- early fused convolutional stages
- later mobile inverted bottleneck stages
- a lightweight classifier head
- an exposed final feature map for Grad-CAM-style inspection
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Convolution -> BatchNorm -> SiLU block used throughout the network."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """Channel reweighting block used in the deeper EfficientNet stages."""

    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        self.activation = nn.SiLU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.activation(self.reduce(scale))
        scale = self.gate(self.expand(scale))
        return x * scale


class FusedMBConv(nn.Module):
    """EfficientNetV2 fused block: expansion and spatial filtering in one stage."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int = 1) -> None:
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_channels, kernel_size=1))
        else:
            hidden_channels = in_channels
        layers.append(ConvBNAct(hidden_channels, out_channels, kernel_size=3, stride=stride))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_residual else out


class MBConv(nn.Module):
    """Mobile inverted bottleneck block with squeeze-excitation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int = 4, se_ratio: float = 0.25) -> None:
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        squeeze_channels = max(1, int(hidden_channels * se_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels)
        self.se = SqueezeExcitation(hidden_channels, squeeze_channels)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        return out + x if self.use_residual else out


class EfficientNetV2S(nn.Module):
    """Lightweight EfficientNetV2-S style wrapper with local initialization."""

    def __init__(self, num_classes: int = 20, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        # Stem: bring the RGB input into a compact feature space.
        self.stem = ConvBNAct(3, 24, kernel_size=3, stride=2)

        # Fused stages: these are the early, hardware-friendly EfficientNetV2 blocks.
        self.fused_stages = nn.Sequential(
            FusedMBConv(24, 24, stride=1, expand_ratio=1),
            FusedMBConv(24, 48, stride=2, expand_ratio=4),
            FusedMBConv(48, 64, stride=2, expand_ratio=4),
        )

        # Deeper MBConv stages: add channel attention and larger receptive fields.
        self.mbconv_stages = nn.Sequential(
            MBConv(64, 128, stride=2, expand_ratio=4),
            MBConv(128, 160, stride=1, expand_ratio=4),
            MBConv(160, 256, stride=2, expand_ratio=4),
        )

        # Head: compress the feature map before classification.
        self.head = nn.Sequential(
            ConvBNAct(256, 512, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        # Store the output of the final convolutional feature map for Grad-CAM.
        self.last_feature_map: torch.Tensor | None = None
        self.mbconv_stages.register_forward_hook(self._capture_last_feature_map)

        self._initialize_weights()
        self.to(self.device)

    def _capture_last_feature_map(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        """Cache the last convolutional feature map from the backbone."""

        self.last_feature_map = output

    def _initialize_weights(self) -> None:
        """Apply Kaiming initialization to convolution and linear layers."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits."""

        x = self.stem(x)
        x = self.fused_stages(x)
        x = self.mbconv_stages(x)
        return self.head(x)

    def get_last_feature_map(self) -> torch.Tensor | None:
        """Return the cached feature map from the most recent forward pass."""

        return self.last_feature_map
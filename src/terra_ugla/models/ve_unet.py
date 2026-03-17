"""VE-focused Robust UNet model definition."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        att = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(att)


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.w_g(g) + self.w_x(x))
        return x * self.psi(psi)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.sa(self.ca(out))
        return self.relu(out + residual)


class DilatedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = max(1, out_channels // 4)
        self.conv1 = nn.Conv2d(in_channels, c, 1)
        self.conv2 = nn.Conv2d(in_channels, c, 3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, c, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, c, 3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(c * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)
        return self.relu(self.bn(out))


class RobustUNet(nn.Module):
    """Robust UNet that returns logits by default for BCEWithLogits loss."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1, base_channels: int = 64, apply_sigmoid: bool = False):
        super().__init__()
        self.inc = ResidualBlock(n_channels, base_channels, dropout_rate=0.1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels, base_channels * 2, dropout_rate=0.1))
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(base_channels * 2, base_channels * 4, dropout_rate=0.2),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(base_channels * 4, base_channels * 8, dropout_rate=0.2),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DilatedBlock(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16, base_channels * 16, dropout_rate=0.3),
        )

        self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
        self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
        self.att1 = AttentionGate(base_channels, base_channels, max(1, base_channels // 2))

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8, dropout_rate=0.2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4, dropout_rate=0.2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, dropout_rate=0.1)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, dropout_rate=0.1)

        self.outc = nn.Conv2d(base_channels, n_classes, 1)
        self.apply_sigmoid = bool(apply_sigmoid)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up4(x5)
        x = torch.cat([self.att4(x, x4), x], dim=1)
        x = self.dec4(x)
        x = self.up3(x)
        x = torch.cat([self.att3(x, x3), x], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([self.att2(x, x2), x], dim=1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([self.att1(x, x1), x], dim=1)
        x = self.dec1(x)

        logits = self.outc(x)
        if self.apply_sigmoid:
            return torch.sigmoid(logits)
        return logits

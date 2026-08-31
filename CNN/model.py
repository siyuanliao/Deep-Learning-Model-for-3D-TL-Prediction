"""Baseline conditional CNN architecture used for 3-D TL field prediction."""

import torch
import torch.nn as nn


class SimpleConvBlock(nn.Module):
    """Basic convolution block without circular padding or FiLM."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class SimpleResBlock(nn.Module):
    """Basic residual block."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 2 * channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 2 * channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class SimpleConditionEncoder(nn.Module):
    """Condition encoder for the 52-dimensional environmental vector."""
    def __init__(self, input_dim=52, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class SimpleConditionalCNN(nn.Module):
    """Baseline conditional CNN with the original architecture and parameter scale."""
    def __init__(self, x1_dim=52, in_ch=4, base_ch=64, cond_dim=128, num_blocks=8):
        super().__init__()

        self.cond_encoder = SimpleConditionEncoder(x1_dim, cond_dim, cond_dim)

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True)
        )

        self.cond_fc = nn.Sequential(
            nn.Linear(cond_dim, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch * 2, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch * 4, base_ch * 2),
        )

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(SimpleResBlock(base_ch))

        self.mid_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 4, in_ch, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        cond = self.cond_encoder(x1)
        cond_params = self.cond_fc(cond)
        gamma, beta = cond_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        features = self.input_conv(x2)
        features = features * (1 + gamma) + beta
        for block in self.res_blocks:
            features = block(features)
        features = self.mid_conv(features)
        output = self.output_conv(features)
        return output

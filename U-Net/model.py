import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConditionEncoder(nn.Module):
    """Encode the 52-dimensional environmental vector for conditioning at the U-Net bottleneck."""

    def __init__(self, input_dim=52, cond_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetDownBlock(in_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2,
        ])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SFUNet(nn.Module):
    """U-Net for 3-D underwater acoustic field prediction."""

    def __init__(self, x1_dim=52, in_ch=4, base_ch=64, cond_dim=256, dropout=0.0):
        super().__init__()
        self.cond_encoder = SimpleConditionEncoder(x1_dim, cond_dim)
        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.down1 = UNetDownBlock(base_ch, base_ch, dropout)
        self.down2 = UNetDownBlock(base_ch, base_ch * 2, dropout)
        self.down3 = UNetDownBlock(base_ch * 2, base_ch * 4, dropout)
        self.down4 = UNetDownBlock(base_ch * 4, base_ch * 8, dropout)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
        )
        self.cond_proj = nn.Linear(cond_dim, base_ch * 16)

        self.up4 = UNetUpBlock(base_ch * 16, base_ch * 8, dropout)
        self.up3 = UNetUpBlock(base_ch * 8, base_ch * 4, dropout)
        self.up2 = UNetUpBlock(base_ch * 4, base_ch * 2, dropout)
        self.up1 = UNetUpBlock(base_ch * 2, base_ch, dropout)
        self.output_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x1, x2):
        cond = self.cond_encoder(x1)
        x = self.init_conv(x2)

        d1 = self.down1(x)
        p1 = self.pool(d1)
        d2 = self.down2(p1)
        p2 = self.pool(d2)
        d3 = self.down3(p2)
        p3 = self.pool(d3)
        d4 = self.down4(p3)
        p4 = self.pool(d4)

        bottleneck = self.bottleneck(p4)
        cond_proj = self.cond_proj(cond)
        cond_proj = cond_proj.view(cond_proj.size(0), cond_proj.size(1), 1, 1)
        cond_proj = cond_proj.expand(-1, -1, 2, 15)
        bottleneck = bottleneck + cond_proj

        u4 = self.up4(bottleneck, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        return self.output_conv(u1)

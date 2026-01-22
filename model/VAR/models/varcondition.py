import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, change_channels=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class MultiStageNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # 上采样阶段 (16x16 -> 64x64)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 阶段0: 64x64 (128通道)
        self.block0_1 = ResidualBlock(128, 128)
        self.block0_2 = ResidualBlock(128, 128)

        # 阶段1: 32x32 (128/256通道)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block1_1 = ResidualBlock(128, 256)
        self.block1_2 = ResidualBlock(256, 256)

        # 阶段2: 16x16 (256/512通道)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block2_1 = ResidualBlock(256, 512)
        self.block2_2 = ResidualBlock(512, 512)

        # 阶段3: 8x8 (512/1024通道)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block3_1 = ResidualBlock(512, 1024)
        self.block3_2 = ResidualBlock(1024, 1024)

    def forward(self, x):
        outputs = []

        # 初始上采样
        x = self.upsample1(x)  # 320->256, 16x16->32x32
        x = self.upsample2(x)  # 256->128, 32x32->64x64

        # 阶段0: 64x64 (128通道)
        outputs.append(x)  # 输出1: [B,128,64,64]
        x = self.block0_1(x)
        outputs.append(x)  # 输出2: [B,128,64,64]
        x = self.block0_2(x)
        outputs.append(x)  # 输出3: [B,128,64,64]

        # 阶段1: 32x32 (128->256通道)
        x = self.downsample1(x)  # 128->128, 64x64->32x32
        outputs.append(x)  # 输出4: [B,128,32,32]
        x = self.block1_1(x)  # 128->256
        outputs.append(x)  # 输出5: [B,256,32,32]
        x = self.block1_2(x)
        outputs.append(x)  # 输出6: [B,256,32,32]

        # 阶段2: 16x16 (256->512通道)
        x = self.downsample2(x)  # 256->256, 32x32->16x16
        outputs.append(x)  # 输出7: [B,256,16,16]
        x = self.block2_1(x)  # 256->512
        outputs.append(x)  # 输出8: [B,512,16,16]
        x = self.block2_2(x)
        outputs.append(x)  # 输出9: [B,512,16,16]

        # 阶段3: 8x8 (512->1024通道)
        x = self.downsample3(x)  # 512->512, 16x16->8x8
        outputs.append(x)  # 输出10: [B,512,8,8]
        x = self.block3_1(x)  # 512->1024
        outputs.append(x)  # 输出11: [B,1024,8,8]
        x = self.block3_2(x)
        outputs.append(x)  # 输出12: [B,1024,8,8]

        return outputs,outputs[-1]

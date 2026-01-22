import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """轻量级跨注意力模块 - 支持不同空间尺寸的引导"""

    def __init__(self, in_channels, guide_channels, reduction=16):
        super().__init__()
        # 通道注意力部分
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels + guide_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力部分
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x, guide):
        # 调整引导特征到与x相同的空间尺寸
        if guide.shape[2:] != x.shape[2:]:
            guide_resized = F.interpolate(guide, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            guide_resized = guide

        # 通道注意力
        x_pool = F.adaptive_avg_pool2d(x, 1)
        g_pool = F.adaptive_avg_pool2d(guide_resized, 1)
        combined = torch.cat([x_pool, g_pool], dim=1)
        channel_att = self.channel_att(combined)

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))

        # 应用双重注意力
        return x * channel_att * spatial_att


class Down_Up_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Down_Up_Conv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


def crop_and_concat(upsampled, bypass):
    h1, w1 = upsampled.shape[2], upsampled.shape[3]
    h2, w2 = bypass.shape[2], bypass.shape[3]
    delta_h = h2 - h1
    delta_w = w2 - w1
    bypass_cropped = bypass[:, :, delta_h // 2:delta_h // 2 + h1, delta_w // 2:delta_w // 2 + w1]
    return torch.cat([upsampled, bypass_cropped], dim=1)


class UNet(nn.Module):
    def __init__(self, num_classes=3, guide_channels=3):
        super(UNet, self).__init__()
        # 引导特征处理路径
        self.guide_path = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(guide_channels, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        # 下采样阶段的跨注意力
        self.ca_down1 = CrossAttention(64, 64)
        self.ca_down2 = CrossAttention(128, 128)
        self.ca_down3 = CrossAttention(256, 256)
        self.ca_down4 = CrossAttention(512, 512)
        self.ca_down5 = CrossAttention(1024, 1024)

        # 上采样阶段的跨注意力
        self.ca_up4 = CrossAttention(512, 512)
        self.ca_up3 = CrossAttention(256, 256)
        self.ca_up2 = CrossAttention(128, 128)
        self.ca_up1 = CrossAttention(64, 64)

        # 原始UNet组件
        self.stage_down1 = Down_Up_Conv(3, 64)
        self.stage_down2 = Down_Up_Conv(64, 128)
        self.stage_down3 = Down_Up_Conv(128, 256)
        self.stage_down4 = Down_Up_Conv(256, 512)
        self.stage_down5 = Down_Up_Conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.stage_up4 = Down_Up_Conv(1024, 512)
        self.stage_up3 = Down_Up_Conv(512, 256)
        self.stage_up2 = Down_Up_Conv(256, 128)
        self.stage_up1 = Down_Up_Conv(128, 64)
        self.stage_out = nn.Conv2d(64, num_classes, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,guidance):
        # guidance = x
        # 处理引导特征 - 生成多尺度引导
        guide_features = []
        g = guidance
        for i, layer in enumerate(self.guide_path):
            g = layer(g)
            guide_features.append(g)

        # 下采样路径
        stage1 = self.stage_down1(x)
        stage1 = self.ca_down1(stage1, guide_features[0])  # 256x256引导
        x = self.maxpool(stage1)

        stage2 = self.stage_down2(x)
        stage2 = self.ca_down2(stage2, guide_features[1])  # 128x128引导
        x = self.maxpool(stage2)

        stage3 = self.stage_down3(x)
        stage3 = self.ca_down3(stage3, guide_features[2])  # 64x64引导
        x = self.maxpool(stage3)

        stage4 = self.stage_down4(x)
        stage4 = self.ca_down4(stage4, guide_features[3])  # 32x32引导
        x = self.maxpool(stage4)

        stage5 = self.stage_down5(x)
        stage5 = self.ca_down5(stage5, guide_features[4])  # 16x16引导

        # 上采样路径
        x = self.up4(stage5)
        x = crop_and_concat(x, stage4)
        x = self.stage_up4(x)
        x = self.ca_up4(x, guide_features[3])  # 32x32引导

        x = self.up3(x)
        x = crop_and_concat(x, stage3)
        x = self.stage_up3(x)
        x = self.ca_up3(x, guide_features[2])  # 64x64引导

        x = self.up2(x)
        x = crop_and_concat(x, stage2)
        x = self.stage_up2(x)
        x = self.ca_up2(x, guide_features[1])  # 128x128引导

        x = self.up1(x)
        x = crop_and_concat(x, stage1)
        x = self.stage_up1(x)
        x = self.ca_up1(x, guide_features[0])  # 256x256引导

        out = self.stage_out(x)
        return out

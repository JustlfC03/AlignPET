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


#
# class CrossAttention(nn.Module):
#     """轻量级跨注意力模块 - 支持不同空间尺寸的引导"""
#
#     def __init__(self, in_channels, guide_channels, reduction=16):
#         super().__init__()
#         # 通道注意力部分
#         self.channel_att = nn.Sequential(
#             nn.Conv2d(in_channels + guide_channels, in_channels // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, 1),
#
#         )
#
#         # # 空间注意力部分
#         # self.spatial_att = nn.Sequential(
#         #     nn.Conv2d(2, 1, kernel_size=7, padding=3),
#         #     nn.Sigmoid()
#         # )
#
#
#
#     def forward(self, x, guide):
#         # 调整引导特征到与x相同的空间尺寸
#         if guide.shape[2:] != x.shape[2:]:
#             guide_resized = F.interpolate(guide, size=x.shape[2:], mode='bilinear', align_corners=False)
#         else:
#             guide_resized = guide
#
#         # 通道注意力
#         combined = torch.cat((x, guide_resized), dim=1)
#         channel_att = self.channel_att(combined)
#
#
#
#         # 应用双重注意力
#         return channel_att

# class CrossAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=8):
#         """
#         Args:
#             in_channels: 输入特征图的通道数（两个输入通道数相同）
#             reduction_ratio: QKV通道缩减比例（默认8）
#         """
#         super().__init__()
#         self.reduced_channels = max(1, in_channels // reduction_ratio)

#         # 共享的轻量化QKV投影层（使用1x1卷积）
#         self.query_proj = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
#         self.key_proj = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
#         self.value_proj = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)

#         # 输出重建层（恢复原始通道数）
#         self.output_proj = nn.Conv2d(self.reduced_channels, in_channels, kernel_size=1)

#         # 层归一化
#         self.norm = nn.LayerNorm(in_channels)

#     def forward(self, x1, x2):
#         """
#         Args:
#             x1: 第一个输入 [B, C, H1, W1]
#             x2: 第二个输入 [B, C, H2, W2]
#         Returns:
#             与x1相同空间尺寸的输出 [B, C, H1, W1]
#         """
#         B, C, H1, W1 = x1.shape
#         _, _, H2, W2 = x2.shape

#         # 生成QKV [B, C_red, H, W]
#         q = self.query_proj(x1)  # [B, C_red, H1, W1]
#         k = self.key_proj(x2)  # [B, C_red, H2, W2]
#         v = self.value_proj(x2)  # [B, C_red, H2, W2]

#         # 展平空间维度 [B, C_red, N]
#         q = q.view(B, self.reduced_channels, -1)  # [B, C_red, H1*W1]
#         k = k.view(B, self.reduced_channels, -1)  # [B, C_red, H2*W2]
#         v = v.view(B, self.reduced_channels, -1)  # [B, C_red, H2*W2]

#         # 高效矩阵乘法 (避免大矩阵转置)
#         # 1. 计算注意力分数 [B, H1*W1, H2*W2]
#         attn = torch.einsum('bci,bcj->bij', q, k)  # 等价于 q^T * k
#         attn = attn / (self.reduced_channels ** 0.5)
#         attn = F.softmax(attn, dim=-1)

#         # 2. 注意力加权 [B, C_red, H1*W1]
#         out = torch.einsum('bij,bcj->bci', attn, v)

#         # 恢复空间结构 [B, C_red, H1, W1]
#         out = out.view(B, self.reduced_channels, H1, W1)

#         # 通道恢复 [B, C, H1, W1]
#         out = self.output_proj(out)

#         # 残差连接 + 层归一化
#         out = out + x1
#         out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
#         out = self.norm(out)
#         return out.permute(0, 3, 1, 2)  # 恢复 [B, C, H, W]

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


class UNet2(nn.Module):
    def __init__(self, num_classes=3, guide_channels=320):
        super(UNet2, self).__init__()
        # 引导特征处理路径 (保持不变)
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
                nn.Conv2d(512, 320, 3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        # 编码器特征适配层 (统一到320通道)
        self.encoder_adapters = nn.ModuleList([
            self._make_adapter(64, 320),  # stage1: 256x256
            self._make_adapter(128, 320),  # stage2: 128x128
            self._make_adapter(256, 320),  # stage3: 64x64
            self._make_adapter(512, 320),  # stage4: 32x32
            self._make_adapter(1024, 320)  # stage5: 16x16
        ])

        # 解码器引导融合适配层
        self.decoder_adapters = nn.ModuleList([
            self._make_adapter(320, 512),  # up4: 32x32
            self._make_adapter(320, 256),  # up3: 64x64
            self._make_adapter(320, 128),  # up2: 128x128
            self._make_adapter(320, 64)  # up1: 256x256
        ])

        # 跨注意力模块 (更新以适应融合后的引导)
        self.ca_down1 = CrossAttention(64, 64)
        self.ca_down2 = CrossAttention(128, 128)
        self.ca_down3 = CrossAttention(256, 256)
        self.ca_down4 = CrossAttention(512, 512)
        self.ca_down5 = CrossAttention(1024, 320)

        self.ca_up4 = CrossAttention(512, 512)
        self.ca_up3 = CrossAttention(256, 256)
        self.ca_up2 = CrossAttention(128, 128)
        self.ca_up1 = CrossAttention(64, 64)

        # 原始UNet组件 (保持不变)
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

    def _make_adapter(self, in_channels, out_channels):
        """创建适配层: 1x1卷积调整通道数"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, guidance):
        # 处理外部引导特征
        guide_features = []
        g = guidance
        for layer in self.guide_path:
            g = layer(g)
            guide_features.append(g)

        # 编码器路径 + 创建编码器引导特征
        encoder_guides = []

        stage1 = self.stage_down1(x)  # 64,256,256
        enc_guide1 = self.encoder_adapters[0](stage1)  # 320,256,256
        encoder_guides.append(enc_guide1)
        x = self.maxpool(stage1)

        stage2 = self.stage_down2(x)  # 128,128,128
        enc_guide2 = self.encoder_adapters[1](stage2)  # 320,128,128
        encoder_guides.append(enc_guide2)
        x = self.maxpool(stage2)

        stage3 = self.stage_down3(x)  # 256,64,64
        enc_guide3 = self.encoder_adapters[2](stage3)  # 320,64,64
        encoder_guides.append(enc_guide3)
        x = self.maxpool(stage3)

        stage4 = self.stage_down4(x)  # 512,32,32
        enc_guide4 = self.encoder_adapters[3](stage4)  # 320,32,32
        encoder_guides.append(enc_guide4)
        x = self.maxpool(stage4)

        stage5 = self.stage_down5(x)  # 1024,16,16
        enc_guide5 = self.encoder_adapters[4](stage5)  # 320,16,16
        encoder_guides.append(enc_guide5)

        # 应用外部引导到编码器路径
        stage1 = self.ca_down1(stage1, guide_features[0])
        stage2 = self.ca_down2(stage2, guide_features[1])
        stage3 = self.ca_down3(stage3, guide_features[2])
        stage4 = self.ca_down4(stage4, guide_features[3])
        stage5 = self.ca_down5(stage5, guide_features[4])

        # 上采样路径 - 使用最远距离编码器引导
        # up4阶段: 使用stage1引导 (最远距离)
        x = self.up4(stage5)  # 1024->512, 16->32
        x = crop_and_concat(x, stage4)  # 512+512=1024, 32x32
        x = self.stage_up4(x)  # 1024->512, 32x32

        # 选择并适配最远编码器引导 (stage1)
        distant_guide = self.decoder_adapters[0](encoder_guides[0])  # 320->512, 256x256
        # 下采样到当前尺寸 (256x256 -> 32x32)
        distant_guide = F.adaptive_avg_pool2d(distant_guide, (x.size(2), (x.size(3))))
        x = self.ca_up4(x, distant_guide)  # 跨注意力融合

        # up3阶段: 使用stage2引导 (次远距离)
        x = self.up3(x)  # 512->256, 32->64
        x = crop_and_concat(x, stage3)  # 256+256=512, 64x64
        x = self.stage_up3(x)  # 512->256, 64x64

        # 选择并适配最远编码器引导 (stage2)
        distant_guide = self.decoder_adapters[1](encoder_guides[1])  # 320->256, 128x128
        # 下采样到当前尺寸 (128x128 -> 64x64)
        distant_guide = F.adaptive_avg_pool2d(distant_guide, (x.size(2), (x.size(3))))
        x = self.ca_up3(x, distant_guide)

        # up2阶段: 使用stage3引导
        x = self.up2(x)  # 256->128, 64->128
        x = crop_and_concat(x, stage2)  # 128+128=256, 128x128
        x = self.stage_up2(x)  # 256->128, 128x128

        # 选择并适配最远编码器引导 (stage3)
        distant_guide = self.decoder_adapters[2](encoder_guides[2])  # 320->128, 64x64
        # 上采样到当前尺寸 (64x64 -> 128x128)
        distant_guide = F.interpolate(distant_guide, size=(x.size(2), (x.size(3))), mode='bilinear', align_corners=True)
        x = self.ca_up2(x, distant_guide)

        # up1阶段: 使用stage4引导
        x = self.up1(x)  # 128->64, 128->256
        x = crop_and_concat(x, stage1)  # 64+64=128, 256x256
        x = self.stage_up1(x)  # 128->64, 256x256

        # 选择并适配最远编码器引导 (stage4)
        distant_guide = self.decoder_adapters[3](encoder_guides[3])  # 320->64, 32x32
        # 上采样到当前尺寸 (32x32 -> 256x256)
        distant_guide = F.interpolate(distant_guide, size=(x.size(2), (x.size(3))),
                                                           mode='bilinear', align_corners=True)
        x = self.ca_up1(x, distant_guide)

        out = self.stage_out(x)
        return out